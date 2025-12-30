from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings
from typing import Sequence
import sys

import torch

from uni2ts.model.moirai import MoiraiForecast, MoiraiModule


@dataclass
class MoiraiPipelineCustom:
    """
    Lightweight wrapper around Uni2TS Moirai models to standardize prediction APIs
    for TSFM Lens.

    Parameters
    ----------
    model_name:
        Hugging Face repository id for a pre-trained MoiraiModule. Ignored if
        ``module`` or ``model`` is provided directly.
    context_length / prediction_length:
        Window sizes used when constructing the ``MoiraiForecast`` wrapper.
    patch_size:
        Patch size handed to the forecast module. Defaults to the first entry in
        the underlying module's ``patch_sizes`` if not provided. Set to an
        integer to avoid the additional validation-loss pass required by
        ``patch_size='auto'``.
    num_samples:
        Number of forecast samples to draw per series.
    target_dim:
        Number of variates in the target. If the provided context has a
        different trailing dimension, the pipeline will temporarily override the
        forecast hyperparameters to match it.
    device:
        Device on which the model should run.
    module:
        Optional pre-loaded ``MoiraiModule``.
    model:
        Optional pre-built ``MoiraiForecast``. If provided, the other model
        construction kwargs are ignored.
    """

    model_name: str | None = None
    context_length: int = 512
    prediction_length: int = 96
    patch_size: int | str | None = None
    num_samples: int = 100
    target_dim: int = 1
    device: str | torch.device = "cpu"
    module: MoiraiModule | None = None
    model: MoiraiForecast | None = None

    def __post_init__(self) -> None:
        if self.model is None:
            if self.module is None:
                if self.model_name is None:
                    raise ValueError("Provide either `model`, `module`, or `model_name` to build the pipeline.")
                self.module = MoiraiModule.from_pretrained(self.model_name)

            if self.patch_size is None:
                # Prefer the module default; fall back to 16 if unavailable.
                self.patch_size = getattr(self.module, "patch_sizes", [16])[0]

            self.model = MoiraiForecast(
                module=self.module,
                prediction_length=self.prediction_length,
                context_length=self.context_length,
                patch_size=self.patch_size,
                num_samples=self.num_samples,
                target_dim=self.target_dim,
                feat_dynamic_real_dim=0,
                past_feat_dynamic_real_dim=0,
            )

        # Move to device (LightningModule.to returns the module)
        self.model = self.model.to(self.device)

    @torch.no_grad()
    def predict(
        self,
        context: torch.Tensor | Sequence[torch.Tensor],
        prediction_length: int | None = None,
        num_samples: int | None = None,
        patch_size: int | str | None = None,
        future_target: torch.Tensor | None = None,
        future_observed_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Generate forecasts given a history tensor.

        Args:
            context: Tensor of shape (batch, time) or (batch, time, dim). Lists
                are stacked along the batch dimension.
            prediction_length: Optional override for the forecast horizon.
            num_samples: Optional override for the number of sampled trajectories.
            patch_size: Optional override for the patch size. If set to ``"auto"``
                without ``future_target``, the pipeline falls back to the first
                available patch size to avoid needing ground-truth futures for
                validation-loss computation.
            future_target: Optional ground-truth horizon used only when
                ``patch_size='auto'`` to score candidate patch sizes.
            future_observed_mask: Optional boolean mask aligned with
                ``future_target``; defaults to all observed.
        """

        self.model.eval()

        # Normalize inputs
        if isinstance(context, (list, tuple)):
            context = torch.stack(list(context), dim=0)
        if context.dim() == 2:
            context = context.unsqueeze(-1)
        if context.dim() != 3:
            raise ValueError(f"Expected context shape (batch, time) or (batch, time, dim); got {tuple(context.shape)}")

        device = next(self.model.parameters()).device
        context = context.to(device)

        # Align context length with the model expectation
        ctx_len = self.model.hparams.context_length
        if context.size(1) < ctx_len:
            pad = torch.zeros(context.size(0), ctx_len - context.size(1), context.size(2), device=device, dtype=context.dtype)
            context = torch.cat([pad, context], dim=1)
        elif context.size(1) > ctx_len:
            context = context[:, -ctx_len:, :]

        # Handle overrides
        actual_pred_len = prediction_length or self.model.hparams.prediction_length
        actual_num_samples = num_samples or self.model.hparams.num_samples
        actual_patch_size: int | str = patch_size or self.model.hparams.patch_size
        actual_target_dim = context.size(-1)

        # Optional future tensors (only used for patch_size='auto')
        past_target_for_model = context
        past_observed_target = torch.ones_like(context, dtype=torch.bool)
        past_is_pad = torch.zeros(context.size(0), context.size(1), device=device, dtype=torch.bool)

        if actual_patch_size == "auto":
            if future_target is None:
                fallback = getattr(self.model.module, "patch_sizes", [16])[0]
                warnings.warn(
                    "patch_size='auto' requires future_target for validation loss; "
                    f"falling back to patch_size={fallback}.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                actual_patch_size = fallback
            else:
                if future_target.dim() == 2:
                    future_target = future_target.unsqueeze(-1)
                future_target = future_target.to(device)
                if future_observed_mask is None:
                    future_observed_mask = torch.ones_like(future_target, dtype=torch.bool, device=device)
                elif future_observed_mask.shape != future_target.shape:
                    raise ValueError(
                        f"future_observed_mask shape {tuple(future_observed_mask.shape)} "
                        f"must match future_target shape {tuple(future_target.shape)}"
                    )
                future_observed_mask = future_observed_mask.to(device)
                if future_target.size(1) < actual_pred_len:
                    pad = torch.zeros(
                        future_target.size(0),
                        actual_pred_len - future_target.size(1),
                        future_target.size(2),
                        device=device,
                        dtype=future_target.dtype,
                    )
                    pad_mask = torch.zeros_like(pad, dtype=torch.bool)
                    warnings.warn(
                        "future_target shorter than prediction_length; padding the remainder with zeros.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    future_target = torch.cat([future_target, pad], dim=1)
                    future_observed_mask = torch.cat([future_observed_mask, pad_mask], dim=1)
                elif future_target.size(1) > actual_pred_len:
                    warnings.warn(
                        "future_target longer than prediction_length; truncating to the requested horizon.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    future_target = future_target[:, :actual_pred_len, :]
                    future_observed_mask = future_observed_mask[:, :actual_pred_len, :]

                past_target_for_model = torch.cat([context, future_target], dim=1)
                past_observed_target = torch.cat([past_observed_target, future_observed_mask], dim=1)
                future_pad = torch.zeros(future_target.size(0), future_target.size(1), device=device, dtype=torch.bool)
                past_is_pad = torch.cat([past_is_pad, future_pad], dim=1)

        with self.model.hparams_context(
            prediction_length=actual_pred_len,
            patch_size=actual_patch_size,
            num_samples=actual_num_samples,
            target_dim=actual_target_dim,
            context_length=ctx_len,
        ):
            preds = self.model(
                past_target=past_target_for_model,
                past_observed_target=past_observed_target,
                past_is_pad=past_is_pad,
                num_samples=actual_num_samples,
            )

        # preds: (batch, samples, horizon) or (batch, samples, horizon, dim)
        return preds
