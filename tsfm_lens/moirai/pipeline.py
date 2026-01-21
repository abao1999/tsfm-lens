from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd
import torch
from gluonts.dataset.common import ListDataset
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
        Returns:
            Torch tensor shaped (batch, num_samples, prediction_length[, dim]) or
            (batch, prediction_length[, dim]) when ``num_samples==1``.
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
            pad = torch.zeros(
                context.size(0), ctx_len - context.size(1), context.size(2), device=device, dtype=context.dtype
            )
            context = torch.cat([pad, context], dim=1)
        elif context.size(1) > ctx_len:
            context = context[:, -ctx_len:, :]

        # Handle overrides
        actual_pred_len = prediction_length or self.model.hparams.prediction_length
        actual_num_samples = num_samples or self.model.hparams.num_samples
        actual_patch_size: int | str = patch_size or self.model.hparams.patch_size
        actual_target_dim = context.size(-1)
        actual_batch_size = context.size(0)

        forecasts = []
        with self.model.hparams_context(
            prediction_length=actual_pred_len,
            target_dim=actual_target_dim,
            context_length=ctx_len,
            patch_size=actual_patch_size,
            num_samples=actual_num_samples,
        ):
            data_entries = [
                {
                    "target": context[i].squeeze().tolist(),
                    "start": pd.Period("2020-01-01", freq="1D"),
                }
                for i in range(context.size(0))
            ]
            dataset = ListDataset(data_entries, freq="1D")
            predictor = self.model.create_predictor(batch_size=actual_batch_size, device=device)
            for forecast in predictor.predict(dataset, num_samples=actual_num_samples):
                forecasts.append(torch.as_tensor(forecast.samples, device=device, dtype=context.dtype))

        if not forecasts:
            raise RuntimeError("No forecasts were produced by the predictor.")

        batch_samples = torch.stack(forecasts, dim=0)

        # Keep a consistent return shape: (batch, num_samples, horizon[, dim]).
        if actual_num_samples == 1:
            batch_samples = batch_samples.squeeze(1)
        if batch_samples.dim() >= 3 and batch_samples.size(-1) == 1:
            batch_samples = batch_samples.squeeze(-1)
        return batch_samples
