from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from chronos.chronos2 import Chronos2Model, Chronos2Pipeline


@dataclass
class Chronos2PipelineCustom:
    """
    Minimal wrapper around Chronos2Pipeline to standardize predict semantics for TSFM Lens.
    """

    model_name: str | None = None
    model: Chronos2Model | None = None
    pipeline: Chronos2Pipeline | None = None
    device: str | torch.device = "cpu"

    def __post_init__(self) -> None:
        if self.pipeline is None:
            if self.model is None:
                if self.model_name is None:
                    raise ValueError("Provide either `model`, `pipeline`, or `model_name`.")
                self.model = Chronos2Model.from_pretrained(self.model_name)  # type: ignore[attr-defined]
            self.model = self.model.to(self.device)  # type: ignore[assignment]
            self.pipeline = Chronos2Pipeline(self.model)
        else:
            # ensure model attribute is set for lens usage
            self.model = self.pipeline.model  # type: ignore[assignment]

        self.quantiles = self.pipeline.quantiles  # type: ignore[attr-defined]

    @torch.no_grad()
    def predict(
        self,
        context: torch.Tensor | Sequence[torch.Tensor],
        prediction_length: int | None = None,
        batch_size: int = 256,
        limit_prediction_length: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run inference and return point forecasts (median quantile).

        Accepts context of shape (batch, time) or (batch, time, dim). If dim>1,
        multivariate forecasting is used. Returns tensor of shape
        (batch, prediction_length) for univariate inputs, or
        (batch, dim, prediction_length) for multivariate.
        """

        if isinstance(context, (list, tuple)):
            context = torch.stack(list(context), dim=0)
        if context.dim() == 2:  # type: ignore
            context = context.unsqueeze(-1)  # type: ignore
        if context.dim() != 3:  # type: ignore
            raise ValueError(
                f"Expected context of shape (batch, time) or (batch, time, dim); got {tuple(context.shape)}"  # type: ignore
            )

        # Chronos2 expects (batch, dim, time)
        context = context.permute(0, 2, 1)  # type: ignore
        context_np = context.cpu().numpy()

        quantiles, mean_list = self.pipeline.predict_quantiles(  # type: ignore[attr-defined]
            context_np,
            prediction_length=prediction_length,
            batch_size=batch_size,
            limit_prediction_length=limit_prediction_length,
            **kwargs,
        )

        # mean_list elements shaped (n_variates, prediction_length)
        preds = torch.stack([torch.as_tensor(arr) for arr in mean_list], dim=0)
        quantiles = torch.stack([torch.as_tensor(arr) for arr in quantiles], dim=0)
        # Reorder back to (batch, prediction_length[, dim])
        preds = preds.permute(0, 2, 1)  # (batch, prediction_length, dim)
        if preds.shape[-1] == 1:
            preds = preds.squeeze(-1)

        quantiles = quantiles.permute(0, 3, 2, 1)  # (batch, quantiles, prediction_length, dim)
        if quantiles.shape[-1] == 1:
            quantiles = quantiles.squeeze(-1)

        return preds.to(context.device), quantiles.to(context.device)

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        device: torch.device = torch.device("cpu"),
        **kwargs,
    ) -> Chronos2PipelineCustom:
        model = Chronos2Model.from_pretrained(model_name, **kwargs)  # type: ignore[attr-defined]
        model = model.to(device)
        return cls(model_name=model_name, model=model, device=device)
