from __future__ import annotations

from dataclasses import dataclass

import torch

import timesfm
from timesfm import TimesFM_2p5_200M_torch


@dataclass
class TimesFMPipelinetsfm_lens:
    """
    Lightweight pipeline wrapper around a TimesFM 2.5 torch module.

    Parameters
    ----------
    model : torch.nn.Module
        A model exposing the TimesFM module interface (see the file docstring).
    device : str | torch.device | None
        Optional device override. Defaults to model.device if available.
    dtype : torch.dtype | None
        Optional dtype override. Defaults to torch.float32.
    """

    model_name: str
    device_map: torch.device | str | None = None
    model: TimesFM_2p5_200M_torch | None = None

    def __post_init__(self):
        torch.set_float32_matmul_precision("high")
        if self.model is None:
            print(f"Loading model from {self.model_name} on device: {self.device_map}")
            self.model = TimesFM_2p5_200M_torch.from_pretrained(self.model_name, device=self.device_map)
        if self.device_map is not None:
            self.model.model.to(self.device_map)
            print(f"Compiling model on device: {self.model_device}")

        self.model.compile(
            timesfm.ForecastConfig(
                max_context=2048,  # 15360,
                max_horizon=1024,
                infer_is_positive=True,
                use_continuous_quantile_head=True,
                fix_quantile_crossing=True,
                force_flip_invariance=True,
                return_backcast=False,
                normalize_inputs=True,
                per_core_batch_size=32,
            )
        )
        self.model.model.eval()

    @property
    def num_quantiles(self) -> int:
        return int(self.model.q)

    @property
    def model_device(self) -> torch.device:
        if hasattr(self.model, "device"):
            return self.model.device  # type: ignore[return-value]
        if isinstance(self.device_map, torch.device):
            return self.device_map
        if isinstance(self.device_map, str):
            return torch.device(self.device_map)
        return torch.device("cpu")

    def predict_point_and_quantiles(
        self,
        context: torch.Tensor | list[torch.Tensor],
        prediction_length: int,
        include_mean_forecast: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate point and quantile forecasts using the TimesFM model.

        NOTE: we modify this method to return quantile_forecast as shape (batch_size, num_quantiles, horizon)
                in order to match the convention used for Chronos and Chronos-Bolt when handling probabilistic samples and quantiles respectively.

        Parameters
        ----------
        context : torch.Tensor | list[torch.Tensor]
            Input context tensor of shape [batch_size, context_length] or list of tensors.
            Each tensor represents a time series to forecast.
        prediction_length : int
            Number of steps to forecast into the future
        include_mean_forecast : bool, default=False
            Whether to include the mean forecast in the quantile predictions. If False,
            removes the mean forecast which is at index 0 by TimesFM convention.

        Returns
        -------
        point_forecast : torch.Tensor
            Point predictions with shape [batch_size, prediction_length]
        quantile_forecast : torch.Tensor
            Quantile predictions with shape [batch_size, num_quantiles, prediction_length].
            The quantiles dimension follows TimesFM's convention, with mean forecast optionally
            included at index 0 based on include_mean_forecast parameter.
        """
        # convert context to list[np.ndarray] (TimesFM expects list of 1D arrays)
        if isinstance(context, torch.Tensor):
            context_np = context.cpu().numpy()
            # If 2D array (batch_size, context_length), convert to list of 1D arrays
            if context_np.ndim == 2:
                context = [context_np[i] for i in range(context_np.shape[0])]
            elif context_np.ndim == 1:
                context = [context_np]
            else:
                raise ValueError(f"Unexpected context shape: {context_np.shape}")
        elif isinstance(context, list):
            context_list = []
            # If 2D array (batch_size, context_length), convert to list of 1D arrays
            for ctx in context:
                if ctx.ndim == 2:
                    context_list.extend([ctx[i] for i in range(ctx.shape[0])])
                else:
                    context_list.append(ctx.squeeze())
            context = context_list

        print(f"context: {len(context)} series. ")

        point_forecast, quantile_forecast = self.model.forecast(
            horizon=prediction_length,
            inputs=context,
        )

        # convert point_forecast and quantile_forecast to torch.Tensor
        point_forecast = torch.tensor(point_forecast)
        quantile_forecast = torch.tensor(quantile_forecast)

        if not include_mean_forecast:
            # the timesfm2p5 convention is to have the mean forecast at index 0
            quantile_forecast = quantile_forecast[..., 1:]

        # reorder full_forecast from (batch_size, horizon, num_quantiles) to (batch_size, num_quantiles, horizon)
        quantile_forecast_reordered = quantile_forecast.permute(0, 2, 1)

        return point_forecast, quantile_forecast_reordered

    def predict(self, context: torch.Tensor | list[torch.Tensor], prediction_length: int) -> torch.Tensor:
        point_forecast, _ = self.predict_point_and_quantiles(context=context, prediction_length=prediction_length)
        return point_forecast
