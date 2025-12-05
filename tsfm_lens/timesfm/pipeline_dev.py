"""
TimesFM pipeline for in-context analysis and hook-based inspection.

This mirrors the ergonomics of Chronos pipelines but targets the TimesFM
architecture (TimesFM 2.5 Torch implementation). It focuses on:
- Converting raw context series into patch inputs and masks
- Running decode to obtain point and quantile forecasts
- Optionally exposing internal layer inputs/outputs via hooks (see CircuitLens)

Design notes:
- We avoid depending on the full TimesFM reference compile path; instead we
  call the model's `decode` directly to obtain forecasts. This makes the
  pipeline suitable for analysis and lightweight tests.
- The pipeline accepts any module that exposes the attributes used by
  `TimesFM_2p5_200M_torch_module`:
  - p (input patch len), o (output patch len), os (output quantile len)
  - x (num transformer layers), h (num heads), hd (head dim), q (num quantiles)
  - tokenizer (nn.Module), stacked_xf (nn.ModuleList),
    output_projection_point (nn.Module), output_projection_quantiles (nn.Module)
  - methods: decode(horizon, inputs, masks) and forward(inputs, masks, decode_caches=None)

If you load an actual TimesFM model from the bundled references, you can pass
`timesfm_references.timesfm_2p5_torch.TimesFM_2p5_200M_torch().model` as the
`model` argument to this pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

import timesfm


def _pad_context_1d(
    context: torch.Tensor,
    patch_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Front-pad raw context [B, T] to a multiple of patch_len and return 1D inputs/masks.

    Returns
    -------
    inputs_1d : torch.Tensor
        Shape: [B, T_pad]
    masks_1d : torch.Tensor (bool)
        True denotes padded (masked) elements. Shape: [B, T_pad]
    """
    if context.ndim != 2:
        raise ValueError("context must be rank-2 tensor [B, T]")

    batch_size, T = context.shape
    remainder = T % patch_len
    if remainder == 0:
        padded = context
        mask_1d = torch.zeros_like(context, dtype=torch.bool)
    else:
        pad_len = patch_len - remainder
        pad_vals = torch.zeros(batch_size, pad_len, device=context.device, dtype=context.dtype)
        padded = torch.cat([pad_vals, context], dim=-1)
        front_mask = torch.ones(batch_size, pad_len, device=context.device, dtype=torch.bool)
        tail_mask = torch.zeros(batch_size, T, device=context.device, dtype=torch.bool)
        mask_1d = torch.cat([front_mask, tail_mask], dim=-1)
    return padded, mask_1d


def _to_patches(inputs_1d: torch.Tensor, masks_1d: torch.Tensor, patch_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Reshape 1D inputs/masks to patched [B, num_patches, patch_len]."""
    batch_size, T_pad = inputs_1d.shape
    num_patches = T_pad // patch_len
    return (
        inputs_1d.view(batch_size, num_patches, patch_len),
        masks_1d.view(batch_size, num_patches, patch_len),
    )


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
    model: torch.nn.Module = None

    def __post_init__(self):
        if self.model is None:
            tm = timesfm.TimesFM_2p5_200M_torch.from_pretrained(self.model_name, device=self.device_map)
            self.model = tm.model
        if self.device_map is not None:
            self.model.to(self.device_map)

    @property
    def p(self) -> int:
        return int(self.model.p)

    @property
    def o(self) -> int:
        return int(self.model.o)

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
        self, context: torch.Tensor | list[torch.Tensor], horizon: int, include_mean_forecast: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forecast using the module's `decode`, returning point and quantile predictions.

        NOTE: we modify this method to return full_forecast as shape (batch_size, num_quantiles, horizon)
                in order to match the convention used for Chronos and Chronos-Bolt when handling probabilistic samples and quantiles respectively.

        Parameters
        ----------
        context : torch.Tensor | list[torch.Tensor]
            Input context tensor of shape [batch_size, context_length] or list of tensors
        horizon : int
            Number of steps to forecast
        include_mean_forecast : bool, default=True
            Whether to include the mean forecast in the quantile predictions. If False,
            removes the mean forecast which at index 0 of the full_forecast, by (timesfm2p5 convention).

        Returns
        -------
        point : torch.Tensor
            Point predictions with shape [batch_size, horizon]
        quantiles : torch.Tensor
            Quantile predictions with shape [batch_size, num_quantiles, horizon]
        """
        context_tensor = self._prepare_context_tensor(context)
        inputs_1d, masks_1d = _pad_context_1d(context_tensor, self.p)

        inputs_1d = inputs_1d.to(self.model_device, dtype=torch.float32)
        masks_1d = masks_1d.to(self.model_device)

        # TimesFM decode expects 1D [B, T_pad] inputs/masks
        pf_outputs, quantile_spreads, ar_outputs = self.model.decode(horizon, inputs_1d, masks_1d)

        # Assemble full forecast to [B, horizon, q]
        batch_size = context_tensor.shape[0]
        to_cat = [pf_outputs[:, -1, ...]]
        if ar_outputs is not None:
            to_cat.append(ar_outputs.reshape(batch_size, -1, self.num_quantiles))
        full_forecast = torch.cat(to_cat, dim=1)[:, :horizon, :]

        # Point forecast is the decode index (usually median, idx 5)
        decode_index = int(getattr(self.model, "aridx", 5))
        # print(f"decode_index: {decode_index}")
        point = full_forecast[..., decode_index]
        # print(f"first 5 points of point: {point[0, :5]}")
        # print(f"first 5 points of full_forecast: {full_forecast[0, :5, :]}")

        if not include_mean_forecast:
            # the timesfm2p5 convention is to have the mean forecast at index 0
            full_forecast = full_forecast[..., 1:]

        # reorder full_forecast from (batch_size, horizon, num_quantiles) to (batch_size, num_quantiles, horizon)
        full_forecast_reordered = full_forecast.permute(0, 2, 1)

        # Return on CPU float32 for parity with Chronos wrappers
        return (
            point.to(dtype=torch.float32, device=torch.device("cpu")),
            full_forecast_reordered.to(dtype=torch.float32, device=torch.device("cpu")),
        )

    def predict(
        self,
        context: torch.Tensor | list[torch.Tensor],
        prediction_length: int,
    ) -> torch.Tensor:
        point, _ = self.predict_point_and_quantiles(context=context, horizon=prediction_length)
        return point

    # ---------- Introspection helpers ----------
    def forward_once(
        self,
        context: torch.Tensor | list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Execute a single forward pass over the patched context to expose
        (input_embeddings, output_embeddings, output_ts, output_quantile_spread).
        Shapes follow the underlying model implementation.
        """
        context_tensor = self._prepare_context_tensor(context)
        inputs_1d, masks_1d = _pad_context_1d(context_tensor, self.p)
        inputs_patched, masks_patched = _to_patches(inputs_1d, masks_1d, self.p)
        inputs_patched = inputs_patched.to(self.model_device, dtype=torch.float32)
        masks_patched = masks_patched.to(self.model_device)
        (inp_emb, out_emb, out_ts, out_qs), _ = self.model(inputs_patched, masks_patched, None)
        return inp_emb, out_emb, out_ts, out_qs

    # ---------- Utilities ----------
    def _prepare_context_tensor(self, context: torch.Tensor | list[torch.Tensor]) -> torch.Tensor:
        if isinstance(context, list):
            if len(context) == 0:
                raise ValueError("context list must be non-empty")
            context_tensor = torch.stack(
                [c if isinstance(c, torch.Tensor) else torch.tensor(c) for c in context], dim=0
            )
        else:
            context_tensor = context
            if context_tensor.ndim == 1:
                context_tensor = context_tensor.unsqueeze(0)
        if context_tensor.ndim != 2:
            raise ValueError("context must be [B, T] or list of 1D tensors")
        return context_tensor
