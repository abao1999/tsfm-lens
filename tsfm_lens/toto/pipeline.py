"""
Our implementation builds on the Toto codebase. Header from the Toto codebase:
    Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.

    This product includes software developed at Datadog (https://www.datadoghq.com/)
    Copyright 2025 Datadog, Inc.
"""

from dataclasses import dataclass

import numpy as np
import torch
from einops import rearrange, repeat
from jaxtyping import Bool, Float, Int
from toto.data.util.dataset import (
    MaskedTimeseries,
    replace_extreme_values,
)
from toto.inference.forecaster import TotoForecaster
from toto.model.backbone import TotoBackbone
from toto.model.toto import Toto


@dataclass
class TotoForecastertsfm_lens(TotoForecaster):
    """
    A forecaster class for the Toto model that handles autoregressive decoding for time series forecasting.

    This class wraps a TotoBackbone model and provides methods to generate forecasts for time series data.
    The forecasting process uses an autoregressive decoding algorithm:

    1. The model first processes the entire input context (historical data)
    2. For each future time step:
       - The model generates a distribution over possible values
       - Either the mean or random samples are drawn from this distribution
       - The generated value(s) are appended to the input sequence
       - The process repeats with this extended sequence

    When generating multiple samples (num_samples > 1), the model creates separate trajectories for each sample:
    - Each trajectory starts with the same historical context
    - As sampling progresses, each trajectory evolves independently
    - This results in num_samples different possible future paths
    - Samples can be processed in batches (samples_per_batch) to manage memory usage

    NOTE: we modify this method to return samples in forecast() as shape (batch_size, num_samples, num_variates, horizon)
            in order to match the convention used for Chronos and Chronos-Bolt when handling probabilistic samples and quantiles respectively.

    The forecaster efficiently reuses computation from the context processing phase using a key-value cache,
    which stores intermediate transformer attention states to avoid redundant computation.

    The forecaster handles data preprocessing, including padding to match the model's patch size,
    and postprocessing to format the outputs as a Forecast object containing means and optional samples.
    """

    model_name: str
    device_map: torch.device | str | None = None
    model: TotoBackbone | None = None

    def __post_init__(self):
        if self.model is None:
            model_wrapper = Toto.from_pretrained(self.model_name)
            if self.device_map is not None:
                model_wrapper.to(self.device_map)
            model_wrapper.compile()  # Uses Torch's JIT compilation for better performance
            self.model = model_wrapper.model
            self.model_wrapper = model_wrapper
        else:
            self.model_wrapper = self.model
            self.model = self.model.model  # type: ignore[assignment]

    def predict(
        self,
        context: torch.Tensor,
        prediction_length: int,
        num_samples: int | None = None,
        samples_per_batch: int = 10,
        use_kv_cache: bool = True,
    ) -> torch.Tensor:
        context = context.to(self.model.device)
        print(f"context shape: {context.shape}")

        ### Dummy variables for future forward compatibility ###
        # Prepare timestamp information (optional, but expected by API; not used by the current model release)
        timestamp_seconds = torch.zeros_like(context).to(self.model.device)
        # time_interval_seconds = torch.full_like((3,), 60 * 15)  # 15-minute intervals
        time_interval_seconds = torch.full((context.shape[0],), 60 * 15).to(self.model.device)  # 15-minute intervals

        # Create a MaskedTimeseries object
        inputs = MaskedTimeseries(
            series=context,
            padding_mask=torch.full_like(context, True, dtype=torch.bool),
            id_mask=torch.zeros_like(context),
            timestamp_seconds=timestamp_seconds,
            time_interval_seconds=time_interval_seconds,
        )

        # forecast.samples has shape (batch_size, num_variates, prediction_length, num_samples)
        # where batch size means total_num_samples / samples_per_batch
        forecast = self.forecast(inputs, prediction_length, num_samples, samples_per_batch, use_kv_cache)

        # Reshape to combine batch_size and num_samples dimensions
        samples = forecast.samples
        batch_size, num_variates, pred_len, num_samples = samples.shape
        # samples is now (total_num_samples, num_variates, pred_len)
        samples = samples.permute(0, 3, 1, 2).reshape(batch_size * num_samples, num_variates, pred_len)
        # reorder dimensions to (num_variates, total_num_samples, pred_len)
        samples = samples.permute(1, 0, 2)  # this is just for consistency with the other pipelines
        return samples

    @torch.no_grad()
    def generate_samples(
        self,
        inputs: Float[torch.Tensor, "batch variate time_steps"],
        prediction_length: int,
        num_samples: int,
        timestamp_seconds: Int[torch.Tensor, "batch variate time_steps"],
        time_interval_seconds: Int[torch.Tensor, "batch variate"],
        input_padding_mask: Bool[torch.Tensor, "batch variate time_steps"] | None = None,
        id_mask: Float[torch.Tensor, "batch variate time_steps"] | None = None,
        sampling_batch_size: int = 10,
        use_kv_cache: bool = False,
    ) -> Float[torch.Tensor, "batch variate time_steps samples"]:
        """
        Generate samples from the output distribution.
        This method works autorregressively, i.e. it feeds the model's predictions back into itself.
        It works by creating num_samples chains. Each chain is a separate sequence of predictions.
        At each time step, for each chain we take a single sample from the output distribution and append
        it to the end of the sequence.
        """
        if input_padding_mask is None:
            input_padding_mask = torch.ones_like(inputs, dtype=torch.bool, device=inputs.device)
        if id_mask is None:
            id_mask = torch.zeros_like(inputs, dtype=torch.int, device=inputs.device)

        assert num_samples % sampling_batch_size == 0, "num_samples must be divisible by sampling_batch_size"
        num_batches = num_samples // sampling_batch_size

        # round up the prediction length to the nearest multiple of the patch size
        patch_size = self.model.patch_embed.patch_size
        rounded_steps = int(np.ceil(prediction_length / patch_size) * patch_size)
        start_index = inputs.shape[-1]
        end_index = start_index + prediction_length

        dummy_padding = torch.ones(
            (
                input_padding_mask.shape[0] * sampling_batch_size,
                input_padding_mask.shape[1],
                patch_size,
            ),
            dtype=torch.bool,
            device=inputs.device,
        )
        dummy_id_mask = repeat(
            id_mask[:, :, -1:],
            "batch variates 1 -> (sampling_batch_size batch) variates patch_size",
            sampling_batch_size=sampling_batch_size,
            patch_size=patch_size,
        )
        inputs = repeat(
            inputs,
            "batch variates seq_len -> (sampling_batch_size batch) variates seq_len",
            sampling_batch_size=sampling_batch_size,
        )
        input_padding_mask = repeat(
            input_padding_mask,
            "batch variates seq_len -> (sampling_batch_size batch) variates seq_len",
            sampling_batch_size=sampling_batch_size,
        )
        id_mask = repeat(
            id_mask,
            "batch variates seq_len -> (sampling_batch_size batch) variates seq_len",
            sampling_batch_size=sampling_batch_size,
        )
        timestamp_seconds = repeat(
            timestamp_seconds,
            "batch variates seq_len -> (sampling_batch_size batch) variates seq_len",
            sampling_batch_size=sampling_batch_size,
        )
        time_interval_seconds = repeat(
            time_interval_seconds,
            "batch variates -> (sampling_batch_size batch) variates",
            sampling_batch_size=sampling_batch_size,
        )

        all_samples = []
        if use_kv_cache:
            kv_cache = self.model.allocate_kv_cache(
                batch_size=inputs.shape[0],
                num_variates=inputs.shape[1],
                max_time_steps=inputs.shape[2] + rounded_steps,
                device=inputs.device,
                dtype=inputs.dtype,
            )
        else:
            kv_cache = None

        scaling_prefix_length = inputs.shape[-1]

        for _ in range(num_batches):
            batch_inputs = torch.clone(inputs)
            batch_input_padding_mask = torch.clone(input_padding_mask)
            batch_id_mask = torch.clone(id_mask)
            batch_timestamp_seconds = torch.clone(timestamp_seconds)

            for _ in range(rounded_steps // patch_size):
                base_distr, loc, scale = self.model(  # type: ignore[attr-defined]
                    inputs=batch_inputs,
                    input_padding_mask=batch_input_padding_mask,
                    id_mask=batch_id_mask,
                    kv_cache=kv_cache,
                    scaling_prefix_length=scaling_prefix_length,
                )
                distr = self.create_affine_transformed(base_distr, loc, scale)

                sample = distr.sample()
                assert sample is not None

                # We remove extreme values that can occur early in training
                # and cause validation metrics to be NaN
                samples = replace_extreme_values(sample[:, :, -patch_size:])
                batch_inputs = torch.cat([batch_inputs, samples], dim=-1)
                batch_id_mask = torch.cat([batch_id_mask, dummy_id_mask], dim=-1)
                batch_input_padding_mask = torch.cat([batch_input_padding_mask, dummy_padding], dim=-1)
                for _ in range(patch_size):
                    next_timestamp = batch_timestamp_seconds[:, :, -1] + time_interval_seconds
                    batch_timestamp_seconds = torch.cat([batch_timestamp_seconds, next_timestamp.unsqueeze(-1)], dim=-1)
            all_samples.append(batch_inputs)
            if kv_cache is not None:
                kv_cache.reset()

        outputs = torch.cat(all_samples, dim=0)
        unfolded_outputs = rearrange(
            outputs,
            "(samples batch) variates seq_len -> batch variates seq_len samples",
            samples=num_samples,
        ).detach()

        trimmed_predictions = unfolded_outputs[:, :, start_index:end_index, :]
        return trimmed_predictions
