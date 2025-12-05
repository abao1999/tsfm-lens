"""
Pipeline for TSFM Lens with Chronos
Builds on top of original Chronos codebase https://github.com/amazon-science/chronos-forecasting
    (under Apache-2.0 license):
    Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
    SPDX-License-Identifier: Apache-2.0
"""

import logging
import warnings
from dataclasses import dataclass

import torch
from chronos.chronos_bolt import (
    ChronosBoltModelForForecasting,
    ChronosBoltOutput,
    ChronosBoltPipeline,
)
from transformers import (
    AutoConfig,
)
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers.models.t5.configuration_t5 import T5Config

EPS = 1e-12
logger = logging.getLogger(__file__)


class ChronosBoltForecastModelFullOutputs(ChronosBoltModelForForecasting):
    """
    Same functionality as ChronosBoltModelForForecasting, but the forward pass returns the full outputs of the model.
    NOTE: this needs to be hashable, so we don't make it a dataclass, for now
    """

    def __init__(self, config: T5Config):
        super().__init__(config)

    def forward(
        self,
        context: torch.Tensor,
        mask: torch.Tensor | None = None,
        target: torch.Tensor | None = None,
        target_mask: torch.Tensor | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        use_cache: bool = True,
        return_dict_in_generate: bool = False,
    ) -> tuple[ChronosBoltOutput, BaseModelOutputWithPastAndCrossAttentions] | ChronosBoltOutput:
        """
        TODO: return the full outputs of the decoder.
        """
        batch_size = context.size(0)

        hidden_states, loc_scale, input_embeds, attention_mask = self.encode(context=context, mask=mask)
        sequence_output, full_decoder_output = self.decode(
            input_embeds,
            attention_mask,
            hidden_states,
            output_attentions=output_attentions,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict_in_generate=return_dict_in_generate,
        )

        quantile_preds_shape = (
            batch_size,
            self.num_quantiles,
            self.chronos_config.prediction_length,
        )
        quantile_preds = self.output_patch_embedding(sequence_output).view(*quantile_preds_shape)

        loss = None
        if target is not None:
            # normalize target
            target, _ = self.instance_norm(target, loc_scale)
            target = target.unsqueeze(1)  # type: ignore
            assert self.chronos_config.prediction_length >= target.shape[-1]

            target = target.to(quantile_preds.device)
            target_mask = (
                target_mask.unsqueeze(1).to(quantile_preds.device) if target_mask is not None else ~torch.isnan(target)
            )
            target[~target_mask] = 0.0

            # pad target and target_mask if they are shorter than model's prediction_length
            if self.chronos_config.prediction_length > target.shape[-1]:
                padding_shape = (
                    *target.shape[:-1],
                    self.chronos_config.prediction_length - target.shape[-1],
                )
                target = torch.cat([target, torch.zeros(padding_shape).to(target)], dim=-1)
                target_mask = torch.cat([target_mask, torch.zeros(padding_shape).to(target_mask)], dim=-1)

            loss = (
                2
                * torch.abs(
                    (target - quantile_preds)
                    * (
                        (target <= quantile_preds).float() - self.quantiles.view(1, self.num_quantiles, 1)  # type: ignore
                    )
                )
                * target_mask.float()
            )
            loss = loss.mean(dim=-2)  # Mean over prediction horizon
            loss = loss.sum(dim=-1)  # Sum over quantile levels
            loss = loss.mean()  # Mean over batch

        # Unscale predictions
        quantile_preds = self.instance_norm.inverse(
            quantile_preds.view(batch_size, -1),
            loc_scale,
        ).view(*quantile_preds_shape)

        if return_dict_in_generate:
            return ChronosBoltOutput(
                loss=loss,
                quantile_preds=quantile_preds,
            ), full_decoder_output  # type: ignore
        else:
            return ChronosBoltOutput(loss=loss, quantile_preds=quantile_preds)

    def decode(
        self,
        input_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        hidden_states: torch.Tensor,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        use_cache: bool = True,
        return_dict_in_generate: bool = False,
    ) -> tuple[
        torch.Tensor,
        BaseModelOutputWithPastAndCrossAttentions | None,
    ]:
        """
        Runs the decoder stack for Chronos-Bolt.

        Parameters
        ----------
        input_embeds : torch.Tensor
            Patched and embedded inputs. Shape: (batch_size, patched_context_length, d_model)
        attention_mask : torch.Tensor
            Attention mask for the patched context. Shape: (batch_size, patched_context_length), dtype: torch.int64
            NOTE: this is actually the encoder attention mask that is passed in (see the forward pass)
        hidden_states : torch.Tensor
            Hidden states returned by the encoder. Shape: (batch_size, patched_context_length, d_model)
        output_attentions : bool, optional
            Whether to return attention weights. Default: False.
        use_cache : bool, optional
            Whether to use past key values for faster decoding. Default: True.
        output_hidden_states : bool, optional
            Whether to return all hidden states. Default: False.

        Returns
        -------
        BaseModelOutputWithPastAndCrossAttentions | None
            The full output object from the decoder, including last_hidden_state and optionally attentions and hidden states.
        """
        batch_size = input_embeds.shape[0]
        decoder_input_ids = torch.full(  # type: ignore
            (batch_size, 1),
            self.config.decoder_start_token_id,  # type: ignore
            device=input_embeds.device,
        )
        # NOTE: self.decoder is a T5Stack https://github.com/huggingface/transformers/blob/a871f6f58d49f3a05ae9dae519caa8aa9d919a07/src/transformers/models/t5/modeling_t5.py#L952
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=None,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            inputs_embeds=None,
            head_mask=None,
            cross_attn_head_mask=None,
            past_key_values=None,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=None,
        )

        if return_dict_in_generate:
            return decoder_outputs.last_hidden_state, decoder_outputs
        else:
            return (
                decoder_outputs.last_hidden_state,
                None,
            )  # sequence_outputs, b x 1 x d_model


@dataclass
class ChronosBoltPipelineCustom(ChronosBoltPipeline):
    """
    A ``ChronosBoltPipelineCustom`` uses the given model to forecast
    input time series.

    Use the ``from_pretrained`` class method to load serialized models.
    Use the ``predict`` method to get forecasts.

    See: https://github.com/amazon-science/chronos-forecasting/blob/fcd09fe8b6bf57643bde7847bf42a6719b7ee205/src/chronos/chronos_bolt.py#L425

    Parameters
    ----------
    model
        The model to use.
    """

    model: ChronosBoltForecastModelFullOutputs

    def predict_with_full_outputs(  # type: ignore[override]
        self,
        context: torch.Tensor | list[torch.Tensor],
        prediction_length: int,
        limit_prediction_length: bool = False,
        output_attentions: bool = True,
        output_hidden_states: bool = True,
        use_cache: bool = True,
        return_dict_in_generate: bool = True,
    ) -> tuple[torch.Tensor, BaseModelOutputWithPastAndCrossAttentions | None]:
        """
        Get forecasts for the given time series.

        Refer to the base method (``BaseChronosPipeline.predict``)
        for details on shared parameters.
        Additional parameters
        ---------------------
        limit_prediction_length
            Force prediction length smaller or equal than the
            built-in prediction length from the model. False by
            default. When true, fail loudly if longer predictions
            are requested, otherwise longer predictions are allowed.

        Returns
        -------
        torch.Tensor
            Forecasts of shape (batch_size, num_quantiles, prediction_length)
            where num_quantiles is the number of quantiles the model has been
            trained to output. For official Chronos-Bolt models, the value of
            num_quantiles is 9 for [0.1, 0.2, ..., 0.9]-quantiles.

        Raises
        ------
        ValueError
            When limit_prediction_length is True and the prediction_length is
            greater than model's trainig prediction_length.
        """
        context_tensor = self._prepare_and_validate_context(context=context)

        model_context_length = self.model.config.chronos_config["context_length"]
        model_prediction_length = self.model.config.chronos_config["prediction_length"]

        if prediction_length > model_prediction_length:
            msg = (
                f"We recommend keeping prediction length <= {model_prediction_length}. "
                "The quality of longer predictions may degrade since the model is not optimized for it. "
            )
            if limit_prediction_length:
                msg += "You can turn off this check by setting `limit_prediction_length=False`."
                raise ValueError(msg)
            warnings.warn(msg)

        predictions = []
        remaining = prediction_length

        # We truncate the context here because otherwise batches with very long
        # context could take up large amounts of GPU memory unnecessarily.
        if context_tensor.shape[-1] > model_context_length:
            context_tensor = context_tensor[..., -model_context_length:]

        # TODO: We unroll the forecast of Chronos Bolt greedily with the full forecast
        # horizon that the model was trained with (i.e., 64). This results in variance collapsing
        # every 64 steps.
        context_tensor = context_tensor.to(
            device=self.model.device,
            dtype=torch.float32,
        )
        full_inference_outputs = []
        while remaining > 0:
            with torch.no_grad():
                prediction, full_decoder_output = self.model(
                    context=context_tensor,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    use_cache=use_cache,
                    return_dict_in_generate=return_dict_in_generate,
                )
                prediction = prediction.quantile_preds.to(context_tensor)

            predictions.append(prediction)
            full_inference_outputs.append(full_decoder_output)
            remaining -= prediction.shape[-1]

            if remaining <= 0:
                break

            central_idx = torch.abs(torch.tensor(self.quantiles) - 0.5).argmin()
            central_prediction = prediction[:, central_idx]

            context_tensor = torch.cat([context_tensor, central_prediction], dim=-1)

        return torch.cat(predictions, dim=-1)[..., :prediction_length].to(
            dtype=torch.float32, device="cpu"
        ), full_inference_outputs if return_dict_in_generate else None  # type: ignore

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        """
        Load the model, either from a local path or from the HuggingFace Hub.
        Supports the same arguments as ``AutoConfig`` and ``AutoModel``
        from ``transformers``.
        """

        config = AutoConfig.from_pretrained(*args, **kwargs)
        assert hasattr(config, "chronos_config"), "Not a Chronos config file"

        architecture = config.architectures[0]  # type: ignore
        logger.info(f"Original model architecture: {architecture}")
        class_ = globals().get(architecture)
        logger.info(f"Class: {class_}")
        logger.info(
            "Setting model architecture to ChronosBoltForecastModelFullOutputs so we can expose the full decoder outputs."
        )
        class_ = ChronosBoltForecastModelFullOutputs

        model = class_.from_pretrained(*args, **kwargs)
        return cls(model=model)
