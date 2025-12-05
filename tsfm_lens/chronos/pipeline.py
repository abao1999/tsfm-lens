"""
Pipeline for TSFM Lens with Chronos
Builds on top of original Chronos codebase https://github.com/amazon-science/chronos-forecasting
    (under Apache-2.0 license):
    Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
    SPDX-License-Identifier: Apache-2.0
"""

import logging
from dataclasses import dataclass

import numpy as np
import torch
from chronos import (
    ChronosConfig,
    ChronosModel,
    ChronosPipeline,
    MeanScaleUniformBins,
)
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
)
from transformers.generation.configuration_utils import GenerationConfig

from tsfm_lens.assimilation import (
    uncertainty_triggered_intervention,
)

EPS = 1e-12
logger = logging.getLogger(__file__)


@dataclass
class ChronosPipelinetsfm_lens(ChronosPipeline):
    """
    A ``ChronosPipelinetsfm_lens`` uses the given tokenizer and model to forecast
    input time series.

    Use the ``from_pretrained`` class method to load serialized models.
    Use the ``predict`` method to get forecasts.

    Parameters
    ----------
    tokenizer
        The tokenizer object to use.
    model
        The model to use.
    """

    tokenizer: MeanScaleUniformBins
    model: ChronosModel

    def predict_with_assimilation(
        self,
        context: torch.Tensor | list[torch.Tensor],
        future_vals: torch.Tensor,
        prediction_length: int | None = None,
        num_samples: int | None = None,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        limit_prediction_length: bool = True,
        deterministic: bool = False,
        latency_delay: int = 0,  # number of timesteps to delay the intervention
        verbose: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forecast with uncertainty-based interventions + context‐growth rollouts.

        Performs interventions (ground truth injection) at the n_edits timesteps with
        highest uncertainty, or when uncertainty exceeds threshold if n_edits is None.

        TODO: implement a latency: can't have faster than r rate of interventions
        Returns:
        - predictions: Tensor of shape (batch_size, num_samples, total_prediction_length)
        - corrected_timesteps: BoolTensor of shape (total_prediction_length,)
        """

        # --- Setup & validation ---
        context_tensor = self._prepare_and_validate_context(context)
        batch_size = context_tensor.shape[0]
        max_model_pred = int(self.model.config.prediction_length)

        total_pred_len = prediction_length or max_model_pred
        if total_pred_len > max_model_pred and limit_prediction_length:
            raise ValueError(
                f"Requested {total_pred_len}>model max {max_model_pred}. Set limit_prediction_length=False to override."
            )

        num_samples = num_samples or self.model.config.num_samples
        temperature = 1.0 if deterministic else temperature
        top_k = 1 if deterministic else top_k
        top_p = 1.0 if deterministic else top_p

        # Accumulate each rollout’s predictions into all_rollouts
        all_rollouts: list[torch.Tensor] = []
        corrected_timesteps = torch.zeros(total_pred_len, dtype=torch.bool, device=self.model.device)

        n_rollouts = 0
        remaining = total_pred_len

        # Loop until we generate full horizon
        while remaining > 0:
            # 1) Determine this rollout’s length
            curr_len = min(remaining, max_model_pred)

            # 2) Prepare context encoding
            encoder_ids, attention_mask, scale = self.tokenizer.context_input_transform(context_tensor)
            device = self.model.device

            # Tile encoder & mask → [B, N, ...] → flatten [B*N, ...]
            encoder_ids = (
                encoder_ids.to(device).unsqueeze(1).expand(-1, num_samples, -1).reshape(-1, encoder_ids.size(-1))
            )
            attention_mask = (
                attention_mask.to(device).unsqueeze(1).expand(-1, num_samples, -1).reshape(-1, attention_mask.size(-1))
            )

            # 3) Slice the appropriate window of future_vals → [B, curr_len]
            start = n_rollouts * max_model_pred
            end = start + curr_len
            future_slice = future_vals[:, start:end]  # [B, curr_len]

            # Tile future → [B, N, curr_len] → flatten [B*N, curr_len]
            future_ids, _, _ = self.tokenizer._input_transform(future_slice, scale=scale)
            future_ids = future_ids.to(device).unsqueeze(1).expand(-1, num_samples, -1).reshape(-1, future_ids.size(-1))

            # 4) Init decoder_ids = start_token → [B*N, 1]
            pad_id = self.tokenizer.config.pad_token_id
            decoder_ids = (
                torch.full((batch_size, 1), pad_id, device=device)
                .unsqueeze(1)
                .expand(-1, num_samples, -1)
                .reshape(-1, 1)
            )

            # 5) Single‐step loop for this rollout
            last_intervention_time = 0  # track the last time an intervention was made
            for t in tqdm(range(curr_len), desc="Generating rollout"):
                out = self.model.model.generate(
                    input_ids=encoder_ids,
                    attention_mask=attention_mask,
                    decoder_input_ids=decoder_ids,
                    generation_config=GenerationConfig(
                        min_new_tokens=1,
                        max_new_tokens=1,
                        do_sample=(not deterministic),
                        num_return_sequences=1,
                        eos_token_id=self.model.config.eos_token_id,
                        pad_token_id=pad_id,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        return_dict_in_generate=True,
                        use_cache=True,
                    ),
                )  # type: ignore
                # [B*N, seq_len_so_far+1]
                decoder_ids = out.sequences

                # Uncertainty-based intervention
                if num_samples > 1 and t - last_intervention_time >= latency_delay:
                    # Get model logits for the last generated token. Calls forward pass for computing model internal representations
                    with torch.no_grad():
                        outputs = self.model.model(
                            input_ids=encoder_ids,
                            attention_mask=attention_mask,
                            decoder_input_ids=decoder_ids,
                            return_dict=True,
                        )
                        logits = outputs.logits[:, -1, :]  # [B*N, vocab_size]

                        # Reshape logits to [B, N, V] for batched processing
                        logits_batched = logits.view(batch_size, num_samples, -1)  # [B, N, V]

                        do_intervene, _ = uncertainty_triggered_intervention(
                            logits_batched,
                            conf_thresh=0.3,
                            dis_thresh=0.7,
                            k=1,
                            verbose=verbose,
                        )

                        # Apply intervention for batches that need it
                        if do_intervene.any():
                            last_intervention_time = t
                            # Create mask for batches that need intervention
                            intervention_mask = do_intervene.unsqueeze(1).expand(-1, num_samples)  # [B, N]
                            intervention_mask = intervention_mask.flatten()  # [B*N]

                            # Replace tokens for batches that need intervention
                            decoder_ids[intervention_mask, -1] = future_ids[:, t][intervention_mask]
                            corrected_timesteps[start + t] = True
                            if verbose:
                                print(f"t={t}, intervened with gt_tok={future_ids[:, t]}")

            # 6) Un‑flatten → [B, N, seq_len]
            seq_len = decoder_ids.size(-1)
            decoder_ids = decoder_ids.view(batch_size, num_samples, seq_len)

            # drop initial pad/start token → [B, N, curr_len]
            rollout_preds = decoder_ids[..., 1:]

            # 7) Inverse‐transform to real scale
            rollout_preds = self.tokenizer.output_transform(rollout_preds.to(scale.device), scale)

            all_rollouts.append(rollout_preds)
            remaining -= curr_len
            n_rollouts += 1

            # 8) Grow the context if needed
            if remaining > 0:
                next_ctx = rollout_preds.median(dim=1).values
                context_tensor = torch.cat([context_tensor, next_ctx], dim=-1)

        # Concatenate all rollout segments along the time axis → [B, N, total_pred_len]
        final_preds = torch.cat(all_rollouts, dim=-1)

        return final_preds, corrected_timesteps

    def predict_with_edits(
        self,
        context: torch.Tensor | list[torch.Tensor],
        future_vals: torch.Tensor,
        p: float = 0.0,
        prediction_length: int | None = None,
        num_samples: int | None = None,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        limit_prediction_length: bool = True,
        deterministic: bool = False,
        corrections_rseed: int = 99,
        return_dict_in_generate: bool = True,
        verbose: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor] | None]:
        """
        Forecast with scheduled‐sampling–style injection + context‐growth rollouts.

        Returns:
        - predictions: Tensor of shape (batch_size, num_samples, total_prediction_length)
        - corrected_timesteps: BoolTensor of shape (total_prediction_length,)
        if return_dict_in_generate:
        - full_inference_outputs: List of tensors of shape (batch_size, num_samples, seq_len)
        """
        corrections_rng = np.random.default_rng(corrections_rseed)

        # --- Setup & validation ---
        context_tensor = self._prepare_and_validate_context(context)
        batch_size = context_tensor.shape[0]
        max_model_pred = int(self.model.config.prediction_length)

        total_pred_len = prediction_length or max_model_pred
        if total_pred_len > max_model_pred and limit_prediction_length:
            raise ValueError(
                f"Requested {total_pred_len}>model max {max_model_pred}. Set limit_prediction_length=False to override."
            )

        num_samples = num_samples or self.model.config.num_samples
        temperature = 1.0 if deterministic else temperature
        top_k = 1 if deterministic else top_k
        top_p = 1.0 if deterministic else top_p

        # Accumulate each rollout’s predictions into all_rollouts
        all_rollouts: list[torch.Tensor] = []
        corrected_timesteps = torch.zeros(total_pred_len, dtype=torch.bool, device=self.model.device)
        full_inference_outputs: list[torch.Tensor] = []

        n_rollouts = 0
        remaining = total_pred_len

        # Loop until we generate full horizon
        while remaining > 0:
            # 1) Determine this rollout’s length
            curr_len = min(remaining, max_model_pred)

            # 2) Prepare context encoding
            encoder_ids, attention_mask, scale = self.tokenizer.context_input_transform(context_tensor)
            device = self.model.device

            # Tile encoder & mask → [B, N, ...] → flatten [B*N, ...]
            encoder_ids = (
                encoder_ids.to(device).unsqueeze(1).expand(-1, num_samples, -1).reshape(-1, encoder_ids.size(-1))
            )
            attention_mask = (
                attention_mask.to(device).unsqueeze(1).expand(-1, num_samples, -1).reshape(-1, attention_mask.size(-1))
            )

            # 3) Slice the appropriate window of future_vals → [B, curr_len]
            start = n_rollouts * max_model_pred
            end = start + curr_len
            future_slice = future_vals[:, start:end]  # [B, curr_len]

            # Tile future → [B, N, curr_len] → flatten [B*N, curr_len]
            future_ids, _, _ = self.tokenizer._input_transform(future_slice, scale=scale)
            future_ids = future_ids.to(device).unsqueeze(1).expand(-1, num_samples, -1).reshape(-1, future_ids.size(-1))

            # 4) Init decoder_ids = start_token → [B*N, 1]
            pad_id = self.tokenizer.config.pad_token_id
            decoder_ids = (
                torch.full((batch_size, 1), pad_id, device=device)
                .unsqueeze(1)
                .expand(-1, num_samples, -1)
                .reshape(-1, 1)
            )

            # 5) Single‐step loop for this rollout
            for t in range(curr_len):
                out = self.model.model.generate(
                    input_ids=encoder_ids,
                    attention_mask=attention_mask,
                    decoder_input_ids=decoder_ids,
                    generation_config=GenerationConfig(
                        min_new_tokens=1,
                        max_new_tokens=1,
                        do_sample=(not deterministic),
                        num_return_sequences=1,
                        eos_token_id=self.model.config.eos_token_id,
                        pad_token_id=pad_id,
                        temperature=temperature,
                        # top_k=top_k,
                        top_p=top_p,
                        return_dict_in_generate=True,
                        use_cache=True,
                    ),
                )  # type: ignore
                # [B*N, seq_len_so_far+1]

                decoder_ids = out.sequences
                if verbose:
                    print(f"t={t}, current decoder id: {decoder_ids[:, -1]}")

                # Ground-truth injection. TODO: make this a mask to allow channel-inconsistent injection
                if corrections_rng.random() < p:
                    gt_tok = future_ids[:, t]  # [B*N]
                    decoder_ids[:, -1] = gt_tok
                    # record if any sample in any batch used GT at global position
                    corrected_timesteps[start + t] = True

            if return_dict_in_generate:
                full_inference_outputs.append(out)
                if verbose:
                    print(f"t={t}, full_inference_outputs shape: {out.sequences.shape}")

            # 6) Un‑flatten → [B, N, seq_len]
            seq_len = decoder_ids.size(-1)
            decoder_ids = decoder_ids.view(batch_size, num_samples, seq_len)

            # drop initial pad/start token → [B, N, curr_len]
            rollout_preds = decoder_ids[..., 1:]

            # 7) Inverse‐transform to real scale
            rollout_preds = self.tokenizer.output_transform(rollout_preds.to(scale.device), scale)

            all_rollouts.append(rollout_preds)
            remaining -= curr_len
            n_rollouts += 1

            # 8) Grow the context if needed
            if remaining > 0:
                next_ctx = rollout_preds.median(dim=1).values
                context_tensor = torch.cat([context_tensor, next_ctx], dim=-1)

        # Concatenate all rollout segments along the time axis → [B, N, total_pred_len]
        final_preds = torch.cat(all_rollouts, dim=-1)

        if return_dict_in_generate:
            return final_preds, corrected_timesteps, full_inference_outputs
        else:
            return final_preds, corrected_timesteps

    def predict_with_full_outputs(
        self,
        context: torch.Tensor | list[torch.Tensor],
        prediction_length: int | None = None,
        num_samples: int | None = None,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        limit_prediction_length: bool = False,
        use_cache: bool = True,
        output_attentions: bool = True,
        output_hidden_states: bool = True,
        return_dict_in_generate: bool = True,
        output_scores: bool = True,
        do_sample: bool = True,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Generate predictions for a given context, with options to return additional
        outputs from the underlying transformer model such as attention scores,
        hidden states, and generation scores.

        Parameters
        ----------
        context : torch.Tensor or list[torch.Tensor]
            The input context(s) for prediction. Can be a single tensor or a list of tensors.
        prediction_length : int, optional
            Number of time steps to predict. If None, uses the model's configured prediction length.
        num_samples : int, optional
            Number of samples to generate per input in the batch. If None, uses the model's default.
        temperature : float, optional
            Sampling temperature for generation. If None, uses the model's default.
        top_k : int, optional
            Top-k sampling parameter. If None, uses the model's default.
        top_p : float, optional
            Top-p (nucleus) sampling parameter. If None, uses the model's default.
        limit_prediction_length : bool, default=False
            If True, raises an error if prediction_length exceeds the model's configured maximum.
            If False, only logs a warning.
        use_cache : bool, optional
            Whether to use the model's cache during generation.
        output_attentions : bool, optional
            Whether to return attention weights from the model's generate method.
        output_hidden_states : bool, optional
            Whether to return hidden states from the model's generate method.
        return_dict_in_generate : bool, optional
            Whether to return a dict from the model's generate method. If True, returns
            a tuple of (predictions, list of full inference outputs); if False, returns
            only the predictions tensor.
        output_scores : bool, optional
            Whether to return generation scores from the model's generate method.
        do_sample : bool, default=True
            Whether to use sampling for generation; if False, uses greedy decoding.

        Returns
        -------
        torch.Tensor or tuple of (torch.Tensor, list)
            If return_dict_in_generate is False, returns a tensor of predictions of shape
            [batch_size, num_samples, prediction_length]. If True, returns a tuple of
            (predictions, list of full inference outputs from the model's generate method),
            where each element in the list contains the full output dict for a rollout.
        """

        context_tensor = self._prepare_and_validate_context(context=context)

        if prediction_length is None:
            prediction_length = self.model.config.prediction_length

        if prediction_length > self.model.config.prediction_length:
            msg = (
                f"We recommend keeping prediction length <= {self.model.config.prediction_length}. "
                "The quality of longer predictions may degrade since the model is not optimized for it. "
            )
            if limit_prediction_length:
                msg += "You can turn off this check by setting `limit_prediction_length=False`."
                raise ValueError(msg)
            logger.warning(msg)

        predictions = []
        full_inference_outputs = []
        remaining = prediction_length

        while remaining > 0:
            token_ids, attention_mask, scale = self.tokenizer.context_input_transform(context_tensor)

            prediction_length = min(remaining, self.model.config.prediction_length)
            if num_samples is None:
                num_samples = self.model.config.num_samples
            if temperature is None:
                temperature = self.model.config.temperature
            if top_k is None:
                top_k = self.model.config.top_k
            if top_p is None:
                top_p = self.model.config.top_p

            assert hasattr(self.model.model, "generate")

            input_ids = token_ids.to(self.model.device)
            attention_mask = attention_mask.to(self.model.device)

            preds = self.model.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=GenerationConfig(
                    min_new_tokens=prediction_length,
                    max_new_tokens=prediction_length,
                    do_sample=do_sample,
                    num_return_sequences=num_samples,
                    eos_token_id=self.model.config.eos_token_id,
                    pad_token_id=self.model.config.pad_token_id,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    use_cache=use_cache,
                ),
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict_in_generate=return_dict_in_generate,
                output_scores=output_scores,
            )  # type: ignore

            if return_dict_in_generate:
                full_inference_outputs.append(preds)
                preds = preds.sequences

            if self.model.config.model_type == "seq2seq":
                preds = preds[..., 1:]  # remove the decoder start token
            else:
                assert self.model.config.model_type == "causal"
                assert preds.size(-1) == token_ids.size(-1) + prediction_length
                preds = preds[..., -prediction_length:]

            samples = preds.reshape(token_ids.size(0), num_samples, -1)

            prediction = self.tokenizer.output_transform(samples.to(scale.device), scale)

            predictions.append(prediction)
            remaining -= prediction.shape[-1]

            if remaining <= 0:
                break

            context_tensor = torch.cat([context_tensor, prediction.median(dim=1).values], dim=-1)
        if return_dict_in_generate:
            return torch.cat(predictions, dim=-1).to(dtype=torch.float32, device="cpu"), full_inference_outputs  # type: ignore
        else:
            return torch.cat(predictions, dim=-1).to(dtype=torch.float32, device="cpu")

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        """
        Load the model, either from a local path or from the HuggingFace Hub.
        Supports the same arguments as ``AutoConfig`` and ``AutoModel``
        from ``transformers``.
        """
        config = AutoConfig.from_pretrained(*args, **kwargs)

        assert hasattr(config, "chronos_config"), "Not a Chronos config file"

        chronos_config = ChronosConfig(**config.chronos_config)

        if chronos_config.tokenizer_class != "MeanScaleUniformBins":
            raise ValueError("ChronosPipelinetsfm_lens currently only supports MeanScaleUniformBins tokenizer")

        if chronos_config.model_type == "seq2seq":
            inner_model = AutoModelForSeq2SeqLM.from_pretrained(*args, **kwargs)
        else:
            assert chronos_config.model_type == "causal"
            inner_model = AutoModelForCausalLM.from_pretrained(*args, **kwargs)

        assert hasattr(inner_model, "generate"), "Inner model does not have a generate method"

        return cls(
            tokenizer=chronos_config.create_tokenizer(),  # type: ignore
            model=ChronosModel(config=chronos_config, model=inner_model),
        )
