"""
Logit Attribution Evaluation framework for Chronos.
"""

import logging
import os
import pickle
from collections import defaultdict
from typing import Literal

import numpy as np
import torch
from gluonts.itertools import batcher
from tqdm.auto import tqdm

from tsfm_lens.chronos.circuitlens import CircuitLensChronos
from tsfm_lens.circuitlens import BaseCircuitLens
from tsfm_lens.dataset import TestDataset
from tsfm_lens.metrics import compute_metrics
from tsfm_lens.utils import (
    beam_search_from_fixed_logits,
    collect_attributions,
    compute_logit_metrics,
    reshape_batch_data,
    set_seed,
    validate_and_get_sample_count,
)

logger = logging.getLogger(__name__)


def _output_transform_tokens_using_context(pipeline, tokens, context):
    """
    Transform tokens back to coordinate space using the context
    """
    _, _, scale = pipeline.tokenizer.context_input_transform(torch.tensor(context).to(dtype=torch.float32))
    return pipeline.tokenizer.output_transform(torch.tensor(tokens).to(scale.device), scale)


def _get_logits_and_tokens_from_attribution_output(
    pipeline,
    attribution_output: torch.Tensor,
    decoding_method: Literal["greedy", "beam"] = "greedy",
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Get logits and tokens from the attribution output
    attribution_output is a tensor of shape (batch_size, num_samples, prediction_length, d_model)
    Returns a tuple of (logits, tokens), where
    logits is of shape (batch_size, num_samples, prediction_length, vocab_size) and
    tokens is of shape (batch_size, num_samples, prediction_length)
    """
    num_samples = attribution_output.shape[1]
    logits = np.stack(
        [pipeline.unembed_residual(attribution_output[:, j, ...]).detach().cpu().numpy() for j in range(num_samples)],
        axis=1,
    )
    # greedy decoding
    if decoding_method == "greedy":
        if verbose and num_samples > 1:
            logger.warning("num_samples > 1, but decoding_method is greedy. ")
        tokens = np.argmax(logits, axis=-1)
    elif decoding_method == "beam":
        if verbose:
            logger.warning("decoding_method is beam. This implementation is not tested.")
        tokens, _ = beam_search_from_fixed_logits(torch.tensor(logits), beam_size=4)
        tokens = tokens.squeeze(2).cpu().numpy()
    else:
        raise ValueError(f"Unknown decoding method: {decoding_method}")
    return logits, tokens


def evaluate_attribution_with_full_outputs(
    pipeline: CircuitLensChronos,
    systems: dict[str, TestDataset],
    batch_size: int,
    prediction_length: int,
    system_dims: dict[str, int],
    metric_names: list[str] | None = None,
    eval_subintervals: list[tuple[int, int]] | None = None,
    parallel_sample_reduction: Literal["mean", "median"] = "median",
    prediction_kwargs: dict = {},
    rseed: int = 99,
    attribution_types: set[Literal["mlp", "read_stream"]] = {"mlp", "read_stream"},
    save_dir: str | None = None,
    save_predictions_from_attribution: bool = False,
) -> tuple[
    dict[int, dict[str, dict[str, float]]],
    dict[str, dict[int, dict[int, dict[str, dict[str, float]]]]],
    dict[str, dict[int, dict[str, np.ndarray]]],
]:
    """
    NOTE: this function is currently stale
    Evaluate the Chronos pipeline on a set of test systems, computing both standard and attribution-based metrics.

    This function runs the model on each provided system, collects predictions, and computes metrics over the
    prediction window and optionally over specified subintervals. It also computes metrics from attribution
    outputs (e.g., MLP and read stream attributions) for each layer.

    Args:
        pipeline: BaseCircuitLens instance used for generating predictions and extracting attributions.
        systems: Dictionary mapping system names to TestDataset objects.
        batch_size: Number of samples per batch during evaluation.
        prediction_length: Number of timesteps to predict for each sample.
        system_dims: Dictionary mapping system names to their data dimensionality.
        metric_names: Optional list of metric names to compute (default: all available).
        eval_subintervals: Optional list of (start, end) index tuples specifying subintervals
            within the prediction window to evaluate metrics on. If None, evaluates on the
            full prediction window.
        parallel_sample_reduction: How to reduce the parallel samples over axis 0 ("mean" or "median").
        prediction_kwargs: Optional dictionary of additional keyword arguments to pass to
            the pipeline's predict method.
        rseed: Random seed for reproducibility (default: 99).
        attribution_types: Set of attribution types to evaluate (e.g., {"mlp", "read_stream"}).
        save_dir: Optional directory path to save evaluation results.
        save_predictions_from_attribution: Whether to save predictions from attribution outputs.

    Returns:
        system_metrics: Nested dictionary of the form
            {system_name: {subinterval: {metric_name: value, ...}, ...}, ...}
            containing computed metrics for each system and subinterval.
        system_metrics_attribution: Nested dictionary of the form
            {attribution_type: {layer_idx: {system_name: {subinterval: {metric_name: value, ...}, ...}, ...}}, ...}
            containing computed metrics for each system and subinterval from attribution outputs, for each layer and attribution type.
    """
    assert isinstance(pipeline, BaseCircuitLens), "pipeline must be a BaseCircuitLens instance"
    num_layers = pipeline.num_layers

    if hasattr(pipeline.model, "eval"):
        pipeline.model.eval()  # redundancy for extra safety
    elif hasattr(pipeline, "set_to_eval_mode"):
        pipeline.set_to_eval_mode()

    system_metrics = defaultdict(dict)
    system_metrics_attribution = {}
    system_metrics_attribution["mlp"] = defaultdict(lambda: defaultdict(dict))
    system_metrics_attribution["read_stream"] = defaultdict(lambda: defaultdict(dict))
    combined_metrics_from_logits = defaultdict(lambda: defaultdict(dict))

    logger.info(f"attribution_types: {attribution_types}")

    if eval_subintervals is None:
        eval_subintervals = [(0, prediction_length)]
    elif (0, prediction_length) not in eval_subintervals:
        eval_subintervals.append((0, prediction_length))

    parallel_sample_reduction_fn = {
        "mean": lambda x: np.mean(x, axis=0),
        "median": lambda x: np.median(x, axis=0),
    }[parallel_sample_reduction]

    num_samples = prediction_kwargs.get("num_samples", 1)
    logger.info(f"num_samples: {num_samples}")

    pbar = tqdm(systems, desc="Forecasting...")
    for system_idx, system in enumerate(pbar):
        dataset = systems[system]
        num_datasets = len(dataset.datasets)
        subdir_name = (
            dataset.datasets[0].name
            if hasattr(dataset.datasets[0], "name")
            else dataset.datasets[0].iterable.path.parent.name
        )
        dim = system_dims[system]

        logger.info(f"subdir_name: {subdir_name} with {num_datasets} trajectory files or samples and dimension {dim}")

        predictions, labels, contexts = [], [], []
        # entropy, effective_vocab, top_k_entropy = [], [], []

        preds_from_mlp_outputs_all_layers = defaultdict(list)
        preds_from_read_stream_outputs_all_layers = defaultdict(list)
        entropy_by_layer = defaultdict(list)
        effective_vocab_by_layer = defaultdict(list)
        top_k_entropy_by_layer = defaultdict(list)
        peak_counts_by_layer = defaultdict(list)

        # Process batches
        for i, batch in enumerate(batcher(dataset, batch_size=batch_size)):
            past_values, future_values = zip(*[(data["past_values"], data["future_values"]) for data in batch])

            context = torch.cat(past_values, dim=0)
            future_vals = torch.cat(future_values, dim=0)

            # NOTE: for the last batch, the actual batch_size may be less than the batcher batch size
            actual_batch_size = context.shape[0]

            # Get predictions
            set_seed(rseed)
            preds, encdec_out = pipeline.predict_with_full_outputs(  # type: ignore
                context,
                prediction_length=prediction_length,
                use_cache=True,
                return_dict_in_generate=True,
                output_scores=True,
                **prediction_kwargs,  # NOTE: should have limit_prediction_length=False
            )

            # Convert to numpy
            context = context.cpu().numpy()
            future_vals = future_vals.cpu().numpy()
            # shape is (num_samples, batch_size, prediction_length)
            preds = preds.transpose(0, 1).cpu().numpy()
            # Truncate predictions if needed
            if preds.shape[-1] > future_vals.shape[-1]:
                preds = preds[..., : future_vals.shape[-1]]

            # # CA head attribution outputs: dict whose keys are layer indices and values are lists of lists of tensors (one for each num_heads heads, for each num_samples parallel samples)
            # ca_head_outputs: dict[int, list[list[torch.Tensor]]] = {
            #     i: [
            #         collect_attributions(pipeline.ca_head_attribution_outputs[i][j])
            #         for j in range(num_heads)
            #     ]
            #     for i in range(num_layers)
            # }

            # MLP and read stream attribution outputs for each layer
            for layer_idx in range(num_layers):
                for name, outputs, preds_dict in [
                    (
                        "mlp",
                        pipeline.mlp_attribution_outputs,
                        preds_from_mlp_outputs_all_layers,
                    ),
                    (
                        "read_stream",
                        pipeline.read_stream_outputs,
                        preds_from_read_stream_outputs_all_layers,
                    ),
                ]:
                    if name not in attribution_types:
                        continue
                    # NOTE: necessary to reshape (num_samples * actual_batch_size, ...) to (actual_batch_size, num_samples, ...)
                    attribution_output_curr_layer = collect_attributions(outputs[layer_idx]).reshape(
                        actual_batch_size,
                        num_samples,
                        *collect_attributions(outputs[layer_idx]).shape[1:],
                    )

                    # logits is of shape (actual_batch_size, num_samples, prediction_length, vocab_size)
                    # tokens is of shape (actual_batch_size, num_samples, prediction_length)
                    logits, tokens = _get_logits_and_tokens_from_attribution_output(
                        pipeline,
                        attribution_output_curr_layer,
                        decoding_method="greedy",
                    )
                    # Output transform the tokens to the coordinate space (predictions)
                    # Shape is (num_samples, actual_batch_size, prediction_length)
                    preds_dict[layer_idx].append(
                        _output_transform_tokens_using_context(pipeline, tokens, context).transpose(0, 1).cpu().numpy()
                    )

                    if name == "read_stream":
                        # Compute entropy and other quantities of logit maps through layers
                        # logits is of shape (actual_batch_size, num_samples, prediction_length, vocab_size)
                        entropy, effective_vocab, top_k_entropy, peak_counts = compute_logit_metrics(
                            logits,
                            threshold=0.01,
                            top_k=10,
                        )
                        entropy_by_layer[layer_idx].append(np.swapaxes(entropy, 0, 1))
                        effective_vocab_by_layer[layer_idx].append(np.swapaxes(effective_vocab, 0, 1))
                        top_k_entropy_by_layer[layer_idx].append(np.swapaxes(top_k_entropy, 0, 1))
                        peak_counts_by_layer[layer_idx].append(np.swapaxes(peak_counts, 0, 1))

            if hasattr(pipeline, "reset_attribution_inputs_and_outputs"):
                pipeline.reset_attribution_inputs_and_outputs()

            labels.append(future_vals)
            predictions.append(preds)
            contexts.append(context)

        # Calculate dimensions
        total_eval_windows = sum(pred.shape[1] for pred in predictions)
        num_eval_windows = total_eval_windows // (num_datasets * dim)

        if total_eval_windows % (num_datasets * dim) != 0:
            raise ValueError(
                f"Total eval windows {total_eval_windows} is not divisible by num_datasets * dim = {num_datasets * dim}"
            )

        logger.info(f"Shape info: num_datasets={num_datasets}, num_eval_windows={num_eval_windows}, dim={dim}")

        # Reshape predictions with explicit dimension tracking
        if num_samples != predictions[0].shape[0]:
            raise ValueError(f"num_samples ({num_samples}) != predictions[0].shape[0] ({predictions[0].shape[0]})")

        def _reshape_predictions(arr):
            arr = arr.reshape(num_samples, num_datasets, num_eval_windows, dim, prediction_length)
            arr = arr.transpose(0, 1, 2, 4, 3)
            return arr.reshape(num_samples, num_datasets * num_eval_windows, prediction_length, dim)

        predictions = np.concatenate(predictions, axis=1)
        predictions = _reshape_predictions(predictions)

        metric_names_from_logits = [
            "entropy",
            "top_k_entropy",
            "effective_vocab",
            "peak_counts",
        ]

        for name, d in [
            ("mlp", preds_from_mlp_outputs_all_layers),
            ("read_stream", preds_from_read_stream_outputs_all_layers),
            ("entropy", entropy_by_layer),
            ("top_k_entropy", top_k_entropy_by_layer),
            ("effective_vocab", effective_vocab_by_layer),
            ("peak_counts", peak_counts_by_layer),
        ]:
            if name not in (metric_names_from_logits + list(attribution_types)):
                continue
            for layer_idx in range(num_layers):
                output = np.concatenate(d[layer_idx], axis=1)
                output = _reshape_predictions(output)

                # Needs to be shape (num_samples, num_datasets * num_eval_windows, prediction_length, dim)
                if output.shape != predictions.shape:
                    raise ValueError(
                        f"{name} predictions shape {output.shape} != predictions.shape {predictions.shape}"
                    )
                if name in metric_names_from_logits:
                    # output is of shape: (num_samples, num_datasets * num_eval_windows, prediction_length, dim)
                    # Compute mean and std over both num_samples and num_datasets*num_eval_windows axes (axes 0 and 1)
                    mean_output_across_timesteps = np.mean(output, axis=(2))
                    mean_output = np.mean(mean_output_across_timesteps, axis=(0, 1))
                    std_output = np.std(mean_output_across_timesteps, axis=(0, 1))
                    # Now of shape (prediction_length, dim). Transpose to (dim, prediction_length) and store

                    combined_metrics_from_logits[subdir_name][layer_idx].update(
                        {
                            f"{name}_mean": mean_output.T,
                            f"{name}_std": std_output.T,
                        }
                    )
                    # delete d[layer_idx] to save memory, since we don't need it anymore
                    del d[layer_idx]
                else:
                    d[layer_idx] = output  # type: ignore

        all_predictions = predictions.copy()

        validate_and_get_sample_count(predictions, num_datasets, num_eval_windows, prediction_length, dim)

        predictions = parallel_sample_reduction_fn(predictions)

        # Reshape and validate labels and contexts
        labels = reshape_batch_data(labels, num_datasets, num_eval_windows, dim, (prediction_length,))
        contexts = reshape_batch_data(
            contexts,
            num_datasets,
            num_eval_windows,
            dim,
            (contexts[0].shape[-1],),  # context_length
        )

        validate_and_get_sample_count(predictions, num_datasets, num_eval_windows, prediction_length, dim)

        expected_labels_shape = (
            num_datasets * num_eval_windows,
            prediction_length,
            dim,
        )
        expected_contexts_shape = (
            num_datasets * num_eval_windows,
            contexts.shape[1],
            dim,
        )

        for arr, name, expected in [
            (labels, "Labels", expected_labels_shape),
            (contexts, "Contexts", expected_contexts_shape),
        ]:
            if arr.shape != expected:
                raise ValueError(f"{name} shape {arr.shape} != expected {expected}")

        logger.info(f"Final shapes: predictions={predictions.shape}, labels={labels.shape}, contexts={contexts.shape}")

        # Compute metrics if requested
        if metric_names is not None:
            assert all(start < prediction_length for start, _ in eval_subintervals), (
                "All start indices must be less than the prediction length"
            )
            for start, end in eval_subintervals:
                system_metrics[end - start][system] = compute_metrics(
                    predictions[:, start:end, :],
                    labels[:, start:end, :],
                    include=metric_names,
                    batch_axis=0,
                    batch_aggregation=None,
                )

            # Compute metrics rollout for predictions from attribution outputs (mlp and read_stream)
            for name, preds_dict in [
                ("mlp", preds_from_mlp_outputs_all_layers),
                ("read_stream", preds_from_read_stream_outputs_all_layers),
            ]:
                if name not in attribution_types:
                    continue
                for layer_idx in range(num_layers):
                    reduced_pred_from_attribution = parallel_sample_reduction_fn(preds_dict[layer_idx])
                    if reduced_pred_from_attribution.shape != predictions.shape:
                        raise ValueError(
                            f"{name} reduced_pred_from_attribution shape {reduced_pred_from_attribution.shape} != predictions.shape {predictions.shape}"
                        )
                    for start, end in eval_subintervals:
                        # Compute metrics against actual predictions
                        system_metrics_attribution[name][layer_idx][end - start][system] = compute_metrics(
                            reduced_pred_from_attribution[:, start:end, :],
                            predictions[:, start:end, :],
                            include=metric_names,
                            batch_axis=0,
                            batch_aggregation=None,
                        )

        # Transpose for final output
        # the median prediction
        predictions = predictions.transpose(0, 2, 1)
        all_predictions = all_predictions.transpose(0, 1, 3, 2)
        contexts = contexts.transpose(0, 2, 1)
        labels = labels.transpose(0, 2, 1)

        save_results_dict = {
            "context": contexts,
            "future_vals": labels,
            "median_predictions": predictions,
            "predictions": all_predictions,
        }

        logger.info(
            f"Final output shapes: predictions={predictions.shape}, labels={labels.shape}, contexts={contexts.shape}"
        )

        if save_predictions_from_attribution:
            for name, preds_dict in [
                ("mlp", preds_from_mlp_outputs_all_layers),
                ("read_stream", preds_from_read_stream_outputs_all_layers),
            ]:
                if name not in attribution_types:
                    continue
                for layer_idx in range(num_layers):
                    preds_dict[layer_idx] = preds_dict[layer_idx].transpose(0, 1, 3, 2)  # type: ignore
                    if preds_dict[layer_idx].shape != all_predictions.shape:  # type: ignore
                        raise ValueError(
                            f"{name} predictions shape {preds_dict[layer_idx].shape} != all_predictions.shape {all_predictions.shape}"  # type: ignore
                        )
                save_results_dict[name] = preds_dict

        # Save results if requested
        if save_dir is not None:
            system_save_dir = os.path.join(save_dir, subdir_name, f"rseed-{rseed}")
            full_outputs_save_path = os.path.join(system_save_dir, "full_outputs.pkl")
            os.makedirs(system_save_dir, exist_ok=True)
            with open(full_outputs_save_path, "wb") as f:
                pickle.dump(save_results_dict, f)

        pbar.set_postfix({"system": system, "num systems": num_datasets})

    # Convert defaultdicts to regular dicts for type compatibility
    return (
        dict(system_metrics),
        {k: {kk: dict(vv) for kk, vv in v.items()} for k, v in system_metrics_attribution.items()},
        {k: dict(v) for k, v in combined_metrics_from_logits.items()},
    )
