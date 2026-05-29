"""
Logit Attribution Evaluation framework for Chronos.
"""

import json
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
    make_json_serializable,
    reshape_batch_data,
    set_seed,
    validate_and_get_sample_count,
)

logger = logging.getLogger(__name__)

LOGIT_METRIC_NAMES = ("entropy", "effective_vocab", "top_k_entropy", "peak_counts")


def _output_transform_tokens_using_context(pipeline, tokens, context):
    """Transform tokens back to coordinate space using the context."""
    _, _, scale = pipeline.tokenizer.context_input_transform(torch.tensor(context).to(dtype=torch.float32))
    return pipeline.tokenizer.output_transform(torch.tensor(tokens).to(scale.device), scale)


def _get_logits_and_tokens_from_attribution_output(
    pipeline,
    attribution_output: torch.Tensor,
    decoding_method: Literal["greedy", "beam"] = "greedy",
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    attribution_output: (batch_size, num_samples, prediction_length, d_model)
    Returns (logits, tokens) of shapes
      logits: (batch_size, num_samples, prediction_length, vocab_size)
      tokens: (batch_size, num_samples, prediction_length)
    """
    num_samples = attribution_output.shape[1]
    logits = np.stack(
        [pipeline.unembed_residual(attribution_output[:, j, ...]).detach().cpu().numpy() for j in range(num_samples)],
        axis=1,
    )
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


def _attr_outputs_for(pipeline: CircuitLensChronos, name: str):
    return {
        "mlp": pipeline.mlp_attribution_outputs,
        "read_stream": pipeline.read_stream_outputs,
    }[name]


def _save_system_metrics_json(
    save_dir: str,
    system: str,
    subdir_name: str,
    metrics_fname: str,
    system_metrics: dict,
    system_metrics_attribution: dict,
    combined_metrics_from_logits: dict,
) -> None:
    """Write per-system slices of the aggregated metric dicts as JSON next to full_outputs.pkl."""
    per_system_metrics = {
        subinterval_len: by_system[system]
        for subinterval_len, by_system in system_metrics.items()
        if system in by_system
    }
    per_system_attribution = {
        attr_name: {
            layer_idx: {
                subinterval_len: by_system[system]
                for subinterval_len, by_system in by_subinterval.items()
                if system in by_system
            }
            for layer_idx, by_subinterval in by_layer.items()
        }
        for attr_name, by_layer in system_metrics_attribution.items()
    }
    per_system_logits = combined_metrics_from_logits.get(subdir_name, {})

    files = {
        f"{metrics_fname}.json": per_system_metrics,
        f"{metrics_fname}_attribution.json": per_system_attribution,
        f"{metrics_fname}_logits.json": per_system_logits,
    }
    for fname, data in files.items():
        with open(os.path.join(save_dir, fname), "w") as f:
            json.dump(make_json_serializable(data), f, indent=4)


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
    metrics_fname: str = "metrics",
    save_predictions_from_attribution: bool = False,
) -> tuple[
    dict[int, dict[str, dict[str, float]]],
    dict[str, dict[int, dict[int, dict[str, dict[str, float]]]]],
    dict[str, dict[int, dict[str, np.ndarray]]],
]:
    """
    Evaluate the Chronos pipeline on a set of test systems, computing both standard and
    attribution-based metrics. Per-system metrics are written to disk inside the loop so
    results are persisted incrementally; the function also returns the aggregated dicts.

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
        prediction_kwargs: Additional kwargs forwarded to the pipeline's predict method.
        rseed: Random seed for reproducibility (default: 99).
        attribution_types: Set of attribution types to evaluate (subset of {"mlp", "read_stream"}).
        save_dir: Optional directory root for per-system results. Each system writes to
            {save_dir}/{subdir_name}/rseed-{rseed}/ containing full_outputs.pkl plus three JSON
            files (metrics, attribution, logits).
        metrics_fname: Base name for the per-system JSON files (default "metrics").
        save_predictions_from_attribution: Whether to include attribution-derived predictions
            in full_outputs.pkl.

    Returns:
        system_metrics, system_metrics_attribution, combined_metrics_from_logits — the aggregated
        dicts across all systems.
    """
    logger.info(f"metric_names: {metric_names}")
    assert isinstance(pipeline, BaseCircuitLens), "pipeline must be a BaseCircuitLens instance"
    num_layers = pipeline.num_layers

    if hasattr(pipeline.model, "eval"):
        pipeline.model.eval()
    elif hasattr(pipeline, "set_to_eval_mode"):
        pipeline.set_to_eval_mode()

    system_metrics: dict[int, dict[str, dict[str, float]]] = defaultdict(dict)
    system_metrics_attribution: dict[str, dict[int, dict[int, dict]]] = {
        name: defaultdict(lambda: defaultdict(dict)) for name in attribution_types
    }
    combined_metrics_from_logits: dict[str, dict[int, dict]] = defaultdict(lambda: defaultdict(dict))

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
        preds_buffers: dict[str, defaultdict] = {name: defaultdict(list) for name in attribution_types}
        logit_buffers: dict[str, defaultdict] = {name: defaultdict(list) for name in LOGIT_METRIC_NAMES}

        for i, batch in enumerate(batcher(dataset, batch_size=batch_size)):
            past_values, future_values = zip(*[(data["past_values"], data["future_values"]) for data in batch])
            context = torch.cat(past_values, dim=0)
            future_vals = torch.cat(future_values, dim=0)
            actual_batch_size = context.shape[0]

            set_seed(rseed)
            preds, encdec_out = pipeline.predict_with_full_outputs(  # type: ignore
                context,
                prediction_length=prediction_length,
                use_cache=True,
                return_dict_in_generate=True,
                output_scores=True,
                **prediction_kwargs,
            )

            context = context.cpu().numpy()
            future_vals = future_vals.cpu().numpy()
            # shape is (num_samples, batch_size, prediction_length)
            preds = preds.transpose(0, 1).cpu().numpy()
            if preds.shape[-1] > future_vals.shape[-1]:
                preds = preds[..., : future_vals.shape[-1]]

            for layer_idx in tqdm(range(num_layers), desc=f"Computing {attribution_types} attribution outputs"):
                for name in attribution_types:
                    raw = collect_attributions(_attr_outputs_for(pipeline, name)[layer_idx])
                    # Reshape (num_samples * actual_batch_size, ...) -> (actual_batch_size, num_samples, ...)
                    attr_out = raw.reshape(actual_batch_size, num_samples, *raw.shape[1:])

                    # logits: (actual_batch_size, num_samples, prediction_length, vocab_size)
                    # tokens: (actual_batch_size, num_samples, prediction_length)
                    logits, tokens = _get_logits_and_tokens_from_attribution_output(
                        pipeline, attr_out, decoding_method="greedy",
                    )
                    # Output transform tokens to coordinate space: (num_samples, actual_batch_size, prediction_length)
                    preds_buffers[name][layer_idx].append(
                        _output_transform_tokens_using_context(pipeline, tokens, context).transpose(0, 1).cpu().numpy()
                    )

                    if name == "read_stream":
                        entropy, effective_vocab, top_k_entropy, peak_counts = compute_logit_metrics(
                            logits, threshold=0.01, top_k=10,
                        )
                        for metric_name, arr in zip(
                            LOGIT_METRIC_NAMES,
                            (entropy, effective_vocab, top_k_entropy, peak_counts),
                        ):
                            logit_buffers[metric_name][layer_idx].append(np.swapaxes(arr, 0, 1))

            if hasattr(pipeline, "reset_attribution_inputs_and_outputs"):
                pipeline.reset_attribution_inputs_and_outputs()

            labels.append(future_vals)
            predictions.append(preds)
            contexts.append(context)

        total_eval_windows = sum(pred.shape[1] for pred in predictions)
        num_eval_windows = total_eval_windows // (num_datasets * dim)

        if total_eval_windows % (num_datasets * dim) != 0:
            raise ValueError(
                f"Total eval windows {total_eval_windows} is not divisible by num_datasets * dim = {num_datasets * dim}"
            )

        logger.info(f"Shape info: num_datasets={num_datasets}, num_eval_windows={num_eval_windows}, dim={dim}")

        if num_samples != predictions[0].shape[0]:
            raise ValueError(f"num_samples ({num_samples}) != predictions[0].shape[0] ({predictions[0].shape[0]})")

        def _reshape_predictions(arr):
            arr = arr.reshape(num_samples, num_datasets, num_eval_windows, dim, prediction_length)
            arr = arr.transpose(0, 1, 2, 4, 3)
            return arr.reshape(num_samples, num_datasets * num_eval_windows, prediction_length, dim)

        predictions = np.concatenate(predictions, axis=1)
        predictions = _reshape_predictions(predictions)

        # Reshape attribution prediction buffers, keep per-layer concatenated arrays
        for name, preds_dict in preds_buffers.items():
            for layer_idx in range(num_layers):
                output = _reshape_predictions(np.concatenate(preds_dict[layer_idx], axis=1))
                if output.shape != predictions.shape:
                    raise ValueError(
                        f"{name} predictions shape {output.shape} != predictions.shape {predictions.shape}"
                    )
                preds_dict[layer_idx] = output  # type: ignore

        # Reduce per-layer logit metrics to (mean, std) summaries and free the raw buffers
        for metric_name, buffer in logit_buffers.items():
            for layer_idx in range(num_layers):
                output = _reshape_predictions(np.concatenate(buffer[layer_idx], axis=1))
                if output.shape != predictions.shape:
                    raise ValueError(
                        f"{metric_name} shape {output.shape} != predictions.shape {predictions.shape}"
                    )
                # output: (num_samples, num_datasets * num_eval_windows, prediction_length, dim)
                mean_across_t = np.mean(output, axis=2)
                combined_metrics_from_logits[subdir_name][layer_idx].update({
                    f"{metric_name}_mean": np.mean(mean_across_t, axis=(0, 1)).T,
                    f"{metric_name}_std": np.std(mean_across_t, axis=(0, 1)).T,
                })
                del buffer[layer_idx]

        all_predictions = predictions.copy()
        validate_and_get_sample_count(predictions, num_datasets, num_eval_windows, prediction_length, dim)
        predictions = parallel_sample_reduction_fn(predictions)

        labels = reshape_batch_data(labels, num_datasets, num_eval_windows, dim, (prediction_length,))
        contexts = reshape_batch_data(
            contexts, num_datasets, num_eval_windows, dim, (contexts[0].shape[-1],),
        )

        validate_and_get_sample_count(predictions, num_datasets, num_eval_windows, prediction_length, dim)

        expected_labels_shape = (num_datasets * num_eval_windows, prediction_length, dim)
        expected_contexts_shape = (num_datasets * num_eval_windows, contexts.shape[1], dim)

        for arr, name, expected in [
            (labels, "Labels", expected_labels_shape),
            (contexts, "Contexts", expected_contexts_shape),
        ]:
            if arr.shape != expected:
                raise ValueError(f"{name} shape {arr.shape} != expected {expected}")

        logger.info(f"Final shapes: predictions={predictions.shape}, labels={labels.shape}, contexts={contexts.shape}")

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

            for name, preds_dict in preds_buffers.items():
                for layer_idx in range(num_layers):
                    reduced = parallel_sample_reduction_fn(preds_dict[layer_idx])
                    if reduced.shape != predictions.shape:
                        raise ValueError(
                            f"{name} reduced shape {reduced.shape} != predictions.shape {predictions.shape}"
                        )
                    for start, end in eval_subintervals:
                        system_metrics_attribution[name][layer_idx][end - start][system] = compute_metrics(
                            reduced[:, start:end, :],
                            predictions[:, start:end, :],
                            include=metric_names,
                            batch_axis=0,
                            batch_aggregation=None,
                        )

        # Transpose for final output
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
            for name, preds_dict in preds_buffers.items():
                for layer_idx in range(num_layers):
                    preds_dict[layer_idx] = preds_dict[layer_idx].transpose(0, 1, 3, 2)  # type: ignore
                    if preds_dict[layer_idx].shape != all_predictions.shape:  # type: ignore
                        raise ValueError(
                            f"{name} predictions shape {preds_dict[layer_idx].shape} != all_predictions.shape {all_predictions.shape}"  # type: ignore
                        )
                save_results_dict[name] = preds_dict

        if save_dir is not None:
            system_save_dir = os.path.join(save_dir, subdir_name, f"rseed-{rseed}")
            os.makedirs(system_save_dir, exist_ok=True)
            with open(os.path.join(system_save_dir, "full_outputs.pkl"), "wb") as f:
                pickle.dump(save_results_dict, f)

            _save_system_metrics_json(
                system_save_dir,
                system,
                subdir_name,
                metrics_fname,
                system_metrics,
                system_metrics_attribution,
                combined_metrics_from_logits,
            )

        pbar.set_postfix({"system": system, "num systems": num_datasets})

    return (
        dict(system_metrics),
        {k: {kk: dict(vv) for kk, vv in v.items()} for k, v in system_metrics_attribution.items()},
        {k: dict(v) for k, v in combined_metrics_from_logits.items()},
    )
