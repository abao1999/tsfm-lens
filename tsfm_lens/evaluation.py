import logging
import os
import pickle
import warnings
from collections import defaultdict
from typing import Literal

import numpy as np
import torch
from gluonts.itertools import batcher
from tqdm.auto import tqdm

from tsfm_lens.circuitlens import BaseCircuitLens
from tsfm_lens.dataset import TestDataset
from tsfm_lens.metrics import compute_metrics  # type: ignore
from tsfm_lens.utils import (
    left_pad_and_stack_1D,
    reshape_batch_data,
    set_seed,
    validate_and_get_sample_count,
)

logger = logging.getLogger(__name__)


def _get_ablations_summary_str(
    pipeline: BaseCircuitLens, attention_types: list[Literal["ca", "sa"]] = ["ca", "sa"]
) -> tuple[str | None, str | None]:
    """
    Generate summary strings describing which components (heads and/or MLPs) are ablated in which layers.
    Simply for book-keeping during the ablations runs.

    Args:
        pipeline: BaseCircuitLens instance containing information about which components are ablated.

    Returns:
        tuple[str | None, str | None]: A tuple containing:
            - ablations_summary_str: A concise string for filenames/keys (e.g. "za_heads_layers_0-1-2")
            - ablations_summary_str_title: A human-readable description (e.g. "Zero-Ablation: Heads in Layers [0,1,2]")
            Both values are None if no ablations are present.

    Raises:
        AssertionError: If cross-attention and self-attention head ablations don't match.
        NotImplementedError: If heads and MLPs are ablated in different layers.
    """
    if not pipeline.is_decoder_only:
        layers_without_ca_heads = list(pipeline.ca_head_ablation_handles.keys()) if "ca" in attention_types else []  # type: ignore
        layers_without_sa_heads = list(pipeline.sa_head_ablation_handles.keys()) if "sa" in attention_types else []  # type: ignore
        assert layers_without_ca_heads == layers_without_sa_heads, "CA and SA ablations must be identical"
        layers_without_heads = layers_without_ca_heads
    else:
        layers_without_heads = list(pipeline.head_ablation_handles.keys())  # type: ignore

    layers_without_mlps = list(pipeline.mlp_ablation_handles.keys())  # type: ignore

    if layers_without_heads and layers_without_mlps:
        if layers_without_heads != layers_without_mlps:
            raise NotImplementedError("Zero-ablation of heads and MLPs in different layers is messier, save for later")
        ablations_summary_str_title = f"Zero-Ablation: Heads and MLPs in Layers {layers_without_heads}"
        ablations_summary_str = f"za_heads_mlps_layers_{'-'.join(map(str, layers_without_heads))}"
    elif layers_without_heads:
        ablations_summary_str_title = f"Zero-Ablation: Heads in Layers {layers_without_heads}"
        ablations_summary_str = f"za_heads_layers_{'-'.join(map(str, layers_without_heads))}"
    elif layers_without_mlps:
        ablations_summary_str_title = f"Zero-Ablation: MLPs in Layers {layers_without_mlps}"
        ablations_summary_str = f"za_mlps_layers_{'-'.join(map(str, layers_without_mlps))}"
    else:
        ablations_summary_str = ablations_summary_str_title = None
    return ablations_summary_str, ablations_summary_str_title


def evaluate_ablations(
    pipeline: BaseCircuitLens,
    systems: dict[str, TestDataset],
    batch_size: int,
    prediction_length: int,
    num_samples: int,
    system_dims: dict[str, int],
    list_of_layers_to_ablate: list[list[int]] = [],
    ablate_n_heads_per_layer: int | None = None,
    ablations_types: list[Literal["head", "mlp"]] = ["head"],
    attention_types: list[Literal["ca", "sa"]] = ["ca", "sa"],
    metric_names: list[str] | None = None,
    eval_subintervals: list[tuple[int, int]] | None = None,
    parallel_sample_reduction: Literal["mean", "median"] = "median",
    prediction_kwargs: dict = {},
    rseed: int = 99,
    save_dir: str | None = None,
    save_predictions_from_ablations: bool = False,
    left_pad_context: bool = False,
    truncate_predictions_to_labels_horizon: bool = False,
    compute_metrics_ablations_against_labels: bool = False,
) -> tuple[
    dict[int, dict[str, dict[str, float]]],
    dict[str, dict[int, dict[str, dict[str, float]]]],
    dict[str, dict[int, dict[str, dict[str, float]]]],
]:
    """
    Evaluate the Chronos pipeline on a set of test systems, computing both standard and ablation-based metrics.

    This function runs the model on each provided system, collects predictions, and computes metrics over the
    prediction window and optionally over specified subintervals. It also computes metrics for ablated versions
    of the model, as specified by the ablation parameters.

    Args:
        pipeline: BaseCircuitLens instance used for generating predictions and extracting attributions.
        systems: Dictionary mapping system names to TestDataset objects.
        batch_size: Number of samples per batch during evaluation.
        prediction_length: Number of timesteps to predict for each sample.
        system_dims: Dictionary mapping system names to their data dimensionality.
        list_of_layers_to_ablate: List of lists, where each sublist contains layer indices to ablate in a single ablation run.
        ablate_n_heads_per_layer: Number of heads to ablate per layer. If None, all heads are ablated. Only used if "head" in ablations_types.
            TODO: once we find induction heads for all models, pass in the induction heads as a restricted subset of heads that are not ablated
        ablations_types: List of ablation types to apply. Each entry should be "head" or "mlp".
        attention_types: List of attention types to ablate. Each entry should be "ca" or "sa".
        metric_names: Optional list of metric names to compute (default: all available).
        eval_subintervals: Optional list of (start, end) index tuples specifying subintervals
            within the prediction window to evaluate metrics on. If None, evaluates on the
            full prediction window.
        parallel_sample_reduction: How to reduce the parallel samples over axis 0 ("mean" or "median").
        prediction_kwargs: Optional dictionary of additional keyword arguments to pass to
            the pipeline's predict method.
        rseed: Random seed for reproducibility (default: 99).
        save_dir: Optional directory path to save evaluation results.
        save_predictions_from_ablations: If True, saves predictions from ablated models as well (can be large).
        left_pad_context: If True, left-pads the context to the maximum length of the past_values.
        truncate_predictions_to_labels_horizon: If True, truncates predictions to the labels horizon.
    Returns:
        system_metrics: Nested dictionary of the form
            {subinterval: {system_name: {metric_name: value, ...}, ...}, ...}
            containing computed metrics for each system and subinterval.
        system_metrics_ablations: Nested dictionary of the form
            {ablations_summary_str: {subinterval: {system_name: {metric_name: value, ...}, ...}, ...}, ...}
            containing computed metrics for each system and subinterval from ablated model outputs, for each ablation configuration.
    """
    assert isinstance(pipeline, BaseCircuitLens), "pipeline must be a BaseCircuitLens instance"
    if hasattr(pipeline.model, "eval"):
        pipeline.model.eval()  # redundancy for extra safety
    elif hasattr(pipeline, "set_to_eval_mode"):
        pipeline.set_to_eval_mode()

    rng = np.random.default_rng(rseed)

    num_layers = pipeline.num_layers
    num_heads = pipeline.num_heads

    assert ablate_n_heads_per_layer is None or 0 < ablate_n_heads_per_layer <= num_heads, (
        f"ablate_n_heads_per_layer ({ablate_n_heads_per_layer}) must be between 0 and {num_heads}"
    )

    assert all(all(0 <= layer <= num_layers - 1 for layer in layers) for layers in list_of_layers_to_ablate), (
        f"All layers in layers_to_ablate must be between 0 and {num_layers - 1}"
    )

    if save_predictions_from_ablations:
        warnings.warn("save_predictions_from_ablations is True, this will save a lot of data")

    system_metrics = defaultdict(dict)  # predictions vs labels (future values)
    system_metrics_ablations = defaultdict(lambda: defaultdict(dict))  # predictions_with_ablations vs predictions
    # predictions_with_ablations vs labels (future values)
    system_metrics_ablations_against_labels = defaultdict(lambda: defaultdict(dict))

    if eval_subintervals is None:
        eval_subintervals = [(0, prediction_length)]
    elif (0, prediction_length) not in eval_subintervals:
        eval_subintervals.append((0, prediction_length))

    parallel_sample_reduction_fn = {
        "mean": lambda x: np.mean(x, axis=0),
        "median": lambda x: np.median(x, axis=0),
    }[parallel_sample_reduction]

    pbar = tqdm(systems, desc="Forecasting...")
    for system_idx, system in enumerate(pbar):
        dataset = systems[system]
        labels_horizon = getattr(dataset.datasets[0], "prediction_length", prediction_length)
        num_datasets = len(dataset.datasets)

        subdir_name = (
            dataset.datasets[0].name
            if hasattr(dataset.datasets[0], "name")
            else dataset.datasets[0].iterable.path.parent.name
        )

        system_dimension = system_dims[system]

        logger.info(
            f"subdir_name: {subdir_name} with {num_datasets} trajectory files or samples and dimension {system_dimension}"
        )

        predictions, labels = [], []
        predictions_with_ablations_dict = defaultdict(list)

        # Process batches
        for i, batch in enumerate(batcher(dataset, batch_size=batch_size)):
            # past_values is a length system_dimension tuple of torch tensors of shape (1, context_length)
            past_values, future_values = zip(*[(data["past_values"], data["future_values"]) for data in batch])
            # past_values is a tuple of tensors of same or different lengths
            if all(past_values[0].shape == pv.shape for pv in past_values):
                context = torch.cat(past_values, dim=0)
                if context.ndim != 2:
                    raise ValueError(f"Unexpected context shape: {context.shape}")
                logger.info(f"context shape: {context.shape}")
            elif left_pad_context:
                context = left_pad_and_stack_1D(list(past_values))
                if context.ndim != 2:
                    raise ValueError(f"Unexpected context shape: {context.shape}")
                logger.info(f"context shape: {context.shape}")
            else:
                context_list = []
                for ctx in past_values:
                    # If 2D (batch_size, context_length), convert to list of 1D arrays
                    if ctx.ndim == 2:
                        context_list.extend([ctx[i] for i in range(ctx.shape[0])])
                    elif ctx.ndim == 1:
                        context_list.append(ctx)
                    else:
                        raise ValueError(f"Unexpected context shape: {ctx.shape}")
                context = context_list
                logger.info(f"context: list of {len(context)} series. First context shape: {context[0].shape}")

            future_vals = torch.cat(future_values, dim=0)
            if future_vals.shape[-1] != labels_horizon:
                raise ValueError(
                    f"future_vals.shape[-1] ({future_vals.shape[-1]}) != labels_horizon ({labels_horizon})"
                )
            for layers_to_ablate in tqdm(list_of_layers_to_ablate, desc="Ablating layers..."):
                # Remove all hooks and reset attribution inputs and outputs
                pipeline.remove_all_hooks()
                if hasattr(pipeline, "reset_attribution_inputs_and_outputs"):
                    pipeline.reset_attribution_inputs_and_outputs()

                if "head" in ablations_types:
                    # Ablate all heads per layer
                    if ablate_n_heads_per_layer is None:
                        # List of tuples of (layer_idx, head_idx) to ablate
                        heads_to_ablate = [(layer, head) for layer in layers_to_ablate for head in range(num_heads)]
                    # Ablate a fixed number of heads (random subset of heads) per layer
                    else:
                        heads_to_ablate = []
                        for layer in layers_to_ablate:
                            heads_to_ablate_for_layer = rng.choice(
                                list(range(num_heads)), size=ablate_n_heads_per_layer, replace=False
                            )
                            heads_to_ablate.extend([(layer, head) for head in heads_to_ablate_for_layer])  # type: ignore
                    if not pipeline.is_decoder_only:
                        for attention_type in attention_types:
                            pipeline.add_head_ablation_hooks(
                                heads_to_ablate,
                                attention_type=attention_type,
                                ablation_method="zero",
                            )
                    else:
                        pipeline.add_head_ablation_hooks(
                            heads_to_ablate,
                            ablation_method="zero",
                        )

                if "mlp" in ablations_types:
                    pipeline.add_mlp_ablation_hooks(layers_to_ablate, ablation_method="zero")

                # logger.info(f"Ablating layers: {layers_to_ablate} for {ablations_types}")
                ablations_summary_str, ablations_summary_str_title = _get_ablations_summary_str(
                    pipeline, attention_types
                )
                # logger.info(f"Ablations summary: {ablations_summary_str_title}")
                # logger.info(f"summary string: {ablations_summary_str}")

                set_seed(rseed)
                # preds_with_ablations shape is either (batch_size, prediction_length) or (batch_size, num_samples, prediction_length)
                preds_with_ablations = pipeline.predict(  # type: ignore
                    context,
                    prediction_length=prediction_length,
                    **prediction_kwargs,  # NOTE: should have limit_prediction_length=False
                )
                if num_samples == 1:
                    if preds_with_ablations.ndim != 2:
                        raise ValueError(
                            f"preds_with_ablations shape {preds_with_ablations.shape} != (batch_size, prediction_length)"
                        )
                    preds_with_ablations = preds_with_ablations.unsqueeze(0)
                else:
                    preds_with_ablations = preds_with_ablations.transpose(0, 1).cpu().numpy()
                # shape is (num_samples, batch_size, prediction_length)
                preds_with_ablations = preds_with_ablations[
                    ..., : labels_horizon if truncate_predictions_to_labels_horizon else prediction_length
                ]
                predictions_with_ablations_dict[ablations_summary_str].append(preds_with_ablations)

            pipeline.remove_all_hooks()
            if hasattr(pipeline, "reset_attribution_inputs_and_outputs"):
                pipeline.reset_attribution_inputs_and_outputs()

            set_seed(rseed)

            # preds shape is either (batch_size (num_dims), prediction_length) or (batch_size (num_dims), num_samples, prediction_length)
            preds = pipeline.predict(  # type: ignore
                context,
                prediction_length=prediction_length,
                **prediction_kwargs,
            )
            if num_samples == 1:
                if preds.ndim != 2:
                    raise ValueError(f"preds shape {preds.shape} != (batch_size, prediction_length)")
                preds = preds.unsqueeze(0)
            else:
                preds = preds.transpose(0, 1).cpu().numpy()
            # logger.info(f"Predictions shape: {preds.shape}")

            # preds shape is (num_samples, batch_size, prediction_length)
            preds = preds[..., : labels_horizon if truncate_predictions_to_labels_horizon else prediction_length]
            future_vals = future_vals.cpu().numpy()

            labels.append(future_vals)
            predictions.append(preds)

        # Calculate dimensions
        total_eval_windows = sum(pred.shape[1] for pred in predictions)
        num_eval_windows = total_eval_windows // (num_datasets * system_dimension)
        if total_eval_windows % (num_datasets * system_dimension) != 0:
            raise ValueError(
                f"Total eval windows {total_eval_windows} is not divisible by num_datasets * system_dimension = {num_datasets * system_dimension}"
            )

        # logger.info(
        #     f"Shape info: num_datasets={num_datasets}, num_eval_windows={num_eval_windows}, dim={system_dimension}"
        # )

        # Reshape predictions with explicit dimension tracking
        if num_samples != predictions[0].shape[0]:
            raise ValueError(f"num_samples ({num_samples}) != predictions[0].shape[0] ({predictions[0].shape[0]})")

        def _reshape_predictions(arr):
            arr = arr.reshape(num_samples, num_datasets, num_eval_windows, system_dimension, prediction_length)
            arr = arr.transpose(0, 1, 2, 4, 3)
            return arr.reshape(num_samples, num_datasets * num_eval_windows, prediction_length, system_dimension)

        predictions = np.concatenate(predictions, axis=1)
        predictions = _reshape_predictions(predictions)

        for k, v in predictions_with_ablations_dict.items():
            arr = _reshape_predictions(np.concatenate(v, axis=1))
            validate_and_get_sample_count(arr, num_datasets, num_eval_windows, prediction_length, system_dimension)
            predictions_with_ablations_dict[k] = parallel_sample_reduction_fn(arr)

        all_predictions = predictions.copy()

        validate_and_get_sample_count(predictions, num_datasets, num_eval_windows, prediction_length, system_dimension)
        predictions = parallel_sample_reduction_fn(predictions)

        # Reshape and validate labels
        labels = reshape_batch_data(labels, num_datasets, num_eval_windows, system_dimension, (labels_horizon,))
        expected_labels_shape = (
            num_datasets * num_eval_windows,
            labels_horizon,
            system_dimension,
        )
        if labels.shape != expected_labels_shape:
            raise ValueError(f"Labels shape {labels.shape} != expected {expected_labels_shape}")
        # Predictions is currently of shape (num_datasets * num_eval_windows, prediction_length, system_dimension)
        if predictions.shape[-1] != system_dimension:
            raise ValueError(
                f"predictions.shape[-1] ({predictions.shape[-1]}) != system_dimension ({system_dimension})"
            )

        for i, selected_dim in enumerate(range(system_dimension)):
            system_name_with_dim = (
                system if (system_dimension == 1 and selected_dim == 0) else f"{system}_dim{selected_dim}"
            )
            predictions_of_dim = predictions[:, :, [selected_dim]]
            labels_of_dim = labels[:, :, [selected_dim]]
            if i == 0 and labels_horizon != prediction_length:
                logger.warning(
                    f"labels_horizon ({labels_horizon}) != prediction_length ({prediction_length}) for {system_name_with_dim}"
                )
            # modify eval_subintervals to be less than or equal to labels_horizon
            eval_subintervals_against_labels = sorted(
                list(set((start, min(end, labels_horizon)) for start, end in eval_subintervals)), key=lambda x: x[1]
            )
            if not metric_names:
                break

            assert all(start < prediction_length for start, _ in eval_subintervals), (
                "All start indices must be less than the prediction length"
            )
            # Compute metrics between predictions and labels (future values)
            for start, end in eval_subintervals_against_labels:
                system_metrics[end - start][system_name_with_dim] = compute_metrics(
                    predictions_of_dim[:, start:end, :],
                    labels_of_dim[:, start:end, :],
                    include=metric_names,
                    batch_axis=0,
                )
            # Compute metrics between predictions_with_ablations and both predictions and labels
            for ablations_summary_str, preds_with_ablations in predictions_with_ablations_dict.items():
                preds_with_ablations_of_dim = preds_with_ablations[:, :, [selected_dim]]  # type: ignore

                # Compare against original predictions
                for start, end in eval_subintervals:
                    system_metrics_ablations[ablations_summary_str][end - start][system_name_with_dim] = (
                        compute_metrics(
                            preds_with_ablations_of_dim[:, start:end, :],
                            predictions_of_dim[:, start:end, :],
                            include=metric_names,
                            batch_axis=0,
                        )
                    )

                if compute_metrics_ablations_against_labels:
                    # Compare against ground truth labels
                    for start, end in eval_subintervals_against_labels:
                        system_metrics_ablations_against_labels[ablations_summary_str][end - start][
                            system_name_with_dim
                        ] = compute_metrics(
                            preds_with_ablations_of_dim[:, start:end, :],
                            labels_of_dim[:, start:end, :],
                            include=metric_names,
                            batch_axis=0,
                        )

        # Transpose for final output
        # the median prediction
        predictions = predictions.transpose(0, 2, 1)
        all_predictions = all_predictions.transpose(0, 1, 3, 2)
        labels = labels.transpose(0, 2, 1)

        save_results_dict = {
            "future_vals": labels,
            "median_predictions": predictions,
            "predictions": all_predictions,
        }

        if save_predictions_from_ablations:
            for ablations_summary_str, preds_with_ablations in predictions_with_ablations_dict.items():
                preds_with_ablations = preds_with_ablations.transpose(0, 2, 1)  # type: ignore
                save_results_dict[ablations_summary_str] = preds_with_ablations

        # Save results if requested
        if save_dir is not None:
            system_save_dir = os.path.join(save_dir, subdir_name, f"rseed-{rseed}")
            ablations_save_path = os.path.join(system_save_dir, "ablations.pkl")
            os.makedirs(system_save_dir, exist_ok=True)
            with open(ablations_save_path, "wb") as f:
                pickle.dump(save_results_dict, f)

        pbar.set_postfix({"system": system, "num systems": num_datasets})

    return system_metrics, system_metrics_ablations, system_metrics_ablations_against_labels  # type: ignore
