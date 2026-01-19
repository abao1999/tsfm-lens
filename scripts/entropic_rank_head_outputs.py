"""
This script computes the entropic rank for each head output and saves the results to a pickle file.
"""

from __future__ import annotations

import logging
import os
import pickle
from collections import defaultdict
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm

from tsfm_lens.chronos.circuitlens import CircuitLensChronos
from tsfm_lens.chronos_bolt.circuitlens import CircuitLensBolt
from tsfm_lens.dataset import TestDataset
from tsfm_lens.timesfm.circuitlens import CircuitLensTimesFM
from tsfm_lens.toto.circuitlens import CircuitLensToto
from tsfm_lens.utils import (
    apply_custom_style,
    collect_attributions,
    get_eval_data_dict,
    set_seed,
)

logger = logging.getLogger(__name__)


MODEL_LABELS = {
    "chronos": "Chronos",
    "chronos_bolt": "Chronos-Bolt",
    "timesfm": "TimesFM",
    "toto": "Toto",
}


def _ensure_shape(tensor: torch.Tensor) -> torch.Tensor:
    """
    Convert head activation tensor to shape (num_samples, num_timesteps, d_model).
    Handles tensors that might contain extra singleton dimensions.
    """
    tensor = torch.nan_to_num(tensor.detach(), nan=0.0, posinf=0.0, neginf=0.0).to(torch.float32)

    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim == 4:
        # collapse sample-like dimensions while keeping time and model dims
        tensor = tensor.reshape(-1, tensor.shape[-2], tensor.shape[-1])
    elif tensor.ndim == 5:
        tensor = tensor.reshape(-1, tensor.shape[-2], tensor.shape[-1])
    elif tensor.ndim not in (3,):
        raise ValueError(f"Unsupported tensor shape for head outputs: {tensor.shape}")

    return tensor.contiguous()


def compute_entropic_rank_per_layer(
    head_outputs: dict[int, list[torch.Tensor]],
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute entropic rank for each layer given the collected head outputs.

    Entropic rank measures the effective dimensionality of attention head outputs,
    quantifying how many independent directions the heads span. It is computed as:

    1. Normalize each head output vector to unit norm.
    2. Compute the Gram matrix G of pairwise cosine similarities between heads.
    3. Compute the singular values σ_i of G (equivalently, eigenvalues since G is PSD).
    4. Form a probability distribution p_i = √σ_i / Σ_j √σ_j from the singular values.
    5. Compute the Shannon entropy H = -Σ_i p_i log(p_i).
    6. Return exp(H), the entropic rank.

    The entropic rank is bounded in [1, num_heads]:
    - A value near 1 indicates highly redundant heads (all pointing in similar directions).
    - A value near num_heads indicates maximally diverse heads (orthogonal directions).

    Parameters
    ----------
    head_outputs : dict[int, list[torch.Tensor]]
        Dictionary mapping layer indices to lists of head output tensors.
        Each tensor has shape (num_samples, num_timesteps, d_model).
    epsilon : float
        Small constant for numerical stability in normalization and log operations.

    Returns
    -------
    layer_indices : np.ndarray
        Sorted array of layer indices for which activations are available.
    entropic_ranks : np.ndarray
        Array of entropic ranks corresponding to each layer index, averaged
        over all samples and timesteps.
    """
    if not head_outputs:
        return np.array([]), np.array([])

    layer_indices = sorted(head_outputs.keys())
    entropic_ranks: list[float] = []

    for layer_idx in layer_indices:
        head_tensors = head_outputs[layer_idx]
        if not head_tensors:
            logger.debug("Layer %s has no head tensors; skipping.", layer_idx)
            continue

        normalized_heads: list[torch.Tensor] = []
        for tensor in head_tensors:
            normalized_heads.append(_ensure_shape(tensor))

        if not normalized_heads:
            continue

        # (num_heads, num_samples, num_timesteps, d_model)
        heads_stacked = torch.stack(normalized_heads, dim=0)
        # reorder to (num_samples, num_timesteps, num_heads, d_model)
        heads_stacked = heads_stacked.permute(1, 2, 0, 3).contiguous()

        norms = heads_stacked.norm(dim=-1, keepdim=True).clamp_min(epsilon)
        normalized = heads_stacked / norms

        gram = torch.matmul(normalized, normalized.transpose(-1, -2))
        gram = 0.5 * (gram + gram.transpose(-1, -2))

        bsz, timesteps, num_heads, _ = gram.shape
        gram_flat = gram.reshape(-1, num_heads, num_heads)

        singular_vals = torch.linalg.svdvals(gram_flat)
        singular_vals = singular_vals.reshape(bsz, timesteps, num_heads)

        sqrt_vals = torch.sqrt(singular_vals.clamp_min(0.0))
        denom = sqrt_vals.sum(dim=-1, keepdim=True).clamp_min(epsilon)
        probs = sqrt_vals / denom

        entropy = -(probs * torch.log(probs.clamp_min(epsilon))).sum(dim=-1)
        mean_entropy = entropy.mean(dim=(0, 1))
        entropic_rank = torch.exp(mean_entropy).item()

        entropic_ranks.append(entropic_rank)

    return np.array(layer_indices, dtype=int), np.array(entropic_ranks, dtype=np.float32)


def prepare_test_datasets(cfg: DictConfig) -> dict[str, TestDataset]:
    eval_sets = get_eval_data_dict(
        cfg.entropic_rank.data_dir,
        num_subdirs=cfg.entropic_rank.num_subdirs,
        num_samples_per_subdir=cfg.entropic_rank.num_samples_per_subdir,
        subdir_names=cfg.entropic_rank.system_names,
    )

    datasets: dict[str, TestDataset] = {}
    for system_name, gluonts_datasets in eval_sets.items():
        datasets[system_name] = TestDataset(
            datasets=gluonts_datasets,
            context_length=cfg.entropic_rank.context_length,
            prediction_length=cfg.entropic_rank.prediction_length,
            num_test_instances=cfg.entropic_rank.num_test_instances,
            window_style=cfg.entropic_rank.window_style,
            window_stride=cfg.entropic_rank.window_stride,
            window_start_time=cfg.entropic_rank.window_start_time,
            random_seed=cfg.entropic_rank.random_seed,
        )
    return datasets


def instantiate_pipeline(
    model_name: str,
    cfg: DictConfig,
    device: torch.device,
    torch_dtype: torch.dtype,
) -> tuple[object, int, int]:
    if model_name == "chronos":
        pipeline = CircuitLensChronos.from_pretrained(
            cfg.chronos.model_id,
            device_map=str(device),
            torch_dtype=torch_dtype,
        )
    elif model_name == "chronos_bolt":
        pipeline = CircuitLensBolt.from_pretrained(
            cfg.chronos_bolt.model_id,
            device_map=str(device),
            torch_dtype=torch_dtype,
        )
    elif model_name == "timesfm":
        pipeline = CircuitLensTimesFM(cfg.timesfm.model_id, device_map=str(device))
    elif model_name == "toto":
        pipeline = CircuitLensToto(cfg.toto.model_id, device_map=str(device))
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    if not hasattr(pipeline, "num_layers") or not hasattr(pipeline, "num_heads"):
        raise AttributeError(
            f"Pipeline for model {model_name} does not expose num_layers/num_heads attributes. "
            "Please ensure the CircuitLens class sets these in __post_init__."
        )

    num_layers = int(getattr(pipeline, "num_layers"))
    num_heads = int(getattr(pipeline, "num_heads"))

    logger.info(
        "Instantiated %s pipeline with %s layers and %s heads.",
        MODEL_LABELS.get(model_name, model_name),
        num_layers,
        num_heads,
    )
    return pipeline, num_layers, num_heads


def add_head_hooks(
    model_name: str,
    pipeline: object,
    num_layers: int,
    num_heads: int,
    attention_types: list[str],
) -> None:
    if hasattr(pipeline, "remove_all_hooks"):
        pipeline.remove_all_hooks()

    all_heads = [(layer_idx, head_idx) for layer_idx in range(num_layers) for head_idx in range(num_heads)]

    if model_name in {"chronos", "chronos_bolt"}:
        for attention_type in attention_types:
            getattr(pipeline, "add_head_attribution_hooks")(all_heads, attention_type=attention_type)
    elif model_name in {"timesfm", "toto"}:
        getattr(pipeline, "add_head_attribution_hooks")(all_heads)
    else:
        raise ValueError(f"Unsupported model for hook setup: {model_name}")


def reset_pipeline_outputs(pipeline: object) -> None:
    if hasattr(pipeline, "reset_attribution_inputs_and_outputs"):
        pipeline.reset_attribution_inputs_and_outputs()
    if hasattr(pipeline, "reset_custom_attribution_outputs"):
        pipeline.reset_custom_attribution_outputs()


def extract_head_outputs(
    model_name: str,
    pipeline: object,
    attention_type: str,
) -> dict[int, list[torch.Tensor]]:
    if model_name in {"chronos", "chronos_bolt"}:
        raw_outputs: dict[int, dict[int, list[torch.Tensor]]] = getattr(
            pipeline, f"{attention_type}_head_attribution_outputs", {}
        )
    elif model_name in {"timesfm", "toto"}:
        raw_outputs = getattr(pipeline, "head_attribution_outputs", {})
    else:
        raise ValueError(f"Unsupported model for extracting head outputs: {model_name}")

    processed: dict[int, list[torch.Tensor]] = {}
    for layer_idx, head_dict in raw_outputs.items():
        head_indices = sorted(head_dict.keys())
        head_tensors: list[torch.Tensor] = []
        for head_idx in head_indices:
            tensors = head_dict.get(head_idx)
            if not tensors:
                continue
            head_tensors.append(collect_attributions(tensors))
        if head_tensors:
            processed[layer_idx] = head_tensors
    return processed


def run_inference_for_model(
    model_name: str,
    pipeline: object,
    context: torch.Tensor,
    cfg: DictConfig,
    device: torch.device,
    torch_dtype: torch.dtype,
) -> None:
    prediction_length = cfg.entropic_rank.prediction_length
    num_samples = cfg.entropic_rank.num_samples
    deterministic = cfg.entropic_rank.deterministic

    if model_name in {"chronos", "chronos_bolt"}:
        # ensure dtype and device alignment
        context_tensor = context.to(device="cpu", dtype=torch.float32)
        if model_name == "chronos":
            pipeline.predict_with_full_outputs(
                context=context_tensor,
                prediction_length=prediction_length,
                num_samples=num_samples,
                limit_prediction_length=False,
                do_sample=not deterministic,
                return_dict_in_generate=True,
            )
        else:
            pipeline.predict_with_full_outputs(
                context=context_tensor,
                prediction_length=prediction_length,
                limit_prediction_length=False,
                return_dict_in_generate=True,
            )
    elif model_name == "timesfm":
        context_tensor = context.to(device=device, dtype=torch.float32)
        pipeline.predict(context=context_tensor, prediction_length=prediction_length)
    elif model_name == "toto":
        context_tensor = context.to(device=device, dtype=torch.float32)
        samples_per_batch = cfg.toto.samples_per_batch or min(num_samples, 10)
        pipeline.predict(
            context=context_tensor,
            prediction_length=prediction_length,
            num_samples=num_samples,
            samples_per_batch=samples_per_batch,
            use_kv_cache=bool(cfg.toto.use_kv_cache),
        )
    else:
        raise ValueError(f"Unsupported model for inference: {model_name}")


def process_model(
    model_name: str,
    cfg: DictConfig,
    datasets: dict[str, TestDataset],
    output_dir: Path,
    device: torch.device,
    torch_dtype: torch.dtype,
) -> None:
    attention_types = cfg.entropic_rank.attention_types.get(model_name, [])
    if not attention_types:
        logger.warning("No attention types configured for %s; skipping.", model_name)
        return

    pipeline, num_layers, num_heads = instantiate_pipeline(model_name, cfg, device, torch_dtype)
    add_head_hooks(model_name, pipeline, num_layers, num_heads, attention_types)

    epsilon = cfg.entropic_rank.epsilon
    per_attention_results: dict[str, list[np.ndarray]] = defaultdict(list)
    layer_index_reference: dict[str, np.ndarray] = {}

    max_series_per_system = cfg.entropic_rank.max_series_per_system
    max_windows_per_series = cfg.entropic_rank.max_windows_per_series
    max_windows_total = cfg.entropic_rank.max_windows_total
    total_windows = 0

    logger.info("Computing entropic rank for %s", model_name)
    os.makedirs(output_dir, exist_ok=True)
    logger.info("Saving results to %s", output_dir)

    with torch.no_grad():
        system_items = list(datasets.items())
        for system_idx, (system_name, test_dataset) in enumerate(
            tqdm(
                system_items,
                desc=f"{MODEL_LABELS.get(model_name, model_name)} systems",
                leave=False,
            )
        ):
            if max_series_per_system is not None and system_idx >= max_series_per_system:
                logger.info("Reached max systems (%s); stopping dataset loop.", max_series_per_system)
                break

            logger.info("Processing system %s for model %s", system_name, MODEL_LABELS.get(model_name, model_name))

            for window_idx, entry in enumerate(test_dataset):
                if max_windows_per_series is not None and window_idx >= max_windows_per_series:
                    break
                if max_windows_total is not None and total_windows >= max_windows_total:
                    logger.info("Reached global max windows (%s); stopping.", max_windows_total)
                    break

                total_windows += 1

                reset_pipeline_outputs(pipeline)

                context = entry["past_values"]
                if context.dim() == 1:
                    context = context.unsqueeze(0)

                run_inference_for_model(model_name, pipeline, context, cfg, device, torch_dtype)

                for attention_type in attention_types:
                    outputs = extract_head_outputs(model_name, pipeline, attention_type)
                    layer_indices, entropic_ranks = compute_entropic_rank_per_layer(outputs, epsilon)

                    if entropic_ranks.size == 0:
                        logger.warning(
                            "No entropic ranks computed for %s (attention=%s); skipping window.",
                            MODEL_LABELS.get(model_name, model_name),
                            attention_type,
                        )
                        continue

                    if attention_type not in layer_index_reference:
                        layer_index_reference[attention_type] = layer_indices
                    elif not np.array_equal(layer_index_reference[attention_type], layer_indices):
                        logger.warning(
                            "Layer indices changed for %s attention %s. Expected %s, got %s. Skipping window.",
                            MODEL_LABELS.get(model_name, model_name),
                            attention_type,
                            layer_index_reference[attention_type],
                            layer_indices,
                        )
                        continue

                    per_attention_results[attention_type].append(entropic_ranks)

                reset_pipeline_outputs(pipeline)

                if max_windows_total is not None and total_windows >= max_windows_total:
                    break

    if hasattr(pipeline, "remove_all_hooks"):
        pipeline.remove_all_hooks()

    if not per_attention_results:
        logger.warning("No entropic rank data collected for %s.", model_name)
        return

    for attention_type, ranks in per_attention_results.items():
        layers = layer_index_reference[attention_type]
        stacked = np.stack(ranks, axis=0)
        median_rank = np.median(stacked, axis=0)

        logger.info(
            "Model %s attention %s: plotting %d entropic-rank curves.",
            model_name,
            attention_type,
            len(ranks),
        )

        data_path = os.path.join(output_dir, f"{model_name}_{attention_type}_entropic_rank.pkl")
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        # save data file to pkl
        with open(data_path, "wb") as f:
            pickle.dump(
                {
                    "layers": layers,
                    "ranks": ranks,
                    "median_rank": median_rank,
                },
                f,
            )

    # save per_attention_results and layer_index_reference to pkl
    full_dict_data_path = os.path.join(output_dir, "entropic_rank_data.pkl")
    os.makedirs(os.path.dirname(full_dict_data_path), exist_ok=True)
    logger.info("Saving full data (per_attention_results and layer_index_reference) to %s", full_dict_data_path)
    with open(full_dict_data_path, "wb") as f:
        pickle.dump(
            {"per_attention_results": per_attention_results, "layer_index_reference": layer_index_reference},
            f,
        )


@hydra.main(config_path="../config", config_name="entropic_rank", version_base=None)
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO)
    logger.info("Configuration:\n%s", OmegaConf.to_yaml(cfg))

    original_cwd = Path(get_original_cwd())
    plotting_config_path = original_cwd / "config" / "plotting.yaml"
    apply_custom_style(str(plotting_config_path))

    set_seed(cfg.entropic_rank.random_seed)

    device = torch.device(cfg.eval.device if torch.cuda.is_available() else "cpu")
    torch_dtype = getattr(torch, cfg.eval.torch_dtype)
    assert isinstance(torch_dtype, torch.dtype)

    datasets = prepare_test_datasets(cfg)
    # # For debugging:
    # datasets = dict(list(datasets.items())[:4])
    logger.info(f"Using {len(datasets)} datasets")
    num_test_instances = sum(test_dataset.num_test_instances for test_dataset in datasets.values())
    num_test_instances_check = sum(len(test_dataset.datasets) for test_dataset in datasets.values())
    assert num_test_instances == num_test_instances_check, "Number of test instances does not match"
    logger.info(f"Total number of test instances: {num_test_instances}")

    save_root = os.path.join(cfg.entropic_rank.output_root, cfg.entropic_rank.save_subdir)
    os.makedirs(save_root, exist_ok=True)

    for model_name in cfg.models_to_run:
        model_dir = os.path.join(save_root, model_name)
        logger.info("Processing %s. Data saved to %s", MODEL_LABELS.get(model_name, model_name), model_dir)
        process_model(model_name, cfg, datasets, model_dir, device, torch_dtype)
        logger.info("Finished processing %s. Data saved to %s", MODEL_LABELS.get(model_name, model_name), model_dir)


if __name__ == "__main__":
    main()
