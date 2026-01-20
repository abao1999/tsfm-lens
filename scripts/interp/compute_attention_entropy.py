"""
Compute average cross-attention entropy per head across multiple context windows.

This script reproduces the exploratory notebook workflow in a reproducible,
Hydra-configurable form. It samples context windows following the configuration
in ``config/attention_entropy.yaml``, runs the selected model(s) to collect
cross-attention scores, aggregates the per-head entropy, and writes both the
numerical results and a heatmap plot to disk.
"""

from __future__ import annotations

import gc
import json
import logging
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm

from scripts.entropic_rank_head_outputs import instantiate_pipeline
from tsfm_lens.dataset import TestDataset
from tsfm_lens.utils import apply_custom_style, get_eval_data_dict, set_seed

logger = logging.getLogger(__name__)


def collect_ca_attention_scores(preds: tuple) -> torch.Tensor:
    """
    Collect cross-attention tensors from a Chronos-style pipeline generate call.

    Parameters
    ----------
    preds
        Tuple returned by ``predict_with_full_outputs(..., output_attentions=True)``.

    Returns
    -------
    torch.Tensor
        Attention scores with shape ``[T, L, num_samples, num_heads, context]``.
        Stored on CPU in float32 to minimise GPU memory pressure.
    """

    decoder_outputs = preds[1]
    rollouts = len(decoder_outputs)
    num_layers = len(decoder_outputs[0].cross_attentions[0])
    num_samples, num_heads, _, context_len = decoder_outputs[0].cross_attentions[0][0].shape

    total_steps = (rollouts - 1) * 64 + len(decoder_outputs[-1].cross_attentions)
    attention_scores = torch.zeros(
        (total_steps, num_layers, num_samples, num_heads, context_len),
        dtype=torch.float32,
        device="cpu",
    )

    step_index = 0
    for rollout in range(rollouts):
        for time_step in decoder_outputs[rollout].cross_attentions:
            for layer_idx in range(num_layers):
                attention_scores[step_index, layer_idx] = time_step[layer_idx][:, :, -1, :].to(
                    dtype=torch.float32, device="cpu"
                )
            step_index += 1

    return attention_scores


def compute_average_entropy(attention_tensor: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Compute the mean entropy per (layer, head) across prediction steps.

    Parameters
    ----------
    attention_tensor
        Array with shape ``[T, L, H, C]`` containing attention weights.
    eps
        Numerical stabiliser for log operations.

    Returns
    -------
    np.ndarray
        Array of shape ``[L, H]`` with averaged entropy.
    """

    probs = attention_tensor / (attention_tensor.sum(axis=-1, keepdims=True) + eps)
    entropy = -(probs * np.log(probs + eps)).sum(axis=-1)  # [T, L, H]
    return entropy.mean(axis=0)  # [L, H]


def build_datasets(cfg: DictConfig) -> dict[str, TestDataset]:
    cfg_model = cfg.attention_entropy
    eval_sets = get_eval_data_dict(
        cfg_model.data_dir,
        num_subdirs=cfg_model.num_subdirs,
        num_samples_per_subdir=cfg_model.num_samples_per_subdir,
        subdir_names=cfg_model.system_names,
    )

    datasets: dict[str, TestDataset] = {}
    for system_name, gluonts_datasets in eval_sets.items():
        datasets[system_name] = TestDataset(
            datasets=gluonts_datasets,
            context_length=cfg_model.context_length,
            prediction_length=cfg_model.prediction_length,
            num_test_instances=cfg_model.num_test_instances,
            window_style=cfg_model.window_style,
            window_stride=cfg_model.window_stride,
            window_start_time=cfg_model.window_start_time,
            random_seed=cfg_model.random_seed,
        )
    return datasets


def plot_entropy_heatmap(avg_entropy: np.ndarray, save_path: Path) -> None:
    """Render and save a heatmap of negative entropy (higher is darker)."""

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(-avg_entropy, aspect="auto", cmap="viridis", origin="lower")
    ax.set_xlabel("Head index")
    ax.set_ylabel("Layer index")
    ax.set_title("Avg Entropy per Head (CA)")
    fig.colorbar(im, ax=ax, label="-Entropy (nats)")
    plt.tight_layout()

    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def process_model(
    model_name: str,
    cfg: DictConfig,
    datasets: dict[str, TestDataset],
    save_dir: Path,
    device: torch.device,
    torch_dtype: torch.dtype,
) -> None:
    if model_name != "chronos":
        logger.warning("Model %s is not yet supported; skipping.", model_name)
        return

    logger.info("Processing model %s", model_name)
    pipeline, num_layers, num_heads = instantiate_pipeline(model_name, cfg, device, torch_dtype)

    cfg_model = cfg.attention_entropy
    max_series_per_system = cfg_model.max_series_per_system
    max_windows_per_series = cfg_model.max_windows_per_series
    max_windows_total = cfg_model.max_windows_total
    prediction_length = cfg_model.prediction_length
    num_samples = cfg_model.num_samples
    deterministic = cfg_model.deterministic

    window_metadata: list[dict[str, int]] = []
    entropy_sum = np.zeros((num_layers, num_heads), dtype=np.float64)
    window_entropies: list[np.ndarray] = []
    processed_windows = 0

    with torch.inference_mode():
        progress = tqdm(total=max_windows_total, desc="Context windows", leave=False)
        for system_idx, (system_name, dataset) in enumerate(datasets.items()):
            if max_series_per_system is not None and system_idx >= max_series_per_system:
                break

            for window_idx, entry in enumerate(dataset):
                if max_windows_per_series is not None and window_idx >= max_windows_per_series:
                    break
                if max_windows_total is not None and processed_windows >= max_windows_total:
                    break

                context = entry["past_values"]
                if context.dim() == 1:
                    context = context.unsqueeze(0)

                context_cpu = context.to(device="cpu", dtype=torch.float32)

                preds = pipeline.predict_with_full_outputs(
                    context=context_cpu,
                    prediction_length=prediction_length,
                    num_samples=num_samples,
                    limit_prediction_length=False,
                    do_sample=not deterministic,
                    output_attentions=True,
                    return_dict_in_generate=True,
                )

                attention = collect_ca_attention_scores(preds)[:, :, 0, :, :]  # [T, L, H, C]
                avg_entropy = compute_average_entropy(attention.numpy())

                entropy_sum += avg_entropy
                window_entropies.append(avg_entropy)
                window_metadata.append({"system_idx": system_idx, "system_name": system_name, "window_idx": window_idx})

                processed_windows += 1
                progress.update(1)

                # Clean up tensors promptly to keep GPU memory usage low.
                del preds, attention, avg_entropy
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                if max_windows_total is not None and processed_windows >= max_windows_total:
                    break

        progress.close()

    if hasattr(pipeline, "remove_all_hooks"):
        pipeline.remove_all_hooks()

    if processed_windows == 0:
        logger.warning("No windows processed for model %s.", model_name)
        return

    avg_entropy = entropy_sum / processed_windows
    stacked_entropies = np.stack(window_entropies, axis=0)

    save_dir.mkdir(parents=True, exist_ok=True)
    np.save(save_dir / "avg_entropy.npy", avg_entropy)
    np.save(save_dir / "window_entropies.npy", stacked_entropies)
    with (save_dir / "metadata.json").open("w") as fp:
        json.dump(
            {
                "num_windows": processed_windows,
                "num_layers": num_layers,
                "num_heads": num_heads,
                "window_metadata": window_metadata,
                "config": OmegaConf.to_container(cfg, resolve=True),
            },
            fp,
        )

    plot_entropy_heatmap(avg_entropy, save_dir / "avg_entropy_heatmap.png")
    logger.info("Saved entropy results for %s to %s", model_name, save_dir)


@hydra.main(config_path="../config", config_name="attention_entropy", version_base=None)
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO)
    logger.info("Configuration\n%s", OmegaConf.to_yaml(cfg))

    original_cwd = Path(get_original_cwd())
    plotting_config = original_cwd / "config" / "plotting.yaml"
    apply_custom_style(str(plotting_config))

    set_seed(cfg.attention_entropy.random_seed)

    device = torch.device(cfg.eval.device if torch.cuda.is_available() else "cpu")
    torch_dtype = getattr(torch, cfg.eval.torch_dtype)
    assert isinstance(torch_dtype, torch.dtype)

    datasets = build_datasets(cfg)

    save_root = Path(cfg.attention_entropy.output_root)
    save_root.mkdir(parents=True, exist_ok=True)

    for model_name in cfg.models_to_run:
        model_dir = save_root / model_name
        process_model(model_name, cfg, datasets, model_dir, device, torch_dtype)


if __name__ == "__main__":
    main()
