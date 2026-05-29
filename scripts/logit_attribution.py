"""
Script to evaluate Chronos models on the evaluation dataset
"""

import logging
import os
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

from tsfm_lens.attribution import evaluate_attribution_with_full_outputs
from tsfm_lens.chronos.circuitlens import CircuitLensChronos
from tsfm_lens.dataset import TestDataset
from tsfm_lens.utils import (
    get_dim_from_dataset,
    get_eval_data_dict,
    get_gift_eval_data_dict,
    save_evaluation_results,
)

logger = logging.getLogger(__name__)


def _load_test_datasets(
    cfg: DictConfig, context_length: int
) -> tuple[dict[str, TestDataset], dict[str, int], dict[str, int]]:
    """Build `{system_name: TestDataset}` plus per-system dim and sample-count metadata.

    Supports both dysts (default) and gift-eval, dispatching on `cfg.eval.dataset_name`.
    For dysts: explicit `system_names` wins; otherwise selects subdirs via
    `[system_start_idx:system_end_idx]` on the alphabetically-sorted subdir list.
    """
    dataset_name = cfg.eval.dataset_name
    if dataset_name == "gift-eval":
        gift = cfg.eval.gift_eval
        names = gift.dataset_names
        if names is None:
            names = gift.short_datasets if gift.term == "short" else gift.medium_long_datasets
        elif isinstance(names, str):
            names = [names]
        if gift.max_num_datasets is not None:
            names = names[: gift.max_num_datasets]
        logger.info(f"gift-eval datasets={names}, term={gift.term}")
        test_data_dict = get_gift_eval_data_dict(
            data_dir=cfg.eval.data_dir,
            dataset_names=names,
            term=gift.term,
            to_univariate=gift.to_univariate,
        )
    else:
        if cfg.eval.dysts.system_names:
            subdir_names = list(cfg.eval.dysts.system_names)
            logger.info(f"Using explicit system_names: {subdir_names}")
        else:
            data_dirs = cfg.eval.data_dir
            if isinstance(data_dirs, str):
                data_dirs = [data_dirs]
            all_names = sorted({d.name for dd in data_dirs for d in Path(dd).iterdir() if d.is_dir()})
            start = cfg.eval.dysts.system_start_idx
            end = cfg.eval.dysts.system_end_idx
            subdir_names = all_names[start:end]
            logger.info(f"Selected systems [{start}:{end}] of {len(all_names)} = {subdir_names}")
        test_data_dict = get_eval_data_dict(
            cfg.eval.data_dir,
            num_samples_per_subdir=cfg.eval.dysts.num_samples_per_subdir,
            subdir_names=subdir_names,
        )

    logger.info(f"Loaded {len(test_data_dict)} test datasets: {list(test_data_dict.keys())}")

    system_dims, n_system_samples = {}, {}
    for system_name, datasets in test_data_dict.items():
        if not datasets:
            raise ValueError(f"No datasets found for system {system_name}")
        sample = datasets[0]
        if dataset_name == "gift-eval":
            system_dims[system_name] = 1 if cfg.eval.gift_eval.to_univariate else sample.target_dim
            n_system_samples[system_name] = len(sample.hf_dataset)
        else:
            system_dims[system_name] = get_dim_from_dataset(sample)
            n_system_samples[system_name] = len(datasets)

    test_datasets = {
        name: TestDataset(
            datasets=test_data_dict[name],
            context_length=context_length,  # not used for GiftEvalDataset (pre-defined splits)
            prediction_length=cfg.eval.prediction_length,
            num_test_instances=cfg.eval.num_test_instances,
            window_style=cfg.eval.window_style,
            window_stride=cfg.eval.window_stride,
            window_start_time=cfg.eval.window_start_time,
            random_seed=cfg.eval.rseed,
        )
        for name in test_data_dict
    }
    return test_datasets, system_dims, n_system_samples


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg):
    ##### 1. Reproducibility and setup #####
    device = torch.device(cfg.eval.device if torch.cuda.is_available() else "cpu")
    torch_dtype = getattr(torch, cfg.eval.torch_dtype)
    assert isinstance(torch_dtype, torch.dtype)
    logger.info(f"Using device: {device}, with torch_dtype: {torch_dtype}")

    dla_dir = os.path.join(cfg.eval.results_save_dir, "logit_attribution")
    os.makedirs(dla_dir, exist_ok=True)
    logger.info(f"Saving logit attribution results to {dla_dir}")

    ##### 2. Load CircuitsLens, a wrapper around model pipeline #####
    pipeline = CircuitLensChronos.from_pretrained(
        cfg.chronos.model_id,
        device_map=cfg.eval.device,
        torch_dtype=torch_dtype,
    )
    logger.info(f"pipeline: {pipeline}")
    num_layers = pipeline.model.model.config.num_decoder_layers
    num_heads = pipeline.model.model.config.num_heads
    logger.info(f"num_layers: {num_layers}, num_heads: {num_heads}")
    logger.info(pipeline.__class__.__name__)
    pipeline.model.eval()

    model_config = dict(vars(pipeline.model.config))
    context_length = model_config["context_length"]
    prediction_length = model_config["prediction_length"]
    logger.info(f"context_length: {context_length}")
    logger.info(f"model prediction_length: {prediction_length}")
    logger.info(f"eval prediction_length: {cfg.eval.prediction_length}")

    prediction_kwargs = {
        "limit_prediction_length": cfg.chronos.limit_prediction_length,
        "do_sample": not cfg.chronos.deterministic,
        "top_k": cfg.chronos.top_k,
        "top_p": cfg.chronos.top_p,
        "temperature": cfg.chronos.temperature,
        "num_samples": 1 if cfg.chronos.deterministic else cfg.chronos.num_samples,
    }
    # redundancy for extra safety
    if cfg.chronos.deterministic and cfg.chronos.num_samples > 1:
        logger.warning("num_samples is greater than 1, but deterministic is True. Setting num_samples to 1.")
        prediction_kwargs["do_sample"] = False
        prediction_kwargs["num_samples"] = 1
    logger.info(f"prediction_kwargs: {prediction_kwargs}")

    ##### 3. Add hooks for logit attribution #####
    pipeline.remove_all_hooks()
    head_indices = [(i, j) for i in range(num_layers) for j in range(num_heads)]
    for attention_type in ("sa", "ca"):
        logger.info(f"Adding {attention_type.upper()} head attribution hooks")
        pipeline.add_head_attribution_hooks(head_indices, attention_type=attention_type)

    logger.info("Adding MLP attribution hooks")
    pipeline.add_mlp_attribution_hooks(list(range(num_layers)))

    logger.info("Adding Read Stream hooks")
    pipeline.add_read_stream_hooks(list(range(num_layers)))

    ##### 4. Load Datasets #####
    test_datasets, system_dims, n_system_samples = _load_test_datasets(cfg, context_length)
    logger.info(f"Running evaluation on {list(test_datasets.keys())}")

    ##### 5. Evaluation (per-system JSON metrics are saved inside the function) #####
    metrics, _, _ = evaluate_attribution_with_full_outputs(
        pipeline,
        test_datasets,
        batch_size=cfg.eval.batch_size,
        prediction_length=cfg.eval.prediction_length,
        metric_names=cfg.eval.metric_names,
        system_dims=system_dims,
        parallel_sample_reduction=cfg.eval.parallel_sample_reduction,
        prediction_kwargs=prediction_kwargs,
        eval_subintervals=[(0, i + 64) for i in range(0, cfg.eval.prediction_length, 64)],
        rseed=cfg.eval.rseed,
        attribution_types={"read_stream"},
        save_dir=dla_dir,
        metrics_fname=cfg.eval.metrics_fname,
        save_predictions_from_attribution=True,
    )

    logger.info(f"Saving metrics as csv to {cfg.eval.metrics_save_dir}")
    save_evaluation_results(
        metrics,
        metrics_metadata={
            "system_dims": system_dims,
            "n_system_samples": n_system_samples,
        },
        metrics_save_dir=cfg.eval.metrics_save_dir,
        overwrite=cfg.eval.overwrite,
        metrics_fname=cfg.eval.metrics_fname,
    )


if __name__ == "__main__":
    main()
