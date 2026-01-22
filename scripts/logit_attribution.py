"""
Script to evaluate Chronos models on the evaluation dataset
"""

import json
import logging
import os
from functools import partial

import hydra
import torch

from tsfm_lens.attribution import evaluate_attribution_with_full_outputs
from tsfm_lens.chronos.circuitlens import CircuitLensChronos
from tsfm_lens.dataset import TestDataset
from tsfm_lens.utils import (
    get_dim_from_dataset,
    get_eval_data_dict,
    make_json_serializable,
    save_evaluation_results,
)

logger = logging.getLogger(__name__)


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg):
    ##### 1. Reproducibility and setup #####
    device = torch.device(cfg.eval.device if torch.cuda.is_available() else "cpu")
    torch_dtype = getattr(torch, cfg.eval.torch_dtype)
    assert isinstance(torch_dtype, torch.dtype)
    logger.info(f"Using device: {device}, with torch_dtype: {torch_dtype}")

    # DLA save directory for results
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

    # settings for  CircuitLensChronos to generate predictions
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
    # Add hooks for head attribution
    logger.info("Adding SA head attribution hooks")
    pipeline.add_head_attribution_hooks(
        [(i, j) for i in range(num_layers) for j in range(num_heads)],
        attention_type="sa",
    )
    logger.info("Adding CA head attribution hooks")
    pipeline.add_head_attribution_hooks(
        [(i, j) for i in range(num_layers) for j in range(num_heads)],
        attention_type="ca",
    )
    logger.info("Adding MLP attribution hooks")
    pipeline.add_mlp_attribution_hooks([i for i in range(num_layers)])

    logger.info("Adding Read Stream hooks")
    pipeline.add_read_stream_hooks([i for i in range(num_layers)])

    # logger.info("Adding SA attn output attribution hooks")
    # pipeline.add_attn_attribution_hooks(
    #     [i for i in range(num_layers)], attention_type="sa"
    # )
    # logger.info("Adding CA attn output attribution hooks")
    # pipeline.add_attn_attribution_hooks(
    #     [i for i in range(num_layers)], attention_type="ca"
    # )

    ##### 4. Load Datasets #####
    test_data_dict = get_eval_data_dict(
        cfg.eval.data_dir,
        num_subdirs=cfg.eval.dysts.num_subdirs,
        num_samples_per_subdir=cfg.eval.dysts.num_samples_per_subdir,
        subdir_names=cfg.eval.dysts.system_names,
    )
    logger.info(f"Number of combined evaluation data subdirectories: {len(test_data_dict)}")
    system_dims = {system_name: get_dim_from_dataset(test_data_dict[system_name][0]) for system_name in test_data_dict}
    n_system_samples = {system_name: len(test_data_dict[system_name]) for system_name in test_data_dict}

    logger.info(f"Running evaluation on {list(test_data_dict.keys())}")

    test_datasets = {
        system_name: TestDataset(
            datasets=test_data_dict[system_name],
            context_length=context_length,  # NOTE: not used when dataset is GiftEvalDataset, since gift-eval has pre-defined splits
            prediction_length=cfg.eval.prediction_length,
            num_test_instances=cfg.eval.num_test_instances,
            window_style=cfg.eval.window_style,
            window_stride=cfg.eval.window_stride,
            window_start_time=cfg.eval.window_start_time,
            random_seed=cfg.eval.rseed,
        )
        for system_name in test_data_dict
    }

    ##### 5. Evaluation and Saving #####
    save_eval_results_fn = partial(
        save_evaluation_results,
        metrics_metadata={
            "system_dims": system_dims,
            "n_system_samples": n_system_samples,
        },  # pass metadata to be saved as columns in metrics csv
        metrics_save_dir=cfg.eval.metrics_save_dir,
        overwrite=cfg.eval.overwrite,
    )

    metrics, metrics_from_attribution, metrics_from_logits = evaluate_attribution_with_full_outputs(
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
        save_predictions_from_attribution=False,
    )

    logger.info(f"Saving metrics as csv to {cfg.eval.metrics_save_dir}")
    save_eval_results_fn(metrics, metrics_fname=cfg.eval.metrics_fname)

    # metrics_save_dir = cfg.eval.results_save_dir
    metrics_save_dir = f"outputs/logit_attribution/rseed-{cfg.eval.rseed}"
    logger.info(f"Saving all metrics as json to {metrics_save_dir}")
    os.makedirs(metrics_save_dir, exist_ok=True)
    metrics_files = {
        f"{cfg.eval.metrics_fname}.json": metrics,
        f"{cfg.eval.metrics_fname}_attribution.json": metrics_from_attribution,
        f"{cfg.eval.metrics_fname}_logits.json": metrics_from_logits,
    }

    for fname, data in metrics_files.items():
        data = make_json_serializable(data)
        with open(os.path.join(metrics_save_dir, fname), "w") as f:
            json.dump(data, f, indent=4)


if __name__ == "__main__":
    main()
