"""
Combined ablations script
"""

import json
import logging
import os
from functools import partial

import hydra
import torch

from tsfm_lens.chronos.circuitlens import CircuitLensChronos
from tsfm_lens.chronos_bolt.circuitlens import CircuitLensBolt
from tsfm_lens.dataset import TestDataset
from tsfm_lens.evaluation import evaluate_ablations
from tsfm_lens.timesfm.circuitlens import CircuitLensTimesFM
from tsfm_lens.toto.circuitlens import CircuitLensToto
from tsfm_lens.utils import (
    clear_cuda_cache,
    get_dim_from_dataset,
    get_eval_data_dict,
    get_gift_eval_data_dict,
    make_json_serializable,
    save_evaluation_results,
)
from tsfm_lens.utils.eval_utils import set_seed

logger = logging.getLogger(__name__)


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg):
    ##### 1. Reproducibility and setup #####
    device = torch.device(cfg.eval.device if torch.cuda.is_available() else "cpu")
    torch_dtype = getattr(torch, cfg.eval.torch_dtype)
    assert isinstance(torch_dtype, torch.dtype)
    logger.info(f"Using device: {device}, with torch_dtype: {torch_dtype}")

    clear_cuda_cache(device)

    # Determine which model type to use
    model_type = cfg.ablation.model_type
    logger.info(f"Model type: {model_type}")

    # Ablations save directory for results
    ablations_subdir = f"ablations_{model_type}_{cfg.ablation.model_name_str}"
    ablations_save_dir = None
    if cfg.eval.results_save_dir is not None:
        ablations_save_dir = os.path.join(cfg.eval.results_save_dir, ablations_subdir)
        os.makedirs(ablations_save_dir, exist_ok=True)
        logger.info(f"Saving ablations results to {ablations_save_dir}")

    ##### 2. Load CircuitLens pipeline (model-specific) #####
    if model_type == "chronos":
        pipeline = CircuitLensChronos.from_pretrained(
            cfg.chronos.model_id,
            device_map=cfg.eval.device,
            torch_dtype=torch_dtype,
        )

        model_config = dict(vars(pipeline.model.config))
        context_length = cfg.chronos.context_length

        prediction_kwargs = {
            "limit_prediction_length": cfg.chronos.limit_prediction_length,
            "top_k": cfg.chronos.top_k,
            "top_p": cfg.chronos.top_p,
            "temperature": cfg.chronos.temperature,
            "num_samples": 1 if cfg.chronos.deterministic else cfg.chronos.num_samples,
        }
        num_samples = prediction_kwargs["num_samples"]
        if cfg.chronos.deterministic and cfg.chronos.num_samples > 1:
            logger.warning("num_samples is greater than 1, but deterministic is True. Setting num_samples to 1.")
            prediction_kwargs["do_sample"] = False
            prediction_kwargs["num_samples"] = 1

        logger.info(f"model was trained with context_length: {model_config['context_length']}")
        logger.info(f"model was trained with prediction_length: {model_config['prediction_length']}")

    elif model_type == "chronos_bolt":
        pipeline = CircuitLensBolt.from_pretrained(
            cfg.chronos_bolt.model_id,
            device_map=cfg.eval.device,
            torch_dtype=torch_dtype,
        )

        quantiles = pipeline.quantiles
        num_samples = len(quantiles)

        chronos_model_config = dict(pipeline.model.config.chronos_config)
        context_length = cfg.chronos_bolt.context_length

        logger.info(f"model was trained with context_length: {chronos_model_config['context_length']}")
        logger.info(f"model was trained with prediction_length: {chronos_model_config['prediction_length']}")

        prediction_kwargs = {
            "limit_prediction_length": cfg.chronos_bolt.limit_prediction_length,
        }
    elif model_type == "timesfm":
        pipeline = CircuitLensTimesFM(
            cfg.timesfm.model_id,
            device_map=cfg.eval.device,
        )

        quantiles = pipeline.quantiles
        # NOTE: only use num_samples = len(quantiles) for predict_point_and_quantiles
        # Otherwise, predict() method returns just the median forecast
        # num_samples = len(quantiles)
        num_samples = 1

        context_length = cfg.timesfm.context_length
        prediction_kwargs = {}

    elif model_type == "toto":
        pipeline = CircuitLensToto(
            cfg.toto.model_id,
            device_map=cfg.eval.device,
        )

        context_length = cfg.toto.context_length
        prediction_kwargs = {
            "num_samples": cfg.toto.num_samples,
            "samples_per_batch": cfg.toto.samples_per_batch,
            "use_kv_cache": cfg.toto.use_kv_cache,
        }
        num_samples = prediction_kwargs["num_samples"]

    else:
        raise ValueError(f"Unknown model type: {model_type}. Expected 'chronos' or 'chronos_bolt'")

    num_layers = pipeline.num_layers
    num_heads = pipeline.num_heads

    logger.info(f"pipeline: {pipeline}")
    logger.info(f"num_layers: {num_layers}, num_heads: {num_heads}")
    logger.info(pipeline.__class__.__name__)
    pipeline.set_to_eval_mode()

    logger.info(f"context_length: {context_length}")
    logger.info(f"eval prediction_length: {cfg.eval.prediction_length}")
    logger.info(f"prediction_kwargs: {prediction_kwargs}")

    ##### 3. Load Test Datasets #####
    dataset_name = cfg.eval.dataset_name
    if dataset_name == "gift-eval":
        gift_eval_kwargs = dict(cfg.eval.gift_eval)
        selected_dataset_names = gift_eval_kwargs["dataset_names"]
        selected_dataset_names = (
            [selected_dataset_names] if isinstance(selected_dataset_names, str) else selected_dataset_names
        )
        selected_term = gift_eval_kwargs["term"]
        if selected_dataset_names is None:
            selected_dataset_names = (
                gift_eval_kwargs["short_datasets"]
                if selected_term == "short"
                else gift_eval_kwargs["medium_long_datasets"]
            )
        logger.info(f"selected_dataset_names: {selected_dataset_names}")
        logger.info(f"selected_term: {selected_term}")
        max_num_datasets = cfg.eval.gift_eval.max_num_datasets
        if max_num_datasets is not None:
            selected_dataset_names = selected_dataset_names[:max_num_datasets]
            logger.info(f"Selected first {max_num_datasets} datasets: {selected_dataset_names}")
        test_data_dict = get_gift_eval_data_dict(
            data_dir=cfg.eval.data_dir,
            dataset_names=selected_dataset_names,
            term=selected_term,
            to_univariate=gift_eval_kwargs["to_univariate"],
        )
    else:
        test_data_dict = get_eval_data_dict(
            cfg.eval.data_dir,
            num_subdirs=cfg.eval.dysts.num_subdirs,
            num_samples_per_subdir=cfg.eval.dysts.num_samples_per_subdir,
            subdir_names=cfg.eval.dysts.system_names,
        )

    logger.info(f"Number of test datasets: {len(test_data_dict)}")

    system_dims = {}
    n_system_samples = {}

    for system_name, datasets in test_data_dict.items():
        if not datasets:
            raise ValueError(f"No datasets found for system {system_name}")
        sample_dataset = datasets[0]
        if dataset_name == "gift-eval":
            system_dims[system_name] = sample_dataset.target_dim if not cfg.eval.gift_eval.to_univariate else 1
            n_system_samples[system_name] = len(sample_dataset.hf_dataset)
        else:
            system_dims[system_name] = get_dim_from_dataset(sample_dataset)
            n_system_samples[system_name] = len(datasets)

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

    ##### 4. Evaluation and Saving #####
    save_eval_results_fn = partial(
        save_evaluation_results,
        metrics_metadata={
            "system_dims": system_dims,
            "n_system_samples": n_system_samples,
        },  # pass metadata to be saved as columns in metrics csv
        metrics_save_dir=cfg.eval.metrics_save_dir,
        overwrite=cfg.eval.overwrite,
    )

    list_of_layers_to_ablate = cfg.ablation.ablations_layers_lst
    ablations_types = cfg.ablation.ablations_types

    logger.info(f"list_of_layers_to_ablate: {list_of_layers_to_ablate}")
    logger.info(f"ablations_types: {ablations_types}")

    set_seed(cfg.eval.rseed)
    metrics, metrics_from_ablations, metrics_from_ablations_against_labels = evaluate_ablations(
        pipeline,
        test_datasets,
        batch_size=cfg.eval.batch_size,
        num_samples=num_samples,
        prediction_length=cfg.eval.prediction_length,
        system_dims=system_dims,
        list_of_layers_to_ablate=list_of_layers_to_ablate,
        ablate_n_heads_per_layer=cfg.ablation.ablate_n_heads_per_layer,
        ablations_types=ablations_types,
        metric_names=cfg.eval.metric_names,
        parallel_sample_reduction=cfg.eval.parallel_sample_reduction,
        prediction_kwargs=prediction_kwargs,
        eval_subintervals=[(0, i + 64) for i in range(0, cfg.eval.prediction_length, 64)],
        rseed=cfg.eval.rseed,
        save_dir=ablations_save_dir,
        save_predictions_from_ablations=False,
        compute_metrics_ablations_against_labels=True,
    )

    logger.info(f"Saving metrics as csv to {cfg.eval.metrics_save_dir}")
    save_eval_results_fn(metrics, metrics_fname=cfg.eval.metrics_fname)

    # metrics_save_dir for JSON outputs
    metrics_save_dir = f"outputs/{ablations_subdir}/rseed-{cfg.eval.rseed}"
    logger.info(f"Saving all metrics as json to {metrics_save_dir}")
    os.makedirs(metrics_save_dir, exist_ok=True)

    nheads_per_layer_str = (
        f"nheads-{cfg.ablation.ablate_n_heads_per_layer}"
        if cfg.ablation.ablate_n_heads_per_layer is not None
        else "all_heads"
    )
    metrics_files = {
        os.path.join(
            metrics_save_dir, "original_vs_labels", nheads_per_layer_str, f"{cfg.eval.metrics_fname}.json"
        ): metrics,
        os.path.join(
            metrics_save_dir,
            "ablations_vs_original",
            nheads_per_layer_str,
            f"{cfg.eval.metrics_fname}_ablations_against_original.json",
        ): metrics_from_ablations,
        os.path.join(
            metrics_save_dir,
            "ablations_vs_labels",
            nheads_per_layer_str,
            f"{cfg.eval.metrics_fname}_ablations_against_labels.json",
        ): metrics_from_ablations_against_labels,
    }

    for fname, data in metrics_files.items():
        if not data:
            logger.warning(f"No data to save for {fname}")
            continue
        print(f"Saving metrics to {fname}")
        os.makedirs(os.path.dirname(fname), exist_ok=True)

        data = make_json_serializable(data)
        with open(fname, "w") as f:
            json.dump(data, f, indent=4)


if __name__ == "__main__":
    main()
