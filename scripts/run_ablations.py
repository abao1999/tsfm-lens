"""Combined ablations script — drives `evaluate_ablations` for all 6 TSFMs."""

import json
import logging
import os

import hydra
import torch
from omegaconf import DictConfig

from tsfm_lens.circuitlens import BaseCircuitLens
from tsfm_lens.dataset import TestDataset
from tsfm_lens.evaluation import evaluate_ablations
from tsfm_lens.utils import (
    clear_cuda_cache,
    get_dim_from_dataset,
    get_eval_data_dict,
    get_gift_eval_data_dict,
    left_pad_and_stack_1D,
    make_json_serializable,
    save_evaluation_results,
    set_seed,
)

logger = logging.getLogger(__name__)

# Moirai 1.1 was trained at this context length; used in both loader and config builder.
MOIRAI_TRAINED_CONTEXT_LENGTH = 4000


# =============================================================================
# Per-model loaders (lazy-imported so unused frameworks aren't pulled in)
# =============================================================================
def _load_chronos(cfg: DictConfig, device: torch.device, dtype: torch.dtype) -> BaseCircuitLens:
    from tsfm_lens.chronos.circuitlens import CircuitLensChronos
    return CircuitLensChronos.from_pretrained(cfg.chronos.model_id, device_map=cfg.eval.device, torch_dtype=dtype)


def _load_chronos_bolt(cfg: DictConfig, device: torch.device, dtype: torch.dtype) -> BaseCircuitLens:
    from tsfm_lens.chronos_bolt.circuitlens import CircuitLensBolt
    return CircuitLensBolt.from_pretrained(cfg.chronos_bolt.model_id, device_map=cfg.eval.device, torch_dtype=dtype)


def _load_chronos2(cfg: DictConfig, device: torch.device, dtype: torch.dtype) -> BaseCircuitLens:
    from tsfm_lens.chronos2.circuitlens import CircuitLensChronos2
    return CircuitLensChronos2(cfg.chronos2.model_id, device=device)


def _load_timesfm(cfg: DictConfig, device: torch.device, dtype: torch.dtype) -> BaseCircuitLens:
    from tsfm_lens.timesfm.circuitlens import CircuitLensTimesFM
    return CircuitLensTimesFM(cfg.timesfm.model_id, device_map=cfg.eval.device)


def _load_toto(cfg: DictConfig, device: torch.device, dtype: torch.dtype) -> BaseCircuitLens:
    from tsfm_lens.toto.circuitlens import CircuitLensToto
    return CircuitLensToto(cfg.toto.model_id, device_map=cfg.eval.device)


def _load_moirai(cfg: DictConfig, device: torch.device, dtype: torch.dtype) -> BaseCircuitLens:
    from tsfm_lens.moirai.circuitlens import CircuitLensMoirai
    return CircuitLensMoirai(
        cfg.moirai.model_id,
        context_length=MOIRAI_TRAINED_CONTEXT_LENGTH,
        prediction_length=1,  # placeholder; replaced inside .predict()
        patch_size=cfg.moirai.patch_size,
        num_samples=cfg.moirai.num_samples,
        target_dim=1,
        device=cfg.eval.device,
    )


# =============================================================================
# Chronos-2 predict wrapper (must run before any forecasting)
# =============================================================================
def _install_chronos2_predict_wrapper(pipeline: BaseCircuitLens) -> None:
    """Coerce heterogeneous tensor contexts into one 2D tensor and unwrap the point forecast.

    Chronos-2 returns ``(point_forecast, quantiles)``; the ablations evaluator expects a
    single forecast tensor, so we strip the second element here.
    """
    inner_predict = pipeline.predict

    def normalize(context):
        if isinstance(context, torch.Tensor) or not isinstance(context, (list, tuple)):
            return context
        if not context:
            raise ValueError("Chronos-2 received empty context input.")
        flat = []
        for ctx in context:
            if not isinstance(ctx, torch.Tensor):
                raise ValueError(f"Chronos-2 expects tensor context entries, got {type(ctx)}")
            if ctx.ndim == 2:
                flat.extend(ctx.unbind(0))
            elif ctx.ndim == 1:
                flat.append(ctx)
            else:
                raise ValueError(f"Unexpected context shape for Chronos-2: {ctx.shape}")
        if all(t.shape == flat[0].shape for t in flat):
            return torch.stack(flat, dim=0)
        return left_pad_and_stack_1D(flat)

    def predict(*args, **kwargs):
        if args:
            args = (normalize(args[0]),) + args[1:]
        elif "context" in kwargs:
            kwargs["context"] = normalize(kwargs["context"])
        point_forecast, _ = inner_predict(*args, **kwargs)
        return point_forecast

    pipeline.predict = predict  # type: ignore[method-assign]


# =============================================================================
# Per-model run config (context_length, prediction_kwargs, num_samples)
# =============================================================================
def _chronos_config(pipeline: BaseCircuitLens, cfg: DictConfig) -> dict:
    num_samples = 1 if cfg.chronos.deterministic else cfg.chronos.num_samples
    prediction_kwargs = {
        "limit_prediction_length": cfg.chronos.limit_prediction_length,
        "top_k": cfg.chronos.top_k,
        "top_p": cfg.chronos.top_p,
        "temperature": cfg.chronos.temperature,
        "num_samples": num_samples,
    }
    if cfg.chronos.deterministic and cfg.chronos.num_samples > 1:
        logger.warning("num_samples > 1 but deterministic=True. Forcing num_samples=1, do_sample=False.")
        prediction_kwargs["do_sample"] = False
    mc = pipeline.model.config
    logger.info(f"chronos trained with context_length={mc.context_length}, prediction_length={mc.prediction_length}")
    return {"context_length": cfg.chronos.context_length, "prediction_kwargs": prediction_kwargs, "num_samples": num_samples}


def _chronos_bolt_config(pipeline: BaseCircuitLens, cfg: DictConfig) -> dict:
    bc = pipeline.model.config.chronos_config
    logger.info(f"chronos_bolt trained with context_length={bc['context_length']}, prediction_length={bc['prediction_length']}")
    return {
        "context_length": cfg.chronos_bolt.context_length,
        "prediction_kwargs": {"limit_prediction_length": cfg.chronos_bolt.limit_prediction_length},
        "num_samples": len(pipeline.quantiles),
    }


def _chronos2_config(pipeline: BaseCircuitLens, cfg: DictConfig) -> dict:
    _install_chronos2_predict_wrapper(pipeline)
    logger.info(f"chronos2 context_length={cfg.chronos2.context_length}, prediction_length=dynamic")
    return {
        "context_length": cfg.chronos2.context_length,
        "prediction_kwargs": {"batch_size": cfg.eval.batch_size, "limit_prediction_length": False},
        "num_samples": 1,
    }


def _timesfm_config(pipeline: BaseCircuitLens, cfg: DictConfig) -> dict:
    # predict() returns the median forecast; use num_samples=len(quantiles) only for predict_point_and_quantiles().
    return {"context_length": cfg.timesfm.context_length, "prediction_kwargs": {}, "num_samples": 1}


def _toto_config(pipeline: BaseCircuitLens, cfg: DictConfig) -> dict:
    prediction_kwargs = {
        "num_samples": cfg.toto.num_samples,
        "samples_per_batch": cfg.toto.samples_per_batch,
        "use_kv_cache": cfg.toto.use_kv_cache,
    }
    return {"context_length": cfg.toto.context_length, "prediction_kwargs": prediction_kwargs, "num_samples": cfg.toto.num_samples}


def _moirai_config(pipeline: BaseCircuitLens, cfg: DictConfig) -> dict:
    return {
        "context_length": MOIRAI_TRAINED_CONTEXT_LENGTH,
        "prediction_kwargs": {"num_samples": cfg.moirai.num_samples},
        "num_samples": cfg.moirai.num_samples,
    }


MODELS = {
    "chronos":      (_load_chronos,      _chronos_config),
    "chronos_bolt": (_load_chronos_bolt, _chronos_bolt_config),
    "chronos2":     (_load_chronos2,     _chronos2_config),
    "timesfm":      (_load_timesfm,      _timesfm_config),
    "toto":         (_load_toto,         _toto_config),
    "moirai":       (_load_moirai,       _moirai_config),
}


# =============================================================================
# Dataset loading
# =============================================================================
def _load_test_datasets(
    cfg: DictConfig, context_length: int
) -> tuple[dict[str, TestDataset], dict[str, int], dict[str, int]]:
    """Build `{system_name: TestDataset}` plus per-system dim and sample-count metadata."""
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
            data_dir=cfg.eval.data_dir, dataset_names=names, term=gift.term, to_univariate=gift.to_univariate
        )
    else:
        test_data_dict = get_eval_data_dict(
            cfg.eval.data_dir,
            num_subdirs=cfg.eval.dysts.num_subdirs,
            num_samples_per_subdir=cfg.eval.dysts.num_samples_per_subdir,
            subdir_names=cfg.eval.dysts.system_names,
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


# =============================================================================
# Metrics JSON saving
# =============================================================================
def _save_metrics_json(
    ablations_subdir: str,
    rseed: int,
    metrics_fname: str,
    ablate_n_heads_per_layer: int | None,
    metrics: dict,
    ablation_vs_original: dict,
    ablation_vs_labels: dict,
) -> None:
    """Write the three result dicts as JSON under `outputs/<subdir>/rseed-<rseed>/...`."""
    base = f"outputs/{ablations_subdir}/rseed-{rseed}"
    nheads_str = f"nheads-{ablate_n_heads_per_layer}" if ablate_n_heads_per_layer is not None else "all_heads"
    targets = [
        ("original_vs_labels",    f"{metrics_fname}.json",                            metrics),
        ("ablations_vs_original", f"{metrics_fname}_ablations_against_original.json", ablation_vs_original),
        ("ablations_vs_labels",   f"{metrics_fname}_ablations_against_labels.json",   ablation_vs_labels),
    ]
    for subdir, fname, data in targets:
        path = os.path.join(base, subdir, nheads_str, fname)
        if not data:
            logger.warning(f"No data to save for {path}")
            continue
        os.makedirs(os.path.dirname(path), exist_ok=True)
        logger.info(f"Saving metrics to {path}")
        with open(path, "w") as f:
            json.dump(make_json_serializable(data), f, indent=4)


# =============================================================================
# Main
# =============================================================================
@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg):
    device = torch.device(cfg.eval.device if torch.cuda.is_available() else "cpu")
    torch_dtype = getattr(torch, cfg.eval.torch_dtype)
    logger.info(f"device={device}, torch_dtype={torch_dtype}")
    clear_cuda_cache(device)

    model_type = cfg.ablation.model_type
    if model_type not in MODELS:
        raise ValueError(f"Unknown model_type: {model_type!r}. Expected one of {list(MODELS)}")
    loader, build_config = MODELS[model_type]

    ablations_subdir = f"ablations_{model_type}_{cfg.ablation.model_name_str}"
    ablations_save_dir = None
    if cfg.eval.results_save_dir is not None:
        ablations_save_dir = os.path.join(cfg.eval.results_save_dir, ablations_subdir)
        os.makedirs(ablations_save_dir, exist_ok=True)
        logger.info(f"ablations_save_dir={ablations_save_dir}")

    pipeline = loader(cfg, device, torch_dtype)
    model_cfg = build_config(pipeline, cfg)
    context_length, prediction_kwargs, num_samples = (
        model_cfg["context_length"], model_cfg["prediction_kwargs"], model_cfg["num_samples"]
    )
    logger.info(
        f"pipeline={pipeline} | layers={pipeline.num_layers}, heads={pipeline.num_heads} | "
        f"context_length={context_length}, eval.prediction_length={cfg.eval.prediction_length} | "
        f"prediction_kwargs={prediction_kwargs}"
    )

    test_datasets, system_dims, n_system_samples = _load_test_datasets(cfg, context_length)

    logger.info(f"ablations_layers_lst={cfg.ablation.ablations_layers_lst}, types={cfg.ablation.ablations_types}")
    set_seed(cfg.eval.rseed)
    metrics, ablation_vs_original, ablation_vs_labels = evaluate_ablations(
        pipeline,
        test_datasets,
        batch_size=cfg.eval.batch_size,
        num_samples=num_samples,
        prediction_length=cfg.eval.prediction_length,
        system_dims=system_dims,
        list_of_layers_to_ablate=cfg.ablation.ablations_layers_lst,
        ablate_n_heads_per_layer=cfg.ablation.ablate_n_heads_per_layer,
        ablations_types=cfg.ablation.ablations_types,
        metric_names=cfg.eval.metric_names,
        parallel_sample_reduction=cfg.eval.parallel_sample_reduction,
        prediction_kwargs=prediction_kwargs,
        eval_subintervals=[(0, i + 64) for i in range(0, cfg.eval.prediction_length, 64)],
        rseed=cfg.eval.rseed,
        save_dir=ablations_save_dir,
        save_predictions_from_ablations=False,
        compute_metrics_ablations_against_labels=True,
    )

    logger.info(f"Saving metrics CSV to {cfg.eval.metrics_save_dir}")
    save_evaluation_results(
        metrics,
        metrics_metadata={"system_dims": system_dims, "n_system_samples": n_system_samples},
        metrics_save_dir=cfg.eval.metrics_save_dir,
        metrics_fname=cfg.eval.metrics_fname,
        overwrite=cfg.eval.overwrite,
    )

    _save_metrics_json(
        ablations_subdir=ablations_subdir,
        rseed=cfg.eval.rseed,
        metrics_fname=cfg.eval.metrics_fname,
        ablate_n_heads_per_layer=cfg.ablation.ablate_n_heads_per_layer,
        metrics=metrics,
        ablation_vs_original=ablation_vs_original,
        ablation_vs_labels=ablation_vs_labels,
    )


if __name__ == "__main__":
    main()
