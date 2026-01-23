"""
Run TSFMs with optional ablations on the GIFT-Eval benchmark.
Configuration is managed via Hydra config files.
"""

import csv
import json
import logging
import os
from itertools import groupby
from typing import Literal

import hydra
import numpy as np
import pandas as pd
import torch
from chronos import ForecastType
from gluonts.ev.metrics import (
    MAE,
    MAPE,
    MASE,
    MSE,
    MSIS,
    ND,
    NRMSE,
    RMSE,
    SMAPE,
    BaseMetricDefinition,
    MeanWeightedSumQuantileLoss,
)
from gluonts.itertools import batcher
from gluonts.model import Forecast, evaluate_model
from gluonts.model.forecast import QuantileForecast, SampleForecast
from gluonts.time_feature import get_seasonality
from omegaconf import DictConfig
from timesfm import configs
from tqdm.auto import tqdm

from tsfm_lens.chronos.circuitlens import CircuitLensChronos
from tsfm_lens.chronos_bolt.circuitlens import CircuitLensBolt
from tsfm_lens.circuitlens import BaseCircuitLens
from tsfm_lens.dataset import GiftEvalDataset
from tsfm_lens.moirai.circuitlens import CircuitLensMoirai
from tsfm_lens.timesfm.circuitlens import CircuitLensTimesFM
from tsfm_lens.toto.circuitlens import CircuitLensToto
from tsfm_lens.toto.predictor import EvalTask, toto_evaluate_tasks
from tsfm_lens.utils import clear_cuda_cache, set_seed

ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets")
DATASET_PROPERTIES_PATH = os.path.join(ASSETS_DIR, "dataset_properties.json")
DEFAULT_RESULTS_SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")

logger = logging.getLogger(__name__)

# Dataset name normalization
PRETTY_NAMES = {
    "saugeenday": "saugeen",
    "temperature_rain_with_missing": "temperature_rain",
    "kdd_cup_2018_with_missing": "kdd_cup_2018",
    "car_parts_with_missing": "car_parts",
}

METRICS = [
    MSE(forecast_type="mean"),
    MSE(forecast_type=0.5),  # type: ignore[arg-type]
    MAE(),
    MASE(),
    MAPE(),
    SMAPE(),
    MSIS(),
    RMSE(),
    NRMSE(),
    ND(),
    MeanWeightedSumQuantileLoss(quantile_levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
]


def get_metric_name(metric: BaseMetricDefinition) -> str:
    """Generate metric name string for CSV header."""
    name_map = {"SMAPE": "sMAPE", "MeanWeightedSumQuantileLoss": "mean_weighted_sum_quantile_loss"}
    metric_name = name_map.get(metric.__class__.__name__, metric.__class__.__name__)
    if hasattr(metric, "forecast_type"):
        return f"eval_metrics/{metric_name}[{metric.forecast_type}]"  # type: ignore[attr-defined]
    if hasattr(metric, "quantile"):
        return f"eval_metrics/{metric_name}[{metric.quantile}]"  # type: ignore[attr-defined]
    return f"eval_metrics/{metric_name}"


def create_forecasts(
    outputs: np.ndarray, test_data: list[dict], forecast_type: ForecastType, quantiles: list[float] | None = None
) -> list[Forecast]:
    """Convert forecast outputs to GluonTS Forecast objects."""
    forecasts = []
    for item, ts in zip(outputs, test_data):
        start = ts["start"] + len(ts["target"])
        if forecast_type == ForecastType.SAMPLES:
            forecasts.append(SampleForecast(samples=item, start_date=start))
        else:
            forecasts.append(
                QuantileForecast(forecast_arrays=item, forecast_keys=list(map(str, quantiles)), start_date=start)  # type: ignore[arg-type]
            )
    return forecasts


def parse_dataset_key(ds_name: str, dataset_properties_map: dict[str, dict]) -> tuple[str, str]:
    """Extract and normalize dataset key and frequency."""
    if "/" in ds_name:
        ds_key, ds_freq = ds_name.split("/")[:2]
    else:
        ds_key = ds_name
        ds_freq = dataset_properties_map.get(ds_key.lower(), {}).get("frequency")
    return PRETTY_NAMES.get(ds_key.lower(), ds_key.lower()), ds_freq  # type: ignore[arg-type]


def get_dataset_terms(ds_name: str, selected_term: str, medium_long_datasets: list[str]) -> list[str]:
    """Determine which terms to evaluate for a dataset."""
    has_medium_long = ds_name in medium_long_datasets
    term_map = {
        "short": ["short"],
        "medium": ["medium", "long"] if has_medium_long else [],
        "long": ["medium", "long"] if has_medium_long else [],
        "all": ["short", "medium", "long"] if has_medium_long else ["short"],
    }
    if selected_term not in term_map:
        raise ValueError(f"Invalid term: {selected_term}")
    return term_map[selected_term]


def load_model(
    model_type: str, cfg: DictConfig, device: torch.device | None = None, torch_dtype: torch.dtype | None = None
) -> BaseCircuitLens:
    """Load and return the appropriate model pipeline."""
    device_map = device or cfg.eval.device
    torch_dtype = torch_dtype or getattr(torch, cfg.eval.torch_dtype) or torch.float32
    assert isinstance(torch_dtype, torch.dtype)
    logger.info(f"Using device: {device}, torch_dtype: {torch_dtype}")
    # NOTE: we follow the settings of the Gift-Eval example notebooks and of each model's github demo
    # TimesFM 2.5 max context length is set to 2048 (see TimesFMPipelineCustom constructor)
    # Chronos and Chronos Bolt max context length is default 512 (see the Chronos offical repo)
    # Toto max context length is set to 4096 (see tsfm_lens/toto/predictor.py and toto offical repo)
    # Moirai max context length is set to 4000 here explicitly, following their Gift-Eval convention
    loaders = {
        "timesfm": lambda: CircuitLensTimesFM(cfg.timesfm.model_id, device_map=device_map),
        "chronos": lambda: CircuitLensChronos.from_pretrained(
            cfg.chronos.model_id, device_map=device_map, torch_dtype=torch_dtype
        ),
        "chronos_bolt": lambda: CircuitLensBolt.from_pretrained(
            cfg.chronos_bolt.model_id, device_map=device_map, torch_dtype=torch_dtype
        ),
        "toto": lambda: CircuitLensToto(cfg.toto.model_id, device_map=device_map),
        "moirai": lambda: CircuitLensMoirai(
            cfg.moirai.model_id,
            context_length=4000,  # NOTE: this is what the Moirai 1.1 was trained with
            prediction_length=1,  # NOTE: this is arbitrary, just a placeholder
            patch_size=cfg.moirai.patch_size,  # NOTE: try "auto". This can also be replaced in .predict() method
            num_samples=cfg.moirai.num_samples,  # NOTE: this is a placeholder, can be replaced in .predict() method
            target_dim=1,
            device=device_map,
        ),
    }
    # logger.info the model type's config
    logger.info(f"Using {model_type} config: {dict(cfg)[model_type]}")
    if model_type not in loaders:
        raise ValueError(f"Invalid model type: {model_type}")
    pipeline = loaders[model_type]()
    logger.info(f"Model: {pipeline} (layers={pipeline.num_layers}, heads={pipeline.num_heads})")
    pipeline.set_to_eval_mode()
    return pipeline


def setup_ablations_by_strategy(
    pipeline: BaseCircuitLens,
    ablations_layers_lst: list[int] | None,
    ablations_types: list[Literal["head", "mlp"]],
    ablate_n_heads_per_layer: int | None,
    head_selection_strategy: str,
    rseed: int,
) -> str | None:
    """
    Configure ablation hooks on the model pipeline based on the specified strategy.

    Selects which attention heads to ablate using one of several strategies, then
    registers ablation hooks on the pipeline. Supports both ranking-based selection
    (using precomputed srank or alignment scores) and random selection.

    Args:
        pipeline: The CircuitLens model pipeline to configure ablations on.
        ablations_layers_lst: List of layer indices to ablate (None to skip ablations).
        ablations_types: List of component types to ablate (e.g., ["head"]).
        ablate_n_heads_per_layer: Number of heads to ablate per layer (None for all).
        head_selection_strategy: One of:
            - "random": Random head selection (seeded by rseed)
            - "srank": Select heads with lowest stable rank (least important)
            - "srank_reverse": Select heads with highest srank (most important)
            - "alignment": Select heads with highest alignment scores
            - "alignment_reverse": Select heads with lowest alignment scores
        rseed: Random seed for reproducible random selection.

    Returns:
        A string describing the ablation configuration (e.g., "head_layers_1-2_heads-10"),
        or None if no ablations are configured.

    Raises:
        ValueError: If the required ranking file is not found for srank/alignment strategies.
    """
    if ablations_layers_lst is None:
        logger.info("No ablations configured")
        return None

    layers = ablations_layers_lst
    n_heads = ablate_n_heads_per_layer
    strategy = head_selection_strategy

    logger.info(f"Ablations: types={ablations_types}, layers={layers}, n_heads={n_heads}, strategy={strategy}")

    pipeline.remove_all_hooks()
    if hasattr(pipeline, "reset_attribution_inputs_and_outputs"):
        pipeline.reset_attribution_inputs_and_outputs()

    # Select heads based on strategy
    if strategy == "random":
        rng = np.random.default_rng(rseed)
        logger.info(f"Selecting heads randomly with seed {rseed}")
        if n_heads is None:
            heads_to_ablate = [(layer, h) for layer in layers for h in range(pipeline.num_heads)]
        else:
            heads_to_ablate = []
            for layer in layers:
                selected = rng.choice(pipeline.num_heads, size=n_heads, replace=False)
                heads_to_ablate.extend((layer, h) for h in selected)
    elif strategy in ["srank", "srank_reverse", "alignment", "alignment_reverse"]:
        # Load ranking file (srank or alignment)
        ranking_type = "srank" if "srank" in strategy else "alignment"
        ranking_path = os.path.join(
            ASSETS_DIR, f"{pipeline.__class__.__name__}_{ranking_type}_low_to_high_by_layer.json"
        )
        if not os.path.exists(ranking_path):
            raise ValueError(f"Ranking file not found: {ranking_path}")

        logger.info(f"Loading {ranking_type} rankings from {ranking_path}")
        rankings = json.load(open(ranking_path))

        # Reverse order for "reverse" strategies or alignment (high-to-low)
        reverse = strategy.endswith("_reverse") or strategy == "alignment"

        heads_to_ablate = []
        for layer in layers:
            head_list = list(rankings[str(layer)].keys())
            if reverse:
                head_list = head_list[::-1]
            heads_to_ablate.extend((int(layer), int(h)) for h in head_list[:n_heads])
    else:
        raise ValueError(f"Invalid strategy: {strategy}")

    logger.info(f"{len(heads_to_ablate)} heads to ablate: {heads_to_ablate}")
    pipeline.add_ablation_hooks_explicit(
        ablations_types=ablations_types,
        layers_to_ablate_mlp=layers,
        heads_to_ablate=heads_to_ablate,
    )

    return f"{'-'.join(ablations_types)}_layers_{'-'.join(map(str, layers))}_heads-{n_heads or 'all'}"


def _get_ablations_summary_str(pipeline: BaseCircuitLens, heads_to_ablate: list[tuple[int, int]]) -> str:
    n_heads_ablated = len(heads_to_ablate)
    logger.info(f"{n_heads_ablated} heads_to_ablate")

    ablations_summary_str = f"ablate_{n_heads_ablated}_heads"

    model_class = pipeline.__class__.__name__
    is_chronos = model_class in ["CircuitLensChronos", "CircuitLensBolt"]
    if is_chronos:
        # NOTE: sa_head_ablation_handles should be the same as ca_head_ablation_handles
        layers_without_heads = list(pipeline.sa_head_ablation_handles.keys())  # type: ignore
        assert layers_without_heads == list(pipeline.ca_head_ablation_handles.keys())  # type: ignore
        layers_without_mlps = list(pipeline.mlp_ablation_handles.keys())  # type: ignore
    else:
        layers_without_heads = list(pipeline.head_ablation_handles.keys())  # type: ignore
        layers_without_mlps = list(pipeline.mlp_ablation_handles.keys())  # type: ignore

    ablations_summary_str_suffix = ""
    if layers_without_heads and layers_without_mlps:
        ablations_summary_str_suffix = f"za_heads_layers_{'-'.join(map(str, layers_without_heads))}-mlps_layers_{'-'.join(map(str, layers_without_mlps))}"
    elif layers_without_heads:
        ablations_summary_str_suffix = f"za_heads_layers_{'-'.join(map(str, layers_without_heads))}"
    elif layers_without_mlps:
        ablations_summary_str_suffix = f"za_mlps_layers_{'-'.join(map(str, layers_without_mlps))}"
    else:
        ablations_summary_str_suffix = ""

    if ablations_summary_str_suffix:
        ablations_summary_str += "_" + ablations_summary_str_suffix

    return ablations_summary_str


def setup_ablations_to_heads_at_1pp(
    pipeline: BaseCircuitLens,
    chosen_layers: list[int],
    num_heads_per_layer_to_skip: int,
    layers_to_keep_at_heads1pp: list[int],
    chosen_layers_mlp: list[int],
) -> str | None:
    """
    Configure ablations to bring heads to 1 principal component (1pp) performance.

    Loads precomputed head ablation configurations from a JSON file and applies them
    to the pipeline. The ablation list is filtered to only include the specified layers,
    then for each layer, the last `num_heads_per_layer_to_skip` heads are removed from
    ablation (i.e., kept active) unless the layer is in `layers_to_keep_at_heads1pp`.

    Args:
        pipeline: The CircuitLens model pipeline to configure ablations on.
        chosen_layers: List of layer indices to include in the ablation.
        num_heads_per_layer_to_skip: Number of heads to skip (keep active) per layer.
            If 0, all chosen layers are treated as layers to keep at heads1pp.
        layers_to_keep_at_heads1pp: Layers where all heads from the JSON config should
            be ablated (no heads skipped).
        chosen_layers_mlp: List of layer indices for MLP ablation.

    Returns:
        A string summarizing the ablation configuration (e.g., "heads1pp_ablate_N_heads_..."),
        or None if no ablations are configured.

    Note:
        The JSON file contains [layer, head] pairs specifying which heads to ablate.
        Heads are grouped by layer and sorted, then the last `num_heads_per_layer_to_skip`
        heads are removed from each layer's ablation list (except for protected layers).
    """
    heads_to_ablate = json.load(
        open(os.path.join(ASSETS_DIR, f"{pipeline.__class__.__name__}_ablate_for_heads_at_1pp.json"))
    )
    heads_to_ablate = [config for config in heads_to_ablate if config[0] in chosen_layers]
    if num_heads_per_layer_to_skip == 0:
        layers_to_keep_at_heads1pp = chosen_layers

    heads_to_ablate = [
        config
        for layer, group in groupby(sorted(heads_to_ablate, key=lambda x: x[0]), key=lambda x: x[0])
        for config in (
            list(group) if layer in layers_to_keep_at_heads1pp else list(group)[:-num_heads_per_layer_to_skip]
        )
    ]

    logger.info(f"Ablating {len(heads_to_ablate)} heads")

    pipeline.remove_all_hooks()
    if hasattr(pipeline, "reset_attribution_inputs_and_outputs"):
        pipeline.reset_attribution_inputs_and_outputs()

    logger.info(f"{len(heads_to_ablate)} heads to ablate: {heads_to_ablate}")
    pipeline.add_ablation_hooks_explicit(
        ablations_types=["head", "mlp"],
        layers_to_ablate_mlp=chosen_layers_mlp,
        heads_to_ablate=heads_to_ablate,
    )

    for layer in chosen_layers:
        num_heads = sum(1 for config in heads_to_ablate if config[0] == layer)
        logger.info(f"Layer {layer}: {num_heads} heads")

    ablations_summary_str = _get_ablations_summary_str(pipeline, heads_to_ablate)
    logger.info(f"Ablations summary string: {ablations_summary_str}")

    # We keep track of the additional flexibility to the heads@1pp ablation
    extra_suffix = ""
    if num_heads_per_layer_to_skip > 0 and len(chosen_layers) > 0:
        logger.info(
            f"Additional flexibility to the heads@1pp ablation: num_heads_per_layer_to_skip={num_heads_per_layer_to_skip}, layers_to_keep_at_heads1pp={'-'.join(map(str, layers_to_keep_at_heads1pp))}"
        )
        extra_suffix = f"_num_heads_per_layer_to_skip-{num_heads_per_layer_to_skip}"
        if layers_to_keep_at_heads1pp:
            extra_suffix += f"_layers_to_keep_at_heads1pp-{'-'.join(map(str, layers_to_keep_at_heads1pp))}"
    ablations_summary_str += extra_suffix
    logger.info(f"Ablations summary string with extra suffix: {ablations_summary_str}")

    return f"heads1pp_{ablations_summary_str}"


class ChronosPredictor:
    def __init__(self, pipeline, prediction_length: int, num_samples: int = 20, rseed: int = 123):
        self.pipeline = pipeline
        self.prediction_length = prediction_length
        self.num_samples = num_samples
        self.rseed = rseed

    def predict(self, test_data_input, batch_size: int = 1024) -> list[Forecast]:
        predict_kwargs = {"limit_prediction_length": False}
        if self.pipeline.forecast_type == ForecastType.SAMPLES:
            predict_kwargs["num_samples"] = self.num_samples  # type: ignore[assignment]

        forecast_outputs = []
        while True:
            try:
                for batch in tqdm(batcher(test_data_input, batch_size=batch_size)):
                    context = [torch.tensor(entry["target"]) for entry in batch]
                    set_seed(self.rseed)
                    forecast_outputs.append(
                        self.pipeline.predict(
                            context, prediction_length=self.prediction_length, **predict_kwargs
                        ).numpy()
                    )
                break
            except torch.cuda.OutOfMemoryError:
                logger.warning(f"OOM at batch_size {batch_size}, reducing to {batch_size // 2}")
                batch_size //= 2
                forecast_outputs = []

        quantiles = getattr(self.pipeline, "quantiles", None)
        return create_forecasts(
            np.concatenate(forecast_outputs), test_data_input, self.pipeline.forecast_type, quantiles
        )


class TimesFmPredictor:
    def __init__(self, tfm, prediction_length: int, rseed: int = 123):
        self.tfm = tfm
        self.prediction_length = prediction_length
        self.quantiles = list(np.arange(1, 10) / 10.0)
        self.rseed = rseed

    def predict(self, test_data_input, batch_size: int = 1024) -> list[Forecast]:
        forecast_outputs = []
        for batch in tqdm(batcher(test_data_input, batch_size=batch_size)):
            context = [np.array(entry["target"]) for entry in batch]
            max_ctx = max(arr.shape[0] for arr in context)
            max_ctx = ((max_ctx + self.tfm.model.p - 1) // self.tfm.model.p) * self.tfm.model.p

            self.tfm.compile(
                forecast_config=configs.ForecastConfig(
                    max_context=min(2048, max_ctx),
                    max_horizon=1024,
                    infer_is_positive=True,
                    use_continuous_quantile_head=True,
                    fix_quantile_crossing=True,
                    force_flip_invariance=True,
                    return_backcast=False,
                    normalize_inputs=True,
                    per_core_batch_size=128,
                )
            )

            set_seed(self.rseed)
            _, full_preds = self.tfm.forecast(horizon=self.prediction_length, inputs=context)
            forecast_outputs.append(full_preds[:, : self.prediction_length, 1:].transpose((0, 2, 1)))

        return create_forecasts(
            np.concatenate(forecast_outputs), test_data_input, ForecastType.QUANTILES, self.quantiles
        )


def run_toto_evaluation(
    pipeline: CircuitLensToto,
    cfg: DictConfig,
    selected_dataset_names: list[str],
    selected_term: str,
    medium_long_datasets: list[str],
    dataset_properties_map: dict[str, dict],
    csv_path: str,
):
    """Run Toto-specific evaluation pipeline."""
    tasks = []
    for ds_name in tqdm(selected_dataset_names, desc="Building tasks"):
        for term in get_dataset_terms(ds_name, selected_term, medium_long_datasets):
            ds_key, ds_freq = parse_dataset_key(ds_name, dataset_properties_map)
            tasks.append(
                EvalTask(
                    dataset_name=ds_name,
                    term=term,
                    checkpoint_path="Toto-Open-Base-1.0",
                    num_samples=cfg.toto.samples_per_batch,
                    use_kv_cache=cfg.toto.use_kv_cache,
                    seed=cfg.eval.rseed,
                    dataset_properties_map=dataset_properties_map,
                    dataset_key=ds_key,
                    dataset_frequency=ds_freq,  # type: ignore[arg-type]
                    evaluation_target="test",
                    pad_short_series=cfg.toto.pad_short_series,
                )
            )

    logger.info(f"Evaluating {len(tasks)} tasks")
    results = toto_evaluate_tasks(tasks, cfg.eval.data_dir, pipeline)
    results.to_csv(csv_path, index=False)


def run_standard_evaluation(
    pipeline: BaseCircuitLens,
    model_type: Literal["timesfm", "chronos", "chronos_bolt", "moirai"],
    cfg: DictConfig,
    selected_dataset_names: list[str],
    selected_term: str,
    medium_long_datasets: list[str],
    dataset_properties_map: dict[str, dict],
    csv_path: str,
    row_header: list[str],
    batch_size: int = 1024,
):
    """
    Run evaluation pipeline for TimesFM, Chronos, Chronos-Bolt, and Moirai models.

    Iterates over selected datasets and terms, creates forecasts using the
    appropriate predictor, computes evaluation metrics, and writes results
    incrementally to a CSV file.

    Args:
        pipeline: The CircuitLens model pipeline to use for predictions.
        model_type: Model identifier ("timesfm", "chronos", or "chronos_bolt", "moirai").
        cfg: Hydra config containing evaluation settings (data_dir, rseed, etc.).
        selected_dataset_names: List of GIFT-Eval dataset names to evaluate.
        selected_term: Forecast horizon term filter ("short", "medium", "long", "all").
        medium_long_datasets: List of dataset names that support medium/long terms.
        dataset_properties_map: Mapping of dataset keys to their properties (domain, num_variates).
        csv_path: Output path for the results CSV file.
        row_header: Column headers for the CSV (dataset, model, metrics, metadata).

    Note:
        Results are written incrementally (one row per dataset/term) to avoid
        data loss if evaluation is interrupted. Multivariate datasets are
        automatically converted to univariate if needed.
    """
    logger.info(f"Running standard evaluation with batch size: {batch_size}")
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(row_header)

    for ds_name in tqdm(selected_dataset_names, desc="Evaluating datasets"):
        suggested_batch_size = batch_size
        for term in get_dataset_terms(ds_name, selected_term, medium_long_datasets):
            ds_key, ds_freq = parse_dataset_key(ds_name, dataset_properties_map)
            logger.info(f"Forecasting on dataset: {ds_name}, term: {term}, key: {ds_key}, freq: {ds_freq}")

            # option for univariate conversion, set by cfg.eval.gift_eval.to_univariate
            to_univariate = cfg.eval.gift_eval.to_univariate
            if model_type not in ["moirai", "moirai2"]:
                # NOTE: since Moirai supports multivariate time series forecast, there is no need to convert the original data into univariate
                # Check if univariate conversion needed
                if not to_univariate:
                    to_univariate = (
                        GiftEvalDataset(
                            name=ds_name, term=term, to_univariate=False, data_dir=cfg.eval.data_dir
                        ).target_dim
                        != 1
                    )
            dataset = GiftEvalDataset(name=ds_name, term=term, to_univariate=to_univariate, data_dir=cfg.eval.data_dir)
            # NOTE: for debugging, set a breakpoint here, and in the pdb debugger:
            # from itertools import islice; context_target = next(islice(dataset.test_data.input, 0, 1))["target"]

            # Create predictor
            if model_type == "timesfm":
                predictor = TimesFmPredictor(pipeline.model, dataset.prediction_length, cfg.eval.rseed)
            elif model_type in ["chronos", "chronos_bolt"]:
                predictor = ChronosPredictor(
                    pipeline, dataset.prediction_length, num_samples=cfg.chronos.num_samples, rseed=cfg.eval.rseed
                )
            elif model_type == "moirai":
                # set the Moirai hyperparameter according to each dataset, then create the predictor
                pipeline.model.hparams.prediction_length = dataset.prediction_length
                pipeline.model.hparams.target_dim = dataset.target_dim
                pipeline.model.hparams.past_feat_dynamic_real_dim = dataset.past_feat_dynamic_real_dim
                # NOTE: context_length is stored in pipeline.context_length
                # and model width (embedding_dim) is stored in pipeline.model.module.mask_encoding.embedding_dim

                # TODO: need to use min(dataset context length, model context length)
                # # Calculate optimal batch size based on available GPU memory
                # suggested_batch_size = calculate_optimal_batch_size(
                #     model=pipeline.model,
                #     target_dim=dataset.target_dim,
                #     prediction_length=dataset.prediction_length,
                #     context_length=pipeline.model.hparams.context_length,
                #     use_kv_cache=True,
                #     num_samples=cfg.moirai.num_samples,
                # )
                # logger.info(f"Suggested batch size: {suggested_batch_size}")

                set_seed(cfg.eval.rseed)
                predictor = pipeline.model.create_predictor(
                    batch_size=suggested_batch_size, device=pipeline.model.device
                )  # NOTE: this is hardcoded following the Moirai Gift-Eval example notebook
                # NOTE: we can double check the past_length (context length) for this predictor by inspecting:
                # predictor.__init_args__["input_transform"].__init_passed_kwargs__["transformations"]

            elif model_type == "toto":
                raise ValueError("Use run_toto_evaluation() for Toto models")
            else:
                raise NotImplementedError(f"Predictor not implemented for model type: {model_type}")

            # Evaluate
            # from gluonts: https://ts.gluon.ai/dev/_modules/gluonts/model/evaluation.html#evaluate_model
            res = evaluate_model(
                predictor,  # type: ignore[arg-type]
                test_data=dataset.test_data,
                metrics=METRICS,
                batch_size=suggested_batch_size,  # default batch size is 1024
                axis=None,
                mask_invalid_label=True,
                allow_nan_forecast=False,
                seasonality=get_seasonality(dataset.freq),  # season length
            )

            # Build result row
            ds_props = dataset_properties_map[ds_key]
            row = [f"{ds_key}/{ds_freq}/{term}", model_type]
            for header in row_header[2:]:
                if header.startswith("eval_metrics/"):
                    row.append(res[header.replace("eval_metrics/", "")][0])  # type: ignore[index]
                elif header == "domain":
                    row.append(ds_props["domain"])
                elif header == "num_variates":
                    row.append(ds_props["num_variates"])
                elif header == "num_datasets":
                    row.append(str(len(dataset.test_data)))  # type: ignore[arg-type]
                else:
                    row.append(str(None))  # type: ignore[arg-type]

            with open(csv_path, "a", newline="") as f:
                csv.writer(f).writerow(row)
            logger.info(f"Completed: {ds_name}/{term}")


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg):
    # Setup
    device = torch.device(cfg.eval.device if torch.cuda.is_available() else "cpu")
    torch_dtype = getattr(torch, cfg.eval.torch_dtype)
    logger.info(f"Device: {device}, dtype: {torch_dtype}")

    # Dataset selection
    gift_cfg = cfg.eval.gift_eval
    all_datasets = list(set(gift_cfg.short_datasets + gift_cfg.medium_long_datasets))
    selected_datasets = gift_cfg.dataset_names or all_datasets
    if isinstance(selected_datasets, str):
        selected_datasets = [selected_datasets]
    if gift_cfg.max_num_datasets:
        selected_datasets = selected_datasets[: gift_cfg.max_num_datasets]

    logger.info(f"Datasets: {len(selected_datasets)}, Term: {gift_cfg.term}")

    with open(DATASET_PROPERTIES_PATH) as f:
        dataset_properties_map = json.load(f)

    # Clear CUDA cache and load model
    clear_cuda_cache(device)

    pipeline = load_model(cfg.ablation.model_type, cfg, device=device, torch_dtype=torch_dtype)

    head_selection_strategy = cfg.ablation.head_selection_strategy

    # NOTE: setup_ablations* modifies the pipeline in-place by adding the ablations hooks
    ablations_name = None
    output_subdir_name = f"original_rseed-{cfg.eval.rseed}"
    if head_selection_strategy is None:
        logger.info("Skipping ablations")
        pass
    elif head_selection_strategy == "heads1pp":
        logger.info("Ablating to heads@1pp")
        ablations_name = setup_ablations_to_heads_at_1pp(
            pipeline,
            chosen_layers=cfg.ablation.to_heads_at_1pp.chosen_layers,
            num_heads_per_layer_to_skip=cfg.ablation.to_heads_at_1pp.num_heads_per_layer_to_skip,
            layers_to_keep_at_heads1pp=cfg.ablation.to_heads_at_1pp.layers_to_keep_at_heads1pp,
            chosen_layers_mlp=cfg.ablation.to_heads_at_1pp.chosen_layers_mlp,
        )
        output_subdir_name = f"heads1pp_rseed-{cfg.eval.rseed}"

    elif head_selection_strategy in ["random", "srank", "srank_reverse", "alignment", "alignment_reverse"]:
        logger.info(f"Ablating by {head_selection_strategy}")
        ablations_name = setup_ablations_by_strategy(
            pipeline,
            ablations_layers_lst=cfg.ablation.ablations_layers_lst,
            ablations_types=cfg.ablation.ablations_types,
            ablate_n_heads_per_layer=cfg.ablation.ablate_n_heads_per_layer,
            head_selection_strategy=head_selection_strategy,
            rseed=cfg.eval.rseed,
        )
        # Output paths
        if not cfg.ablation.model_name_str:
            raise ValueError("model_name_str is required")

        strategy_prefix = {s: f"{s}_" for s in ["alignment", "alignment_reverse", "srank", "srank_reverse", "random"]}
        output_subdir_name = f"{strategy_prefix[head_selection_strategy]}rseed-{cfg.eval.rseed}"
    else:
        raise ValueError(f"Invalid head selection strategy: {head_selection_strategy}")

    # directory to save results into
    output_dir = os.path.join(
        cfg.eval.results_save_dir or DEFAULT_RESULTS_SAVE_DIR,
        cfg.ablation.model_name_str,
        output_subdir_name,
    )
    logger.info(f"Results will be saved to: {output_dir}")

    # Suppress GluonTS warning
    logging.getLogger("gluonts.model.forecast").addFilter(
        type("Filter", (logging.Filter,), {"filter": lambda s, r: "mean prediction" not in r.getMessage().lower()})()
    )

    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"{ablations_name or 'original'}_{gift_cfg.term}_results.csv")
    logger.info(f"Results: {csv_path}")

    # Run evaluation
    row_header = (
        ["dataset", "model"] + [get_metric_name(m) for m in METRICS] + ["domain", "num_variates", "num_datasets"]
    )

    if cfg.ablation.model_type == "toto":
        run_toto_evaluation(
            pipeline,  # type: ignore[arg-type]
            cfg,
            selected_datasets,
            gift_cfg.term,
            gift_cfg.medium_long_datasets,
            dataset_properties_map,
            csv_path,
        )
    else:
        run_standard_evaluation(
            pipeline,
            cfg.ablation.model_type,
            cfg,
            selected_datasets,
            gift_cfg.term,
            gift_cfg.medium_long_datasets,
            dataset_properties_map,
            csv_path,
            row_header,
            batch_size=cfg.eval.batch_size,
        )

    logger.info(f"\nFinal results: {csv_path}")
    logger.info(f"\n{pd.read_csv(csv_path)}")


if __name__ == "__main__":
    main()
