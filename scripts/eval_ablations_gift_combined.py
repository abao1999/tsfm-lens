"""
Quick Start: Running TSFMs, with optional ablations, on gift-eval benchmark

We will use the `Dataset` class to load the data and run the model.
Configuration is managed via Hydra config files.
"""

import csv
import gc
import json
import logging
import os

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
    MeanWeightedSumQuantileLoss,
)
from gluonts.itertools import batcher
from gluonts.model import Forecast, evaluate_model
from gluonts.model.forecast import QuantileForecast, SampleForecast
from gluonts.time_feature import get_seasonality
from tqdm.auto import tqdm

from tsfm_lens.chronos.circuitlens import CircuitLensChronos
from tsfm_lens.chronos_bolt.circuitlens import CircuitLensBolt
from tsfm_lens.dataset import GiftEvalDataset
from tsfm_lens.timesfm.circuitlens import CircuitLensTimesFM
from tsfm_lens.toto.circuitlens import CircuitLensToto
from tsfm_lens.toto.predictor import EvalTask, toto_evaluate_tasks
from tsfm_lens.utils import set_seed
from timesfm import configs

ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets")
DATASET_PROPERTIES_PATH = os.path.join(ASSETS_DIR, "dataset_properties.json")
DEFAULT_RESULTS_SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")

logger = logging.getLogger(__name__)


class WarningFilter(logging.Filter):
    def __init__(self, text_to_filter):
        super().__init__()
        self.text_to_filter = text_to_filter

    def filter(self, record):
        return self.text_to_filter not in record.getMessage()


def create_forecasts(outputs, test_data, forecast_type, quantiles=None):
    """Convert forecast outputs to GluonTS Forecast objects."""
    forecasts = []
    for item, ts in zip(outputs, test_data):
        forecast_start_date = ts["start"] + len(ts["target"])
        if forecast_type == ForecastType.SAMPLES:
            forecasts.append(SampleForecast(samples=item, start_date=forecast_start_date))
        elif forecast_type == ForecastType.QUANTILES or quantiles is not None:
            forecasts.append(
                QuantileForecast(
                    forecast_arrays=item,
                    forecast_keys=list(map(str, quantiles)),  # type: ignore[arg-type]
                    start_date=forecast_start_date,
                )
            )
    return forecasts


class ChronosPredictor:
    def __init__(self, pipeline, num_samples: int, prediction_length: int, rseed: int = 123):
        print("prediction_length:", prediction_length)
        self.pipeline = pipeline
        self.prediction_length = prediction_length
        self.num_samples = num_samples
        self.rseed = rseed

    def predict(self, test_data_input, batch_size: int = 1024) -> list[Forecast]:
        predict_kwargs = (
            {"num_samples": self.num_samples} if self.pipeline.forecast_type == ForecastType.SAMPLES else {}
        )
        # predict_kwargs = {"num_samples": self.num_samples, "limit_prediction_length": False}
        predict_kwargs["limit_prediction_length"] = False
        while True:
            try:
                forecast_outputs = []
                for batch in tqdm(batcher(test_data_input, batch_size=batch_size)):
                    context = [torch.tensor(entry["target"]) for entry in batch]
                    set_seed(self.rseed)
                    forecast_outputs.append(
                        self.pipeline.predict(
                            context, prediction_length=self.prediction_length, **predict_kwargs
                        ).numpy()
                    )
                forecast_outputs = np.concatenate(forecast_outputs)
                break
            except torch.cuda.OutOfMemoryError:
                print(f"OutOfMemoryError at batch_size {batch_size}, reducing to {batch_size // 2}")
                batch_size //= 2

        quantiles = None
        if hasattr(self.pipeline, "quantiles"):
            quantiles = self.pipeline.quantiles
        return create_forecasts(forecast_outputs, test_data_input, self.pipeline.forecast_type, quantiles)


class TimesFmPredictor:
    def __init__(self, tfm, prediction_length: int, rseed: int = 123, *args, **kwargs):
        self.tfm = tfm
        self.prediction_length = prediction_length
        self.quantiles = list(np.arange(1, 10) / 10.0)
        self.rseed = rseed

    def predict(self, test_data_input, batch_size: int = 1024) -> list[Forecast]:
        forecast_outputs = []
        for batch in tqdm(batcher(test_data_input, batch_size=batch_size)):
            context = [np.array(entry["target"]) for entry in batch]
            max_context = max(arr.shape[0] for arr in context)
            max_context = ((max_context + self.tfm.model.p - 1) // self.tfm.model.p) * self.tfm.model.p

            self.tfm.compile(
                forecast_config=configs.ForecastConfig(
                    max_context=min(2048, max_context),
                    max_horizon=1024,
                    infer_is_positive=True,
                    use_continuous_quantile_head=True,
                    fix_quantile_crossing=True,
                    force_flip_invariance=True,
                    return_backcast=False,
                    normalize_inputs=True,
                    per_core_batch_size=128,
                ),
            )

            set_seed(self.rseed)
            _, full_preds = self.tfm.forecast(horizon=self.prediction_length, inputs=context)
            forecast_outputs.append(full_preds[:, : self.prediction_length, 1:].transpose((0, 2, 1)))

        forecast_outputs = np.concatenate(forecast_outputs)
        return create_forecasts(forecast_outputs, test_data_input, ForecastType.QUANTILES, self.quantiles)


def get_metric_name(metric):
    """Generate metric name string for CSV header."""
    name_map = {"SMAPE": "sMAPE", "MeanWeightedSumQuantileLoss": "mean_weighted_sum_quantile_loss"}
    metric_name = name_map.get(metric.__class__.__name__, metric.__class__.__name__)

    if hasattr(metric, "forecast_type"):
        return f"eval_metrics/{metric_name}[{metric.forecast_type}]"
    elif hasattr(metric, "quantile"):
        return f"eval_metrics/{metric_name}[{metric.quantile}]"
    return f"eval_metrics/{metric_name}"


def get_dataset_terms(ds_name, selected_term, medium_long_datasets):
    """Determine which terms to evaluate for a dataset."""
    if selected_term == "short":
        return ["short"]
    elif selected_term in ["medium", "long"]:
        return ["medium", "long"] if ds_name in medium_long_datasets else []
    elif selected_term == "all":
        return ["short", "medium", "long"] if ds_name in medium_long_datasets else ["short"]
    raise ValueError(f"Invalid term: {selected_term}")


def parse_dataset_key(ds_name, dataset_properties_map):
    """Extract and normalize dataset key and frequency."""
    pretty_names = {
        "saugeenday": "saugeen",
        "temperature_rain_with_missing": "temperature_rain",
        "kdd_cup_2018_with_missing": "kdd_cup_2018",
        "car_parts_with_missing": "car_parts",
    }

    if "/" in ds_name:
        ds_key, ds_freq = ds_name.split("/")[:2]
    else:
        ds_key = ds_name
        ds_freq = dataset_properties_map.get(ds_key.lower(), {}).get("frequency")

    ds_key = pretty_names.get(ds_key.lower(), ds_key.lower())
    return ds_key, ds_freq


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg):
    ##### 1. Setup #####
    device = torch.device(cfg.eval.device if torch.cuda.is_available() else "cpu")
    torch_dtype = getattr(torch, cfg.eval.torch_dtype)
    assert isinstance(torch_dtype, torch.dtype)
    logger.info(f"Using device: {device}, with torch_dtype: {torch_dtype}")

    gift_eval_kwargs = dict(cfg.eval.gift_eval)
    selected_dataset_names = gift_eval_kwargs["dataset_names"]
    selected_dataset_names = (
        [selected_dataset_names] if isinstance(selected_dataset_names, str) else selected_dataset_names
    )
    selected_term = gift_eval_kwargs["term"]
    short_datasets = gift_eval_kwargs["short_datasets"]
    medium_long_datasets = gift_eval_kwargs["medium_long_datasets"]
    # get set of union between short_datasets and medium_long_datasets
    all_dataset_names = list(set(short_datasets + medium_long_datasets))

    if selected_dataset_names is None:
        selected_dataset_names = all_dataset_names

    if cfg.eval.gift_eval.max_num_datasets is not None:
        selected_dataset_names = selected_dataset_names[: cfg.eval.gift_eval.max_num_datasets]
        logger.info(f"Selected first {cfg.eval.gift_eval.max_num_datasets} datasets")

    logger.info(f"Datasets: {selected_dataset_names}, Term: {selected_term}")

    # Load dataset properties
    with open(DATASET_PROPERTIES_PATH) as f:
        dataset_properties_map = json.load(f)

    # Setup metrics and CSV header
    metrics = [
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
    row_header = (
        ["dataset", "model"] + [get_metric_name(m) for m in metrics] + ["domain", "num_variates", "num_datasets"]
    )
    logger.info(f"Row header: {row_header}")

    ##### 2. Clear CUDA caches and load model #####
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info(f"GPU: {torch.cuda.get_device_name(device)}")
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

    model_type = cfg.ablation.model_type

    if model_type == "timesfm":
        pipeline = CircuitLensTimesFM(cfg.timesfm.model_id, device_map=cfg.eval.device)
        logger.info(f"Model: {pipeline} (layers={pipeline.num_layers}, heads={pipeline.num_heads})")
    elif model_type == "chronos":
        pipeline = CircuitLensChronos.from_pretrained(
            cfg.chronos.model_id, device_map=cfg.eval.device, torch_dtype=torch_dtype
        )
        logger.info(f"Model: {pipeline} (layers={pipeline.num_layers}, heads={pipeline.num_heads})")
    elif model_type == "chronos_bolt":
        pipeline = CircuitLensBolt.from_pretrained(
            cfg.chronos_bolt.model_id, device_map=cfg.eval.device, torch_dtype=torch_dtype
        )
        logger.info(f"Model: {pipeline} (layers={pipeline.num_layers}, heads={pipeline.num_heads})")
    elif model_type == "toto":
        pipeline = CircuitLensToto(cfg.toto.model_id, device_map=cfg.eval.device)
        logger.info(f"Model: {pipeline} (layers={pipeline.num_layers}, heads={pipeline.num_heads})")
    else:
        raise ValueError(f"Invalid model type: {model_type}")

    pipeline.set_to_eval_mode()

    ablations_name_str = None
    head_selection_strategy = cfg.ablation.head_selection_strategy

    ##### 3. Setup ablation hooks #####
    if cfg.ablation.ablations_layers_lst is not None:
        ablations_types = cfg.ablation.ablations_types
        layers_to_ablate = cfg.ablation.ablations_layers_lst
        ablate_n_heads_per_layer = cfg.ablation.ablate_n_heads_per_layer
        logger.info(
            f"Ablations types: {ablations_types}, layers to ablate: {layers_to_ablate}, ablate n heads per layer: {ablate_n_heads_per_layer}, head selection strategy: {head_selection_strategy}"
        )
        ablations_name_str = f"{'-'.join(ablations_types)}_layers_{'-'.join(map(str, layers_to_ablate))}_heads-{ablate_n_heads_per_layer or 'all'}"
        logger.info(f"Ablations summary string: {ablations_name_str}")
        pipeline.remove_all_hooks()
        if hasattr(pipeline, "reset_attribution_inputs_and_outputs"):
            pipeline.reset_attribution_inputs_and_outputs()
        if head_selection_strategy in ["srank", "srank_reverse"]:
            model_srank_per_layer_path = os.path.join(
                ASSETS_DIR, f"{pipeline.__class__.__name__}_srank_low_to_high_by_layer.json"
            )
            if not os.path.exists(model_srank_per_layer_path):
                raise ValueError(f"Srank per layer file not found at {model_srank_per_layer_path}")
            logger.info(f"Loading srank per layer from {model_srank_per_layer_path}")
            srank_per_layer = json.load(open(model_srank_per_layer_path))
            heads_to_ablate = []
            for layer in layers_to_ablate:
                head_list = list(srank_per_layer[str(layer)].keys())
                if head_selection_strategy == "srank_reverse":
                    head_list = head_list[::-1]
                heads_to_ablate.extend([(int(layer), int(head)) for head in head_list[:ablate_n_heads_per_layer]])
            logger.info(f"Ablating heads: {heads_to_ablate}")
            pipeline.add_ablation_hooks_explicit(
                ablations_types=ablations_types,
                layers_to_ablate_mlp=layers_to_ablate,
                heads_to_ablate=heads_to_ablate,
            )
        elif head_selection_strategy in ["alignment", "alignment_reverse"]:
            alignment_per_layer_path = os.path.join(
                ASSETS_DIR, f"{pipeline.__class__.__name__}_alignment_low_to_high_by_layer.json"
            )
            # if not exists, raise error
            if not os.path.exists(alignment_per_layer_path):
                raise ValueError(f"Alignment per layer file not found at {alignment_per_layer_path}")
            logger.info(f"Loading alignment per layer from {alignment_per_layer_path}")
            alignment_per_layer = json.load(open(alignment_per_layer_path))
            heads_to_ablate = []
            for layer in layers_to_ablate:
                head_list = list(alignment_per_layer[str(layer)].keys())
                if head_selection_strategy == "alignment":
                    # NOTE: we are selecting the heads from the highest to lowest alignment score
                    head_list = head_list[::-1]
                heads_to_ablate.extend([(int(layer), int(head)) for head in head_list[:ablate_n_heads_per_layer]])
            logger.info(f"Ablating heads: {heads_to_ablate}")
            pipeline.add_ablation_hooks_explicit(
                ablations_types=ablations_types,
                layers_to_ablate_mlp=layers_to_ablate,
                heads_to_ablate=heads_to_ablate,
            )
        elif head_selection_strategy == "random":
            rng = np.random.default_rng(cfg.eval.rseed)
            logger.info(f"Selecting heads randomly with rseed {cfg.eval.rseed} and rng: {rng}")
            # Ablate all heads per layer
            if ablate_n_heads_per_layer is None:
                heads_to_ablate = [(layer, head) for layer in layers_to_ablate for head in range(pipeline.num_heads)]
            # Ablate a fixed number of heads (random subset of heads) per layer
            else:
                heads_to_ablate = []
                for layer in layers_to_ablate:
                    heads_to_ablate_for_layer = rng.choice(
                        list(range(pipeline.num_heads)), size=ablate_n_heads_per_layer, replace=False
                    )
                    heads_to_ablate.extend([(layer, head) for head in heads_to_ablate_for_layer])  # type: ignore
            logger.info(f"Ablating heads: {heads_to_ablate}")
            pipeline.add_ablation_hooks_explicit(
                ablations_types=ablations_types,
                layers_to_ablate_mlp=layers_to_ablate,
                heads_to_ablate=heads_to_ablate,
            )
        else:
            raise ValueError(f"Invalid head selection strategy: {head_selection_strategy}")
    else:
        logger.info("No ablations configured")

    # Setup logging filter
    logging.getLogger("gluonts.model.forecast").addFilter(
        WarningFilter("The mean prediction is not stored in the forecast data")
    )

    ##### 4. Setup output paths #####
    if cfg.ablation.model_name_str is None:
        raise ValueError("model_name_str is required")

    strategy_prefix_map = {
        "alignment": "alignment_",
        "alignment_reverse": "alignment_reverse_",
        "srank": "srank_",
        "srank_reverse": "srank_reverse_",
        "random": "",
    }
    strategy_prefix = strategy_prefix_map[head_selection_strategy]

    output_dir = os.path.join(
        cfg.eval.results_save_dir or DEFAULT_RESULTS_SAVE_DIR,
        cfg.ablation.model_name_str,
        f"{strategy_prefix}rseed-{cfg.eval.rseed}",
    )

    os.makedirs(output_dir, exist_ok=True)
    csv_file_path = os.path.join(output_dir, f"{ablations_name_str or 'original'}_{selected_term}_results.csv")
    logger.info(f"Results: {csv_file_path}")

    ##### 5. Run evaluation #####

    if model_type == "toto":
        # Toto-specific evaluation pipeline
        assert isinstance(pipeline, CircuitLensToto)
        num_samples = cfg.toto.samples_per_batch
        use_kv_cache = cfg.toto.use_kv_cache
        pad_short_series = cfg.toto.pad_short_series
        evaluation_target = "test"

        logger.info(
            f"Using Toto-specific evaluation pipeline with num_samples: {num_samples}, use_kv_cache: {use_kv_cache}, pad_short_series: {pad_short_series}, evaluation_target: {evaluation_target}"
        )

        # Create all tasks as a flat list
        all_tasks = []
        for ds_num, ds_name in tqdm(enumerate(selected_dataset_names), total=len(selected_dataset_names)):
            logger.info(f"Processing dataset: {ds_name} ({ds_num + 1}/{len(selected_dataset_names)})")
            terms = get_dataset_terms(ds_name, selected_term, medium_long_datasets)
            logger.info(f"Terms: {terms}")

            for term in tqdm(terms):
                ds_key, ds_freq = parse_dataset_key(ds_name, dataset_properties_map)

                task = EvalTask(
                    dataset_name=ds_name,
                    term=term,
                    checkpoint_path="Toto-Open-Base-1.0",  # NOTE: this is purely for naming the column on the saved csv, doesn't actually load the model
                    num_samples=num_samples,
                    use_kv_cache=use_kv_cache,
                    seed=cfg.eval.rseed,
                    dataset_properties_map=dataset_properties_map,
                    dataset_key=ds_key,
                    dataset_frequency=ds_freq,
                    evaluation_target=evaluation_target,
                    pad_short_series=pad_short_series,
                )
                all_tasks.append(task)

        logger.info(f"Processing {len(all_tasks)} tasks sequentially")
        results = toto_evaluate_tasks(all_tasks, cfg.eval.data_dir, pipeline)
        results.to_csv(csv_file_path, index=False)

    else:
        # Standard evaluation pipeline for timesfm, chronos, chronos_bolt
        with open(csv_file_path, "w", newline="") as csvfile:
            csv.writer(csvfile).writerow(row_header)

        for ds_num, ds_name in tqdm(enumerate(selected_dataset_names), total=len(selected_dataset_names)):
            logger.info(f"Processing dataset: {ds_name} ({ds_num + 1}/{len(selected_dataset_names)})")
            terms = get_dataset_terms(ds_name, selected_term, medium_long_datasets)
            logger.info(f"Terms: {terms}")

            for term in tqdm(terms):
                ds_key, ds_freq = parse_dataset_key(ds_name, dataset_properties_map)
                ds_config = f"{ds_key}/{ds_freq}/{term}"

                # Determine if univariate conversion needed
                to_univariate = gift_eval_kwargs["to_univariate"]
                if not to_univariate:
                    to_univariate = (
                        GiftEvalDataset(
                            name=ds_name, term=term, to_univariate=False, data_dir=cfg.eval.data_dir
                        ).target_dim
                        != 1
                    )

                dataset = GiftEvalDataset(
                    name=ds_name, term=term, to_univariate=to_univariate, data_dir=cfg.eval.data_dir
                )
                logger.info(f"Dataset size: {len(dataset.test_data)}")

                # Create predictor based on model type
                if model_type == "timesfm":
                    predictor = TimesFmPredictor(
                        tfm=pipeline.model, prediction_length=dataset.prediction_length, rseed=cfg.eval.rseed
                    )
                elif model_type in ["chronos", "chronos_bolt"]:
                    predictor = ChronosPredictor(
                        pipeline=pipeline,
                        prediction_length=dataset.prediction_length,
                        num_samples=20,
                        rseed=cfg.eval.rseed,
                    )
                else:
                    raise ValueError(f"Invalid model type: {model_type}")

                # Evaluate model
                res = evaluate_model(
                    predictor,  # type: ignore[arg-type]
                    test_data=dataset.test_data,
                    metrics=metrics,
                    batch_size=1024,  # cfg.eval.batch_size
                    axis=None,
                    mask_invalid_label=True,
                    allow_nan_forecast=False,
                    seasonality=get_seasonality(dataset.freq),
                )

                # Build and write result row
                ds_props = dataset_properties_map[ds_key]
                row_data = [ds_config, model_type]
                for header in row_header[2:]:
                    if header.startswith("eval_metrics/"):
                        row_data.append(res[header.replace("eval_metrics/", "")][0])  # type: ignore[index]
                    elif header == "domain":
                        row_data.append(ds_props["domain"])
                    elif header == "num_variates":
                        row_data.append(ds_props["num_variates"])
                    elif header == "num_datasets":
                        row_data.append(len(dataset.test_data))  # type: ignore[arg-type]
                    else:
                        row_data.append(None)  # type: ignore[arg-type]

                with open(csv_file_path, "a", newline="") as csvfile:
                    csv.writer(csvfile).writerow(row_data)

                logger.info(f"Results written for {ds_name}/{term}")

    ##### 6. Display final results #####
    logger.info(f"\nFinal results: {csv_file_path}")
    logger.info(f"\n{pd.read_csv(csv_file_path)}")


if __name__ == "__main__":
    main()
