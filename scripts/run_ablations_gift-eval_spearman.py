"""
Compute per-context-window Spearman correlations between original and ablated TSFM predictions
on GIFT-Eval datasets. Outputs one correlation value per context window (no aggregation).
"""

import csv
import json
import logging
import os

import hydra
import numpy as np
import torch
from chronos import ForecastType
from gluonts.itertools import batcher
from omegaconf import DictConfig
from timesfm import configs
from tqdm.auto import tqdm

from tsfm_lens.chronos.circuitlens import CircuitLensChronos
from tsfm_lens.chronos_bolt.circuitlens import CircuitLensBolt
from tsfm_lens.circuitlens import BaseCircuitLens
from tsfm_lens.dataset import GiftEvalDataset
from tsfm_lens.metrics import _spearman_batched
from tsfm_lens.moirai.circuitlens import CircuitLensMoirai
from tsfm_lens.timesfm.circuitlens import CircuitLensTimesFM
from tsfm_lens.toto.circuitlens import CircuitLensToto
from tsfm_lens.utils import clear_cuda_cache, set_seed

ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets")
DATASET_PROPERTIES_PATH = os.path.join(ASSETS_DIR, "dataset_properties.json")
DEFAULT_RESULTS_SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")

logger = logging.getLogger(__name__)

PRETTY_NAMES = {
    "saugeenday": "saugeen",
    "temperature_rain_with_missing": "temperature_rain",
    "kdd_cup_2018_with_missing": "kdd_cup_2018",
    "car_parts_with_missing": "car_parts",
}


def load_model(model_type: str, cfg: DictConfig, device: torch.device) -> BaseCircuitLens:
    """Load the appropriate CircuitLens model pipeline."""
    torch_dtype = getattr(torch, cfg.eval.torch_dtype) or torch.float32

    loaders = {
        "timesfm": lambda: CircuitLensTimesFM(cfg.timesfm.model_id, device_map=device),
        "chronos": lambda: CircuitLensChronos.from_pretrained(
            cfg.chronos.model_id, device_map=device, torch_dtype=torch_dtype
        ),
        "chronos_bolt": lambda: CircuitLensBolt.from_pretrained(
            cfg.chronos_bolt.model_id, device_map=device, torch_dtype=torch_dtype
        ),
        "toto": lambda: CircuitLensToto(cfg.toto.model_id, device_map=device),
        "moirai": lambda: CircuitLensMoirai(
            cfg.moirai.model_id,
            context_length=4000,
            prediction_length=1,  # Placeholder, overridden per dataset
            patch_size=cfg.moirai.patch_size,
            num_samples=cfg.moirai.num_samples,
            target_dim=1,
            device=device,
        ),
    }
    if model_type not in loaders:
        raise ValueError(f"Unsupported model type: {model_type}. Use: {list(loaders.keys())}")

    pipeline = loaders[model_type]()
    pipeline.set_to_eval_mode()
    logger.info(f"Loaded {model_type}: {pipeline.num_layers} layers, {pipeline.num_heads} heads")
    return pipeline


def setup_ablations(
    pipeline: BaseCircuitLens,
    layers: list[int],
    n_heads_per_layer: int | None,
    strategy: str,
    rseed: int,
) -> list[tuple[int, int]]:
    """Configure ablation hooks and return list of (layer, head) tuples being ablated."""
    pipeline.remove_all_hooks()
    if hasattr(pipeline, "reset_attribution_inputs_and_outputs"):
        pipeline.reset_attribution_inputs_and_outputs()

    # Select heads to ablate
    if strategy == "random":
        rng = np.random.default_rng(rseed)
        if n_heads_per_layer is None:
            heads = [(layer, h) for layer in layers for h in range(pipeline.num_heads)]
        else:
            heads = []
            for layer in layers:
                selected = rng.choice(pipeline.num_heads, size=n_heads_per_layer, replace=False)
                heads.extend((layer, int(h)) for h in selected)
    elif strategy in ["srank", "srank_reverse"]:
        ranking_path = os.path.join(ASSETS_DIR, f"{pipeline.__class__.__name__}_srank_low_to_high_by_layer.json")
        with open(ranking_path) as f:
            rankings = json.load(f)
        reverse = strategy == "srank_reverse"
        heads = []
        for layer in layers:
            head_list = list(rankings[str(layer)].keys())
            if reverse:
                head_list = head_list[::-1]
            heads.extend((layer, int(h)) for h in head_list[:n_heads_per_layer])
    elif strategy == "all":
        heads = [(layer, h) for layer in layers for h in range(pipeline.num_heads)]
    else:
        raise ValueError(f"Unknown strategy: {strategy}. Use: random, srank, srank_reverse, all")

    pipeline.add_ablation_hooks_explicit(
        ablations_types=["head"],
        layers_to_ablate_mlp=[],
        heads_to_ablate=heads,
    )
    logger.info(f"Ablating {len(heads)} heads across {len(layers)} layers")
    return heads


def get_predictions(
    pipeline: BaseCircuitLens,
    model_type: str,
    test_data: list[dict],
    prediction_length: int,
    cfg: DictConfig,
    batch_size: int = 512,
) -> np.ndarray:
    """Get predictions from model. Returns shape (n_windows, prediction_length)."""
    outputs = []

    for batch in batcher(test_data, batch_size=batch_size):
        set_seed(cfg.eval.rseed)

        if model_type == "timesfm":
            tfm = pipeline.model  # type: ignore
            context = [np.array(e["target"]) for e in batch]
            max_ctx = max(c.shape[0] for c in context)
            patch_size = tfm.model.p  # type: ignore
            max_ctx = ((max_ctx + patch_size - 1) // patch_size) * patch_size
            tfm.compile(
                configs.ForecastConfig(  # type: ignore
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
            preds, _ = tfm.forecast(horizon=prediction_length, inputs=context)  # type: ignore
            outputs.append(preds[:, :prediction_length])

        elif model_type in ["chronos", "chronos_bolt"]:
            context = [torch.tensor(e["target"]) for e in batch]
            kwargs: dict = {"limit_prediction_length": False}
            if pipeline.forecast_type == ForecastType.SAMPLES:  # type: ignore
                kwargs["num_samples"] = cfg.chronos.num_samples
            preds = pipeline.predict(context, prediction_length=prediction_length, **kwargs).numpy()  # type: ignore
            if preds.ndim == 3:
                preds = np.median(preds, axis=1)  # Reduce samples/quantiles to median
            outputs.append(preds)

        elif model_type == "toto":
            # Toto's predict() treats dim 0 as variates, not batch - must loop over series
            batch_preds = []
            for entry in batch:
                ctx = torch.tensor(np.array(entry["target"])).float().unsqueeze(0)  # (1, time)
                pred = pipeline.predict(  # type: ignore
                    ctx,
                    prediction_length=prediction_length,
                    num_samples=cfg.toto.samples_per_batch,
                    samples_per_batch=cfg.toto.samples_per_batch,
                    use_kv_cache=cfg.toto.use_kv_cache,
                )
                # pred shape: (1, num_samples, pred_len) -> take median -> (pred_len,)
                if isinstance(pred, torch.Tensor):
                    pred = pred.cpu().numpy()
                batch_preds.append(np.median(pred, axis=1).squeeze())
            outputs.append(np.stack(batch_preds, axis=0))  # (batch, pred_len)

        elif model_type == "moirai":
            # Stack contexts into tensor (batch, time)
            contexts = [np.array(e["target"]) for e in batch]
            max_len = max(c.shape[0] for c in contexts)
            padded = np.zeros((len(contexts), max_len), dtype=np.float32)
            for i, c in enumerate(contexts):
                padded[i, -len(c) :] = c  # Right-align
            context = torch.tensor(padded).float()
            preds = pipeline.predict(context, prediction_length=prediction_length)  # type: ignore
            if isinstance(preds, torch.Tensor):
                preds = preds.cpu().numpy()
            if preds.ndim == 3:
                preds = np.median(preds, axis=1)  # Reduce samples to median
            outputs.append(preds)

    return np.concatenate(outputs, axis=0)


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    device = torch.device(cfg.eval.device if torch.cuda.is_available() else "cpu")
    clear_cuda_cache(device)

    # Load dataset properties
    with open(DATASET_PROPERTIES_PATH) as f:
        ds_props = json.load(f)

    # Dataset selection
    gift_cfg = cfg.eval.gift_eval
    datasets = gift_cfg.dataset_names or (gift_cfg.short_datasets + gift_cfg.medium_long_datasets)
    datasets = [datasets] if isinstance(datasets, str) else list(set(datasets))
    if gift_cfg.max_num_datasets:
        datasets = datasets[: gift_cfg.max_num_datasets]

    # Load model
    model_type = cfg.ablation.model_type
    pipeline = load_model(model_type, cfg, device)

    # Setup output
    layers = cfg.ablation.ablations_layers_lst or list(range(pipeline.num_layers))
    n_heads = cfg.ablation.ablate_n_heads_per_layer
    strategy = cfg.ablation.head_selection_strategy or "all"
    ablation_name = f"{strategy}_L{'-'.join(map(str, layers))}_H{n_heads or 'all'}"

    output_dir = os.path.join(
        cfg.eval.results_save_dir or DEFAULT_RESULTS_SAVE_DIR,
        cfg.ablation.model_name_str,
        f"spearman_rseed{cfg.eval.rseed}",
    )
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"{ablation_name}_{gift_cfg.term}.csv")

    # Write header
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(["dataset", "term", "window_idx", "spearman"])

    # Process each dataset
    for ds_name in tqdm(datasets, desc="Datasets"):
        # Parse dataset info
        if "/" in ds_name:
            ds_key, ds_freq = ds_name.split("/")[:2]
        else:
            ds_key = ds_name
            ds_freq = ds_props.get(ds_key.lower(), {}).get("frequency", "")
        ds_key = PRETTY_NAMES.get(ds_key.lower(), ds_key.lower())

        # Determine terms to evaluate
        terms = ["short"]
        if ds_name in gift_cfg.medium_long_datasets and gift_cfg.term in ["medium", "long", "all"]:
            terms = ["medium", "long"] if gift_cfg.term != "short" else terms

        for term in terms:
            dataset = GiftEvalDataset(
                name=ds_name, term=term, to_univariate=gift_cfg.to_univariate, data_dir=cfg.eval.data_dir
            )
            test_data = list(dataset.test_data.input)
            logger.info(f"{ds_name}/{term}: {len(test_data)} windows")

            # Original predictions (no ablations)
            pipeline.remove_all_hooks()
            original = get_predictions(pipeline, model_type, test_data, dataset.prediction_length, cfg)

            # Ablated predictions
            setup_ablations(pipeline, layers, n_heads, strategy, cfg.eval.rseed)
            ablated = get_predictions(pipeline, model_type, test_data, dataset.prediction_length, cfg)

            # Compute per-window Spearman
            spearman = _spearman_batched(original, ablated)

            # Write results
            rows = [[f"{ds_key}/{ds_freq}", term, i, s] for i, s in enumerate(spearman)]
            with open(csv_path, "a", newline="") as f:
                csv.writer(f).writerows(rows)

            logger.info(f"  Mean Spearman: {spearman.mean():.4f}")

    logger.info(f"Results saved to: {csv_path}")


if __name__ == "__main__":
    main()
