from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from itertools import islice
from pathlib import Path
from typing import Any

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import torch

from tsfm_lens.chronos.circuitlens import CircuitLensChronos
from tsfm_lens.chronos2.circuitlens import CircuitLensChronos2
from tsfm_lens.chronos_bolt.circuitlens import CircuitLensBolt
from tsfm_lens.dataset import GiftEvalDataset
from tsfm_lens.moirai.circuitlens import CircuitLensMoirai
from tsfm_lens.timesfm.circuitlens import CircuitLensTimesFM
from tsfm_lens.toto.circuitlens import CircuitLensToto

WORK_DIR = os.environ.get("WORK", "/work")
DATA_DIR = os.path.join(WORK_DIR, "data", "gift-eval")


def list_dataset_names(data_dir: str) -> list[str]:
    path = Path(data_dir)
    if not path.exists():
        return []
    return sorted([child.name for child in path.iterdir() if child.is_dir() and child.name != ".cache"])


def list_dataset_frequencies(data_dir: str, dataset_name: str | None) -> list[str]:
    if not dataset_name:
        return []

    dataset_path = Path(data_dir) / dataset_name
    if not dataset_path.exists() or not dataset_path.is_dir():
        return []

    frequencies = [child.name for child in dataset_path.iterdir() if child.is_dir() and child.name != ".cache"]
    return sorted(frequencies)


def resolve_dataset_name(dataset_name: str, frequency: str | None) -> str:
    return f"{dataset_name}/{frequency}" if frequency else dataset_name


DEFAULT_DATASETS = list_dataset_names(DATA_DIR)
DEFAULT_DATASET = DEFAULT_DATASETS[0] if DEFAULT_DATASETS else None
DEFAULT_FREQUENCIES = list_dataset_frequencies(DATA_DIR, DEFAULT_DATASET)
DEFAULT_FREQUENCY = DEFAULT_FREQUENCIES[0] if DEFAULT_FREQUENCIES else None


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    model_id: str
    default_context_length: int
    loader: Any


MODEL_SPECS: dict[str, ModelSpec] = {
    "chronos": ModelSpec(
        key="chronos",
        label="Chronos",
        model_id="amazon/chronos-t5-base",
        default_context_length=512,
        loader=lambda device: CircuitLensChronos.from_pretrained(
            "amazon/chronos-t5-base",
            device_map=device,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        ),
    ),
    "chronos-bolt": ModelSpec(
        key="chronos-bolt",
        label="Chronos-Bolt",
        model_id="amazon/chronos-bolt-base",
        default_context_length=512,
        loader=lambda device: CircuitLensBolt.from_pretrained(
            "amazon/chronos-bolt-base",
            device_map=device,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        ),
    ),
    "chronos2": ModelSpec(
        key="chronos2",
        label="Chronos-2",
        model_id="amazon/chronos-2",
        default_context_length=8192,
        loader=lambda device: CircuitLensChronos2.from_pretrained("amazon/chronos-2", device=device),
    ),
    "timesfm": ModelSpec(
        key="timesfm",
        label="TimesFM 2.5",
        model_id="google/timesfm-2.5-200m-pytorch",
        default_context_length=2048,
        loader=lambda device: CircuitLensTimesFM("google/timesfm-2.5-200m-pytorch", device_map=device),
    ),
    "toto": ModelSpec(
        key="toto",
        label="Toto",
        model_id="Datadog/Toto-Open-Base-1.0",
        default_context_length=4096,
        loader=lambda device: CircuitLensToto("Datadog/Toto-Open-Base-1.0", device_map=device),
    ),
    "moirai": ModelSpec(
        key="moirai",
        label="Moirai 1.1 R Base",
        model_id="Salesforce/moirai-1.1-R-base",
        default_context_length=4000,
        loader=lambda device: CircuitLensMoirai(
            model_name="Salesforce/moirai-1.1-R-base",
            context_length=4000,
            prediction_length=1,
            patch_size=32,
            num_samples=100,
            target_dim=1,
            device=device,
        ),
    ),
}


PIPELINE_CACHE: dict[tuple[str, str], Any] = {}


@dataclass
class SampleBundle:
    dataset_name: str
    term: str
    item_id: str
    freq: str
    prediction_length: int
    total_samples: int
    available_dims: int
    selected_dims: list[int]
    original_context_length: int
    used_context_length: int
    context: torch.Tensor
    future: torch.Tensor


def get_default_device() -> str:
    requested = os.environ.get("TSFM_LENS_DEVICE")
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def parse_index_spec(spec: str, upper_bound: int | None = None) -> list[int]:
    cleaned = (spec or "").strip()
    if not cleaned:
        return []

    values: set[int] = set()
    for raw_chunk in cleaned.replace("\n", ",").split(","):
        chunk = raw_chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start_str, end_str = [part.strip() for part in chunk.split("-", maxsplit=1)]
            start = int(start_str)
            end = int(end_str)
            if end < start:
                raise ValueError(f"Invalid range '{chunk}'.")
            values.update(range(start, end + 1))
        else:
            values.add(int(chunk))

    ordered = sorted(values)
    if upper_bound is not None:
        invalid = [value for value in ordered if value < 0 or value >= upper_bound]
        if invalid:
            raise ValueError(f"Indices out of range: {invalid}. Valid range is [0, {upper_bound - 1}].")
    return ordered


def parse_head_spec(spec: str, num_layers: int, num_heads: int) -> list[tuple[int, int]]:
    cleaned = (spec or "").strip()
    if not cleaned:
        return []

    heads_to_ablate: set[tuple[int, int]] = set()
    for raw_line in cleaned.replace(";", "\n").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if ":" not in line:
            raise ValueError("Head ablations must use 'layer: heads' syntax, for example '3: 0,1,2' or '7: all'.")

        layer_part, head_part = [part.strip() for part in line.split(":", maxsplit=1)]
        layer_idx = int(layer_part)
        if layer_idx < 0 or layer_idx >= num_layers:
            raise ValueError(f"Layer {layer_idx} is out of range. Valid layers are [0, {num_layers - 1}].")

        if head_part.lower() == "all":
            selected_heads = list(range(num_heads))
        else:
            selected_heads = parse_index_spec(head_part, upper_bound=num_heads)

        heads_to_ablate.update((layer_idx, head_idx) for head_idx in selected_heads)

    return sorted(heads_to_ablate)


def summarize_heads(heads_to_ablate: list[tuple[int, int]]) -> str:
    if not heads_to_ablate:
        return "None"
    grouped: dict[int, list[int]] = {}
    for layer_idx, head_idx in heads_to_ablate:
        grouped.setdefault(layer_idx, []).append(head_idx)
    parts = [f"L{layer_idx}: {grouped[layer_idx]}" for layer_idx in sorted(grouped)]
    return "; ".join(parts)


@lru_cache(maxsize=64)
def get_dataset_summary(data_dir: str, dataset_name: str, frequency: str | None, term: str) -> dict[str, Any]:
    resolved_name = resolve_dataset_name(dataset_name, frequency)
    dataset = GiftEvalDataset(name=resolved_name, term=term, to_univariate=False, data_dir=data_dir)
    return {
        "resolved_name": resolved_name,
        "prediction_length": dataset.prediction_length,
        "target_dim": dataset.target_dim,
        "windows": dataset.windows,
        "rows": dataset.hf_dataset.num_rows,
        "total_samples": dataset.hf_dataset.num_rows * dataset.windows,  # type: ignore
        "freq": dataset.freq,
    }


def load_sample(
    data_dir: str,
    dataset_name: str,
    frequency: str | None,
    term: str,
    sample_index: int,
    context_length: int,
    dims_spec: str,
    max_context_length: int | None = None,
) -> SampleBundle:
    resolved_name = resolve_dataset_name(dataset_name, frequency)
    dataset = GiftEvalDataset(name=resolved_name, term=term, to_univariate=False, data_dir=data_dir)
    total_samples = dataset.hf_dataset.num_rows * dataset.windows  # type: ignore
    if total_samples == 0:
        raise ValueError(f"No test windows found for dataset '{resolved_name}'.")

    if sample_index < 0 or sample_index >= total_samples:
        raise ValueError(f"Sample index {sample_index} is out of range. Valid range is [0, {total_samples - 1}].")

    context_entry = next(islice(dataset.test_data.input, sample_index, None))
    label_entry = next(islice(dataset.test_data.label, sample_index, None))

    context_target = np.asarray(context_entry["target"], dtype=np.float32)
    future_target = np.asarray(label_entry["target"], dtype=np.float32)

    if context_target.ndim == 1:
        context_target = context_target[None, :]
        future_target = future_target[None, :]

    selected_dims = parse_index_spec(dims_spec, upper_bound=context_target.shape[0]) if dims_spec.strip() else [0]
    if not selected_dims:
        raise ValueError("Select at least one dimension.")

    context_target = context_target[selected_dims]
    future_target = future_target[selected_dims]

    original_context_length = int(context_target.shape[-1])
    context_cap = (
        original_context_length if max_context_length is None else min(original_context_length, max_context_length)
    )
    used_context_length = min(int(context_length), context_cap)
    if used_context_length <= 0:
        raise ValueError("Context length must be positive.")

    context_target = context_target[:, -used_context_length:]

    return SampleBundle(
        dataset_name=dataset_name,
        term=term,
        item_id=str(context_entry.get("item_id", f"{dataset_name}:{sample_index}")),
        freq=str(context_entry.get("freq", dataset.freq)),
        prediction_length=int(dataset.prediction_length),
        total_samples=total_samples,
        available_dims=int(dataset.target_dim),
        selected_dims=selected_dims,
        original_context_length=original_context_length,
        used_context_length=used_context_length,
        context=torch.tensor(context_target, dtype=torch.float32),
        future=torch.tensor(future_target, dtype=torch.float32),
    )


def get_pipeline(model_key: str, device: str) -> Any:
    cache_key = (model_key, device)
    if cache_key not in PIPELINE_CACHE:
        PIPELINE_CACHE[cache_key] = MODEL_SPECS[model_key].loader(device)
        PIPELINE_CACHE[cache_key].set_to_eval_mode()
    return PIPELINE_CACHE[cache_key]


def normalize_point_forecast(model_key: str, raw_prediction: Any) -> np.ndarray:
    if isinstance(raw_prediction, tuple):
        raw_prediction = raw_prediction[0]

    if not isinstance(raw_prediction, torch.Tensor):
        raw_prediction = torch.as_tensor(raw_prediction)

    prediction = raw_prediction.detach().float().cpu()

    if model_key in {"chronos", "chronos-bolt"} and prediction.ndim == 3:
        prediction = prediction.median(dim=1).values
    elif model_key == "toto" and prediction.ndim == 3:
        prediction = prediction.median(dim=1).values
    elif model_key == "moirai" and prediction.ndim == 3:
        prediction = prediction.median(dim=1).values

    if prediction.ndim == 1:
        prediction = prediction.unsqueeze(0)
    elif prediction.ndim > 2:
        raise ValueError(f"Unexpected prediction shape after normalization: {tuple(prediction.shape)}")

    return prediction.numpy()


def run_forecast(
    pipeline: Any,
    model_key: str,
    context: torch.Tensor,
    prediction_length: int,
    num_samples: int,
) -> np.ndarray:
    with torch.no_grad():
        if model_key == "chronos":
            raw_prediction = pipeline.predict(
                context,
                prediction_length=prediction_length,
                num_samples=num_samples,
                limit_prediction_length=False,
            )
        elif model_key == "chronos-bolt":
            raw_prediction = pipeline.predict(
                context,
                prediction_length=prediction_length,
                limit_prediction_length=False,
            )
        elif model_key == "chronos2":
            raw_prediction = pipeline.predict(
                context=context,
                prediction_length=prediction_length,
                limit_prediction_length=False,
            )
        elif model_key == "timesfm":
            raw_prediction = pipeline.predict(context=context, prediction_length=prediction_length)
        elif model_key == "toto":
            raw_prediction = pipeline.predict(
                context=context,
                prediction_length=prediction_length,
                num_samples=num_samples,
                samples_per_batch=max(1, num_samples),
            )
        elif model_key == "moirai":
            raw_prediction = pipeline.predict(
                context=context,
                prediction_length=prediction_length,
                num_samples=num_samples,
            )
        else:
            raise ValueError(f"Unsupported model key: {model_key}")

    return normalize_point_forecast(model_key, raw_prediction)


def build_plot(sample: SampleBundle, baseline_pred: np.ndarray, ablated_pred: np.ndarray):
    num_dims = len(sample.selected_dims)
    fig, axes = plt.subplots(num_dims, 1, figsize=(12, max(4, 3.0 * num_dims)), squeeze=False)
    context_np = sample.context.numpy()
    future_np = sample.future.numpy()

    for row_idx, dim_idx in enumerate(sample.selected_dims):
        ax = axes[row_idx, 0]
        context_series = context_np[row_idx]
        future_series = future_np[row_idx]
        baseline_series = baseline_pred[row_idx]
        ablated_series = ablated_pred[row_idx]

        context_x = np.arange(len(context_series))
        future_x = np.arange(len(context_series), len(context_series) + len(future_series))

        ax.plot(context_x, context_series, color="black", linewidth=1.2, label="Context")
        ax.plot(future_x, future_series, color="#2f6fed", linewidth=1.5, label="Ground truth")
        ax.plot(future_x, baseline_series, color="#15a34a", linewidth=1.5, label="Baseline")
        ax.plot(future_x, ablated_series, color="#dc2626", linewidth=1.5, label="Ablated")
        ax.axvline(len(context_series) - 1, color="#6b7280", linestyle="--", linewidth=1)
        ax.set_title(f"Dimension {dim_idx}")
        ax.grid(alpha=0.25)
        ax.legend(loc="best")

    fig.tight_layout()
    return fig


def build_summary(
    sample: SampleBundle,
    model_spec: ModelSpec,
    pipeline: Any,
    device: str,
    heads_to_ablate: list[tuple[int, int]],
    mlp_layers: list[int],
    baseline_pred: np.ndarray,
    ablated_pred: np.ndarray,
) -> str:
    future_np = sample.future.numpy()
    baseline_mae = float(np.nanmean(np.abs(baseline_pred - future_np)))
    ablated_mae = float(np.nanmean(np.abs(ablated_pred - future_np)))
    delta_mae = ablated_mae - baseline_mae

    return "\n".join(
        [
            "### Run Summary",
            f"- Model: `{model_spec.label}` on `{device}`",
            f"- Dataset: `{sample.dataset_name}` (`{sample.term}`), item `{sample.item_id}`, freq `{sample.freq}`",
            f"- Sample index space: `{sample.total_samples}` total windows",
            f"- Dimensions: selected `{sample.selected_dims}` out of `{sample.available_dims}`",
            f"- Context length: using `{sample.used_context_length}` of `{sample.original_context_length}` available points",
            f"- Prediction length: `{sample.prediction_length}`",
            f"- Model depth: `{pipeline.num_layers}` layers, `{pipeline.num_heads}` heads",
            f"- Head ablations: `{len(heads_to_ablate)}` heads -> {summarize_heads(heads_to_ablate)}",
            f"- MLP ablations: `{mlp_layers or 'None'}`",
            f"- Baseline MAE: `{baseline_mae:.6f}`",
            f"- Ablated MAE: `{ablated_mae:.6f}`",
            f"- Delta MAE: `{delta_mae:+.6f}`",
        ]
    )


def describe_dataset(data_dir: str, dataset_name: str | None, term: str) -> str:
    if not data_dir:
        return "Set a Gift-Eval directory."
    path = Path(data_dir)
    if not path.exists():
        return f"Gift-Eval directory not found: `{data_dir}`"
    if not dataset_name:
        names = list_dataset_names(data_dir)
        if not names:
            return f"No dataset folders found under `{data_dir}`."
        return f"Found `{len(names)}` datasets under `{data_dir}`."

    frequencies = list_dataset_frequencies(data_dir, dataset_name)
    if frequencies:
        frequency = frequencies[0]
    else:
        frequency = None

    return describe_dataset_frequency(data_dir, dataset_name, frequency, term)


def describe_dataset_frequency(data_dir: str, dataset_name: str | None, frequency: str | None, term: str) -> str:
    if not data_dir:
        return "Set a Gift-Eval directory."
    path = Path(data_dir)
    if not path.exists():
        return f"Gift-Eval directory not found: `{data_dir}`"
    if not dataset_name:
        names = list_dataset_names(data_dir)
        if not names:
            return f"No dataset folders found under `{data_dir}`."
        return f"Found `{len(names)}` datasets under `{data_dir}`."

    try:
        summary = get_dataset_summary(data_dir, dataset_name, frequency, term)
    except Exception as exc:
        resolved_name = resolve_dataset_name(dataset_name, frequency)
        return f"Could not inspect dataset `{resolved_name}` in `{data_dir}`: `{exc}`"
    return "\n".join(
        [
            "### Dataset Summary",
            f"- Dataset: `{dataset_name}`",
            f"- Frequency: `{frequency or summary['freq']}`",
            f"- Resolved name: `{summary['resolved_name']}`",
            f"- Term: `{term}`",
            f"- Prediction length: `{summary['prediction_length']}`",
            f"- Target dimension: `{summary['target_dim']}`",
            f"- Rows: `{summary['rows']}`",
            f"- Test windows: `{summary['windows']}` per row, `{summary['total_samples']}` total",
        ]
    )


def refresh_dataset_dropdown(data_dir: str, term: str):
    names = list_dataset_names(data_dir)
    dataset_value = names[0] if names else None
    frequencies = list_dataset_frequencies(data_dir, dataset_value)
    frequency_value = frequencies[0] if frequencies else None
    description = describe_dataset_frequency(data_dir, dataset_value, frequency_value, term)
    return (
        gr.update(choices=names, value=dataset_value),
        gr.update(choices=frequencies, value=frequency_value),
        description,
    )


def refresh_frequency_dropdown(data_dir: str, dataset_name: str, term: str):
    frequencies = list_dataset_frequencies(data_dir, dataset_name)
    frequency_value = frequencies[0] if frequencies else None
    description = describe_dataset_frequency(data_dir, dataset_name, frequency_value, term)
    return gr.update(choices=frequencies, value=frequency_value), description


def update_context_length(model_key: str):
    return MODEL_SPECS[model_key].default_context_length


def run_demo(
    model_key: str,
    data_dir: str,
    dataset_name: str,
    frequency: str | None,
    term: str,
    sample_index: float,
    context_length: float,
    dims_spec: str,
    head_spec: str,
    mlp_spec: str,
    num_samples: float,
):
    if not dataset_name:
        raise gr.Error("Choose a dataset first.")

    device = get_default_device()
    model_spec = MODEL_SPECS[model_key]
    pipeline = get_pipeline(model_key, device)

    sample = load_sample(
        data_dir=data_dir,
        dataset_name=dataset_name,
        frequency=frequency,
        term=term,
        sample_index=int(sample_index),
        context_length=int(context_length),
        dims_spec=dims_spec,
        max_context_length=model_spec.default_context_length,
    )

    try:
        heads_to_ablate = parse_head_spec(head_spec, num_layers=pipeline.num_layers, num_heads=pipeline.num_heads)
        mlp_layers = parse_index_spec(mlp_spec, upper_bound=pipeline.num_layers)
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc

    try:
        pipeline.remove_all_hooks()
        baseline_pred = run_forecast(
            pipeline=pipeline,
            model_key=model_key,
            context=sample.context,
            prediction_length=sample.prediction_length,
            num_samples=max(1, int(num_samples)),
        )

        if heads_to_ablate or mlp_layers:
            ablation_types: list[str] = []
            if heads_to_ablate:
                ablation_types.append("head")
            if mlp_layers:
                ablation_types.append("mlp")
            pipeline.add_ablation_hooks_explicit(
                ablations_types=ablation_types,
                layers_to_ablate_mlp=mlp_layers,
                heads_to_ablate=heads_to_ablate,
            )

        ablated_pred = run_forecast(
            pipeline=pipeline,
            model_key=model_key,
            context=sample.context,
            prediction_length=sample.prediction_length,
            num_samples=max(1, int(num_samples)),
        )
    finally:
        pipeline.remove_all_hooks()

    figure = build_plot(sample, baseline_pred=baseline_pred, ablated_pred=ablated_pred)
    summary = build_summary(
        sample=sample,
        model_spec=model_spec,
        pipeline=pipeline,
        device=device,
        heads_to_ablate=heads_to_ablate,
        mlp_layers=mlp_layers,
        baseline_pred=baseline_pred,
        ablated_pred=ablated_pred,
    )
    return figure, summary


with gr.Blocks(title="TSFM Lens Ablation Demo") as demo:
    gr.Markdown(
        "\n".join(
            [
                "# TSFM Lens Ablation Demo",
                "Compare baseline and ablated TSFM forecasts on Gift-Eval windows.",
                "Use `layer: heads` lines like `3: 0,1,2` or `7: all`. Use comma or range syntax for dimensions and MLP layers.",
            ]
        )
    )

    with gr.Row():
        with gr.Column(scale=1):
            model_key = gr.Dropdown(
                choices=[(spec.label, spec.key) for spec in MODEL_SPECS.values()],
                value="chronos",
                label="Model",
            )
            data_dir = gr.Textbox(value=DATA_DIR, label="Gift-Eval directory")
            refresh_button = gr.Button("Refresh dataset list")
            dataset_name = gr.Dropdown(
                choices=DEFAULT_DATASETS,
                value=DEFAULT_DATASET,
                label="Dataset",
            )
            frequency = gr.Dropdown(
                choices=DEFAULT_FREQUENCIES,
                value=DEFAULT_FREQUENCY,
                label="Frequency",
            )
            term = gr.Dropdown(choices=["short", "medium", "long"], value="short", label="Gift-Eval term")
            sample_index = gr.Number(value=0, precision=0, label="Sample index")
            dims_spec = gr.Textbox(value="0", label="Dimensions", info="Examples: 0 or 0,1,2 or 0-3")
            context_length = gr.Number(
                value=MODEL_SPECS["chronos"].default_context_length,
                precision=0,
                label="Context length",
            )
            num_samples = gr.Number(
                value=20,
                precision=0,
                label="Forecast samples",
                info="Used by probabilistic models before collapsing to a point forecast.",
            )

        with gr.Column(scale=1):
            head_spec = gr.Textbox(
                value="",
                lines=8,
                label="Attention head ablations",
                info="One layer per line, for example: 3: 0,1,2",
            )
            mlp_spec = gr.Textbox(
                value="",
                lines=2,
                label="MLP layer ablations",
                info="Examples: 3,5,7 or 3-7",
            )
            dataset_summary = gr.Markdown(
                value=describe_dataset_frequency(DATA_DIR, DEFAULT_DATASET, DEFAULT_FREQUENCY, "short")
                if DEFAULT_DATASET
                else "No datasets discovered yet."
            )
            run_button = gr.Button("Run ablation demo", variant="primary")

    plot_output = gr.Plot(label="Forecast comparison")
    summary_output = gr.Markdown()

    refresh_button.click(
        fn=refresh_dataset_dropdown,
        inputs=[data_dir, term],
        outputs=[dataset_name, frequency, dataset_summary],
    )
    data_dir.submit(
        fn=refresh_dataset_dropdown,
        inputs=[data_dir, term],
        outputs=[dataset_name, frequency, dataset_summary],
    )
    dataset_name.change(
        fn=refresh_frequency_dropdown,
        inputs=[data_dir, dataset_name, term],
        outputs=[frequency, dataset_summary],
    )
    frequency.change(
        fn=describe_dataset_frequency,
        inputs=[data_dir, dataset_name, frequency, term],
        outputs=[dataset_summary],
    )
    term.change(
        fn=describe_dataset_frequency,
        inputs=[data_dir, dataset_name, frequency, term],
        outputs=[dataset_summary],
    )
    model_key.change(fn=update_context_length, inputs=[model_key], outputs=[context_length])

    run_button.click(
        fn=run_demo,
        inputs=[
            model_key,
            data_dir,
            dataset_name,
            frequency,
            term,
            sample_index,
            context_length,
            dims_spec,
            head_spec,
            mlp_spec,
            num_samples,
        ],
        outputs=[plot_output, summary_output],
    )


if __name__ == "__main__":
    demo.launch()
