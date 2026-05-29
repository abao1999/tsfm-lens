"""Unit tests for `scripts/entropic_rank_head_outputs.py`.

Covered: `_ensure_shape`, `compute_entropic_rank_per_layer`, the four
dispatch helpers (`add_head_hooks`, `extract_head_outputs`,
`reset_pipeline_outputs`, `run_inference_for_model`).
No GPU, no model downloads, no real Hydra runtime, no real data.

`instantiate_pipeline`, `prepare_test_datasets`, `process_model`, and
`main()` are out of scope — they need real models / Hydra / data files.

Run with:
.venv/bin/pytest tests/test_entropic_rank_head_outputs.py -v
"""

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from scripts.entropic_rank_head_outputs import (
    _ensure_shape,
    add_head_hooks,
    compute_entropic_rank_per_layer,
    extract_head_outputs,
    reset_pipeline_outputs,
    run_inference_for_model,
)


# =============================================================================
# Fixtures
# =============================================================================
@pytest.fixture
def cfg():
    """Synthetic Hydra DictConfig with the keys `run_inference_for_model` reads."""
    return OmegaConf.create(
        {
            "entropic_rank": {
                "prediction_length": 7,
                "num_samples": 5,
                "deterministic": False,
            },
            "eval": {"batch_size": 13},
            "toto": {"samples_per_batch": 4, "use_kv_cache": True},
            "moirai": {"patch_size": 32},
        }
    )


class RecordingPipeline:
    """Mock with method recorders for hook setup, inference, and cleanup.

    Output attributes (`head_attribution_outputs`, `ca_head_attribution_outputs`,
    `sa_head_attribution_outputs`) are set per-test by writing to the instance.
    """

    def __init__(self):
        self.hook_calls: list[tuple[tuple, dict]] = []
        self.predict_calls: list[tuple[tuple, dict]] = []
        self.predict_full_calls: list[tuple[tuple, dict]] = []
        self.remove_all_hooks_count = 0
        self.reset_attribution_count = 0
        self.reset_custom_count = 0
        self.head_attribution_outputs: dict = {}
        self.ca_head_attribution_outputs: dict = {}
        self.sa_head_attribution_outputs: dict = {}

    def add_head_attribution_hooks(self, *args, **kwargs):
        self.hook_calls.append((args, kwargs))

    def predict(self, *args, **kwargs):
        self.predict_calls.append((args, kwargs))

    def predict_with_full_outputs(self, *args, **kwargs):
        self.predict_full_calls.append((args, kwargs))

    def remove_all_hooks(self):
        self.remove_all_hooks_count += 1

    def reset_attribution_inputs_and_outputs(self):
        self.reset_attribution_count += 1

    def reset_custom_attribution_outputs(self):
        self.reset_custom_count += 1


@pytest.fixture
def pipeline():
    return RecordingPipeline()


# =============================================================================
# Section A — `_ensure_shape`
# =============================================================================
def test_ensure_shape_2d_unsqueezes():
    out = _ensure_shape(torch.randn(3, 4))
    assert out.shape == (1, 3, 4)
    assert out.dtype == torch.float32


def test_ensure_shape_3d_passthrough():
    inp = torch.randn(2, 3, 4)
    out = _ensure_shape(inp)
    assert out.shape == (2, 3, 4)
    assert out.is_contiguous()


def test_ensure_shape_4d_collapses_leading():
    out = _ensure_shape(torch.randn(2, 5, 3, 4))
    assert out.shape == (10, 3, 4)


def test_ensure_shape_5d_collapses_leading():
    out = _ensure_shape(torch.randn(2, 3, 5, 7, 4))
    assert out.shape == (2 * 3 * 5, 7, 4)


def test_ensure_shape_6d_raises():
    with pytest.raises(ValueError, match="Unsupported tensor shape"):
        _ensure_shape(torch.randn(2, 2, 2, 2, 2, 2))


def test_ensure_shape_scrubs_nonfinite():
    inp = torch.tensor([[1.0, float("nan"), float("inf"), float("-inf")]])
    out = _ensure_shape(inp)
    assert torch.isfinite(out).all()
    assert out[0, 0, 1].item() == 0.0
    assert out[0, 0, 2].item() == 0.0
    assert out[0, 0, 3].item() == 0.0


def test_ensure_shape_casts_to_float32():
    out = _ensure_shape(torch.randn(3, 4).to(torch.float16))
    assert out.dtype == torch.float32


# =============================================================================
# Section B — `compute_entropic_rank_per_layer`
# =============================================================================
def test_entropic_rank_empty_dict_returns_empty_arrays():
    layers, ranks = compute_entropic_rank_per_layer({}, epsilon=1e-8)
    assert layers.size == 0
    assert ranks.size == 0


def test_entropic_rank_skips_layers_with_no_tensors():
    layers, ranks = compute_entropic_rank_per_layer({3: [], 1: []}, epsilon=1e-8)
    # No layer produces a rank → both arrays empty
    assert ranks.size == 0


def test_entropic_rank_redundant_heads_approaches_one():
    # Four identical heads → rank ≈ 1.
    T, D = 3, 8
    same = torch.ones(T, D)
    head_outputs = {0: [same.clone() for _ in range(4)]}
    _, ranks = compute_entropic_rank_per_layer(head_outputs, epsilon=1e-8)
    assert ranks[0] == pytest.approx(1.0, abs=1e-3)


def test_entropic_rank_orthogonal_heads_approaches_num_heads():
    # Four orthogonal one-hot directions → rank ≈ num_heads = 4.
    T, D = 3, 8
    eye = torch.eye(4, D)  # 4 orthogonal unit vectors in R^D
    head_outputs = {0: [eye[i].unsqueeze(0).expand(T, D).contiguous() for i in range(4)]}
    _, ranks = compute_entropic_rank_per_layer(head_outputs, epsilon=1e-8)
    assert ranks[0] == pytest.approx(4.0, abs=0.05)


def test_entropic_rank_layers_sorted_even_if_inserted_unordered():
    T, D = 2, 4
    same = torch.ones(T, D)
    head_outputs = {
        5: [same.clone(), same.clone()],
        1: [same.clone(), same.clone()],
        3: [same.clone(), same.clone()],
    }
    layers, ranks = compute_entropic_rank_per_layer(head_outputs, epsilon=1e-8)
    assert list(layers) == [1, 3, 5]
    assert len(ranks) == 3


def test_entropic_rank_output_dtypes():
    same = torch.ones(2, 4)
    layers, ranks = compute_entropic_rank_per_layer({0: [same, same]}, epsilon=1e-8)
    assert layers.dtype == np.int64 or layers.dtype == np.int32
    assert ranks.dtype == np.float32


# =============================================================================
# Section C — `add_head_hooks`
# =============================================================================
def test_add_head_hooks_chronos_calls_once_per_attention_type(pipeline):
    add_head_hooks("chronos", pipeline, num_layers=2, num_heads=3, attention_types=["ca", "sa"])
    assert pipeline.remove_all_hooks_count == 1
    assert len(pipeline.hook_calls) == 2
    # All-heads list has length L*H = 6
    all_heads_arg, kwargs_first = pipeline.hook_calls[0]
    assert len(all_heads_arg[0]) == 6
    assert kwargs_first == {"attention_type": "ca"}
    assert pipeline.hook_calls[1][1] == {"attention_type": "sa"}


def test_add_head_hooks_chronos_bolt_calls_once_per_attention_type(pipeline):
    add_head_hooks("chronos_bolt", pipeline, num_layers=1, num_heads=2, attention_types=["ca"])
    assert pipeline.hook_calls[0][1] == {"attention_type": "ca"}


@pytest.mark.parametrize("model_name", ["timesfm", "toto", "chronos2", "moirai"])
def test_add_head_hooks_single_attention_models_call_without_kwarg(pipeline, model_name):
    add_head_hooks(model_name, pipeline, num_layers=2, num_heads=4, attention_types=["attn"])
    assert len(pipeline.hook_calls) == 1
    args, kwargs = pipeline.hook_calls[0]
    assert len(args[0]) == 8
    assert kwargs == {}


def test_add_head_hooks_unknown_model_raises(pipeline):
    with pytest.raises(ValueError, match="Unsupported model for hook setup"):
        add_head_hooks("bogus", pipeline, num_layers=1, num_heads=1, attention_types=["attn"])


def test_add_head_hooks_works_when_remove_all_hooks_missing():
    class Bare:
        def __init__(self):
            self.hook_calls = []

        def add_head_attribution_hooks(self, *args, **kwargs):
            self.hook_calls.append((args, kwargs))

    bare = Bare()
    add_head_hooks("toto", bare, num_layers=1, num_heads=2, attention_types=["attn"])
    assert len(bare.hook_calls) == 1


# =============================================================================
# Section D — `extract_head_outputs`
# =============================================================================
def _head_output_dict(num_layers: int, num_heads: int, T: int, D: int) -> dict:
    """Build a raw output dict: {layer: {head: [tensor(b=1, T, D)]}}.

    `collect_attributions` produces shape (b, num_calls, D) where each call's
    last time-step is taken. With one tensor per head, output is (1, 1, D).
    """
    out: dict[int, dict[int, list[torch.Tensor]]] = {}
    for layer in range(num_layers):
        out[layer] = {head: [torch.randn(1, T, D)] for head in range(num_heads)}
    return out


def test_extract_head_outputs_chronos_uses_attention_typed_attr(pipeline):
    pipeline.ca_head_attribution_outputs = _head_output_dict(2, 3, T=4, D=8)
    result = extract_head_outputs("chronos", pipeline, attention_type="ca")
    assert sorted(result.keys()) == [0, 1]
    assert len(result[0]) == 3
    # collect_attributions yields shape (b, 1, D) when given one tensor per head
    assert result[0][0].shape == (1, 1, 8)


def test_extract_head_outputs_chronos_bolt_sa(pipeline):
    pipeline.sa_head_attribution_outputs = _head_output_dict(1, 2, T=3, D=4)
    result = extract_head_outputs("chronos_bolt", pipeline, attention_type="sa")
    assert list(result.keys()) == [0]
    assert len(result[0]) == 2


@pytest.mark.parametrize("model_name", ["timesfm", "toto", "chronos2", "moirai"])
def test_extract_head_outputs_single_attention_models_use_generic_attr(pipeline, model_name):
    pipeline.head_attribution_outputs = _head_output_dict(2, 4, T=3, D=6)
    result = extract_head_outputs(model_name, pipeline, attention_type="attn")
    assert sorted(result.keys()) == [0, 1]
    assert len(result[0]) == 4


def test_extract_head_outputs_drops_layers_with_only_empty_heads(pipeline):
    pipeline.head_attribution_outputs = {
        0: {0: [torch.randn(1, 3, 4)], 1: [torch.randn(1, 3, 4)]},
        1: {0: [], 1: []},  # both heads empty → layer dropped
    }
    result = extract_head_outputs("toto", pipeline, attention_type="attn")
    assert list(result.keys()) == [0]


def test_extract_head_outputs_unknown_model_raises(pipeline):
    with pytest.raises(ValueError, match="Unsupported model for extracting head outputs"):
        extract_head_outputs("bogus", pipeline, attention_type="attn")


# =============================================================================
# Section E — `reset_pipeline_outputs`
# =============================================================================
def test_reset_pipeline_outputs_calls_both_resets(pipeline):
    reset_pipeline_outputs(pipeline)
    assert pipeline.reset_attribution_count == 1
    assert pipeline.reset_custom_count == 1


def test_reset_pipeline_outputs_works_when_one_method_missing():
    class OnlyAttribution:
        def __init__(self):
            self.reset_attribution_count = 0

        def reset_attribution_inputs_and_outputs(self):
            self.reset_attribution_count += 1

    p = OnlyAttribution()
    reset_pipeline_outputs(p)  # must not raise
    assert p.reset_attribution_count == 1


def test_reset_pipeline_outputs_no_methods_is_noop():
    class Empty:
        pass

    reset_pipeline_outputs(Empty())  # must not raise


# =============================================================================
# Section F — `run_inference_for_model`
# =============================================================================
def test_run_inference_chronos_forwards_sampling_args(cfg, pipeline):
    ctx = torch.randn(1, 10)
    run_inference_for_model("chronos", pipeline, ctx, cfg, torch.device("cpu"), torch.float32)
    assert len(pipeline.predict_full_calls) == 1
    _, kw = pipeline.predict_full_calls[0]
    assert kw["prediction_length"] == 7
    assert kw["num_samples"] == 5
    assert kw["limit_prediction_length"] is False
    assert kw["return_dict_in_generate"] is True
    assert kw["do_sample"] is True  # deterministic=False → do_sample=True
    assert kw["context"].dtype == torch.float32
    assert kw["context"].device.type == "cpu"


def test_run_inference_chronos_bolt_no_sampling_args(cfg, pipeline):
    ctx = torch.randn(1, 10)
    run_inference_for_model("chronos_bolt", pipeline, ctx, cfg, torch.device("cpu"), torch.float32)
    _, kw = pipeline.predict_full_calls[0]
    assert "num_samples" not in kw
    assert "do_sample" not in kw
    assert kw["limit_prediction_length"] is False


def test_run_inference_timesfm_minimal_kwargs(cfg, pipeline):
    ctx = torch.randn(1, 10)
    run_inference_for_model("timesfm", pipeline, ctx, cfg, torch.device("cpu"), torch.float32)
    _, kw = pipeline.predict_calls[0]
    assert kw["prediction_length"] == 7
    assert "num_samples" not in kw


def test_run_inference_toto_forwards_kv_cache_and_samples_per_batch(cfg, pipeline):
    ctx = torch.randn(1, 10)
    run_inference_for_model("toto", pipeline, ctx, cfg, torch.device("cpu"), torch.float32)
    _, kw = pipeline.predict_calls[0]
    assert kw["num_samples"] == 5
    assert kw["samples_per_batch"] == 4  # from cfg.toto.samples_per_batch
    assert kw["use_kv_cache"] is True


def test_run_inference_toto_falls_back_when_samples_per_batch_null(pipeline):
    cfg = OmegaConf.create(
        {
            "entropic_rank": {"prediction_length": 7, "num_samples": 20, "deterministic": False},
            "toto": {"samples_per_batch": None, "use_kv_cache": False},
        }
    )
    ctx = torch.randn(1, 10)
    run_inference_for_model("toto", pipeline, ctx, cfg, torch.device("cpu"), torch.float32)
    _, kw = pipeline.predict_calls[0]
    assert kw["samples_per_batch"] == 10  # min(num_samples=20, 10)


def test_run_inference_chronos2_forwards_batch_size(cfg, pipeline):
    ctx = torch.randn(1, 10)
    run_inference_for_model("chronos2", pipeline, ctx, cfg, torch.device("cpu"), torch.float32)
    _, kw = pipeline.predict_calls[0]
    assert kw["prediction_length"] == 7
    assert kw["batch_size"] == 13  # from cfg.eval.batch_size
    assert kw["limit_prediction_length"] is False


def test_run_inference_moirai_forwards_patch_size_and_num_samples(cfg, pipeline):
    ctx = torch.randn(1, 10)
    run_inference_for_model("moirai", pipeline, ctx, cfg, torch.device("cpu"), torch.float32)
    _, kw = pipeline.predict_calls[0]
    assert kw["prediction_length"] == 7
    assert kw["num_samples"] == 5
    assert kw["patch_size"] == 32


def test_run_inference_unknown_model_raises(cfg, pipeline):
    ctx = torch.randn(1, 10)
    with pytest.raises(ValueError, match="Unsupported model for inference"):
        run_inference_for_model("bogus", pipeline, ctx, cfg, torch.device("cpu"), torch.float32)
