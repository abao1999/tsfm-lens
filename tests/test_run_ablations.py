"""Unit tests for `scripts/run_ablations.py`.

Covered: MODELS registry, the 6 per-model `_<model>_config` builders,
`_install_chronos2_predict_wrapper`'s normalize-and-unwrap behaviour, and
`_save_metrics_json`. No GPU, no model downloads, no real data.

Loaders (`_load_<model>`), `_load_test_datasets`, and `main()` are out of
scope — they need real models / Hydra runtime / data files.

Run with:
.venv/bin/pytest tests/ -v
"""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from scripts.run_ablations import (
    MODELS,
    MOIRAI_TRAINED_CONTEXT_LENGTH,
    _chronos2_config,
    _chronos_bolt_config,
    _chronos_config,
    _install_chronos2_predict_wrapper,
    _moirai_config,
    _save_metrics_json,
    _timesfm_config,
    _toto_config,
)


# =============================================================================
# Fixtures
# =============================================================================
@pytest.fixture
def cfg():
    """Synthetic Hydra DictConfig with every key the 6 config builders touch."""
    return OmegaConf.create(
        {
            "eval": {"batch_size": 16},
            "chronos": {
                "model_id": "amazon/chronos-t5-base",
                "deterministic": False,
                "num_samples": 20,
                "limit_prediction_length": False,
                "top_k": 50,
                "top_p": 1.0,
                "temperature": 1.0,
                "context_length": 512,
            },
            "chronos_bolt": {
                "model_id": "amazon/chronos-bolt-base",
                "limit_prediction_length": False,
                "context_length": 512,
            },
            "chronos2": {"model_id": "amazon/chronos-2", "context_length": 8192},
            "timesfm": {"model_id": "google/timesfm-2.5-200m-pytorch", "context_length": 512},
            "toto": {
                "model_id": "Datadog/Toto-Open-Base-1.0",
                "num_samples": 10,
                "samples_per_batch": 10,
                "use_kv_cache": True,
                "context_length": 4096,
            },
            "moirai": {
                "model_id": "Salesforce/moirai-1.1-R-base",
                "num_samples": 100,
                "patch_size": 32,
            },
        }
    )


@pytest.fixture
def t5_pipeline():
    """Mock for chronos: pipeline.model.config has .context_length and .prediction_length."""
    return SimpleNamespace(model=SimpleNamespace(config=SimpleNamespace(context_length=2048, prediction_length=64)))


@pytest.fixture
def bolt_pipeline():
    """Mock for chronos_bolt: .model.config.chronos_config (dict) and .quantiles list."""
    return SimpleNamespace(
        model=SimpleNamespace(config=SimpleNamespace(chronos_config={"context_length": 2048, "prediction_length": 64})),
        quantiles=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    )


class RecordingChronos2:
    """Minimal mock with a `predict` method that records its inputs and returns (point, quantiles)."""

    def __init__(self):
        self.last_args = None
        self.last_kwargs = None
        self.point_return = torch.tensor([[1.0, 2.0, 3.0]])

    def predict(self, *args, **kwargs):
        self.last_args = args
        self.last_kwargs = kwargs
        return self.point_return, "QUANTILES_PLACEHOLDER"


@pytest.fixture
def chronos2_pipeline():
    return RecordingChronos2()


# =============================================================================
# Section A — MODELS registry
# =============================================================================
def test_models_registry_keys():
    assert set(MODELS) == {"chronos", "chronos_bolt", "chronos2", "timesfm", "toto", "moirai"}


def test_models_registry_values_shape():
    for model_type, value in MODELS.items():
        assert isinstance(value, tuple) and len(value) == 2, f"{model_type}: expected (loader, builder) tuple"
        loader, builder = value
        assert callable(loader), f"{model_type}: loader not callable"
        assert callable(builder), f"{model_type}: config builder not callable"


def test_moirai_trained_context_length():
    assert MOIRAI_TRAINED_CONTEXT_LENGTH == 4000


# =============================================================================
# Section B — Per-model config builders
# =============================================================================
def test_chronos_default_num_samples(cfg, t5_pipeline):
    out = _chronos_config(t5_pipeline, cfg)
    assert out["context_length"] == cfg.chronos.context_length
    assert out["num_samples"] == cfg.chronos.num_samples
    pk = out["prediction_kwargs"]
    assert pk["num_samples"] == cfg.chronos.num_samples
    assert pk["top_k"] == cfg.chronos.top_k
    assert pk["top_p"] == cfg.chronos.top_p
    assert pk["temperature"] == cfg.chronos.temperature
    assert pk["limit_prediction_length"] == cfg.chronos.limit_prediction_length
    assert "do_sample" not in pk  # non-deterministic path doesn't add this


def test_chronos_deterministic_forces_num_samples_1(cfg, t5_pipeline):
    cfg.chronos.deterministic = True
    cfg.chronos.num_samples = 20  # >1 to trigger the warning branch
    out = _chronos_config(t5_pipeline, cfg)
    assert out["num_samples"] == 1
    assert out["prediction_kwargs"]["num_samples"] == 1
    assert out["prediction_kwargs"]["do_sample"] is False


def test_chronos_bolt_uses_quantiles_length(cfg, bolt_pipeline):
    out = _chronos_bolt_config(bolt_pipeline, cfg)
    assert out["context_length"] == cfg.chronos_bolt.context_length
    assert out["num_samples"] == len(bolt_pipeline.quantiles)
    assert out["prediction_kwargs"] == {"limit_prediction_length": cfg.chronos_bolt.limit_prediction_length}


def test_chronos2_installs_wrapper_and_returns_config(cfg, chronos2_pipeline):
    original_predict = chronos2_pipeline.predict
    out = _chronos2_config(chronos2_pipeline, cfg)
    # Wrapper installed
    assert chronos2_pipeline.predict is not original_predict
    # Config shape
    assert out["context_length"] == cfg.chronos2.context_length
    assert out["num_samples"] == 1
    assert out["prediction_kwargs"]["batch_size"] == cfg.eval.batch_size
    assert out["prediction_kwargs"]["limit_prediction_length"] is False


def test_timesfm_empty_prediction_kwargs(cfg):
    out = _timesfm_config(pipeline=object(), cfg=cfg)
    assert out == {"context_length": cfg.timesfm.context_length, "prediction_kwargs": {}, "num_samples": 1}


def test_toto_prediction_kwargs(cfg):
    out = _toto_config(pipeline=object(), cfg=cfg)
    assert out["context_length"] == cfg.toto.context_length
    assert out["num_samples"] == cfg.toto.num_samples
    assert set(out["prediction_kwargs"]) == {"num_samples", "samples_per_batch", "use_kv_cache"}
    assert out["prediction_kwargs"]["use_kv_cache"] is True


def test_moirai_uses_constant_context_length(cfg):
    out = _moirai_config(pipeline=object(), cfg=cfg)
    assert out["context_length"] == MOIRAI_TRAINED_CONTEXT_LENGTH
    assert out["num_samples"] == cfg.moirai.num_samples
    assert out["prediction_kwargs"] == {"num_samples": cfg.moirai.num_samples}


# =============================================================================
# Section C — Chronos-2 predict wrapper (normalize + unwrap)
# =============================================================================
def test_wrapper_passes_tensor_through(chronos2_pipeline):
    _install_chronos2_predict_wrapper(chronos2_pipeline)
    ctx = torch.randn(3, 5)
    chronos2_pipeline.predict(ctx)
    received = chronos2_pipeline.last_args[0]
    assert torch.equal(received, ctx)


def test_wrapper_stacks_list_of_same_shape_1d(chronos2_pipeline):
    _install_chronos2_predict_wrapper(chronos2_pipeline)
    parts = [torch.arange(5.0), torch.arange(5.0) + 10, torch.arange(5.0) + 100]
    chronos2_pipeline.predict(parts)
    received = chronos2_pipeline.last_args[0]
    assert received.shape == (3, 5)
    assert torch.equal(received, torch.stack(parts, dim=0))


def test_wrapper_unbinds_2d_entries(chronos2_pipeline):
    _install_chronos2_predict_wrapper(chronos2_pipeline)
    ctx_2d = torch.arange(10.0).reshape(2, 5)
    chronos2_pipeline.predict([ctx_2d])
    received = chronos2_pipeline.last_args[0]
    assert received.shape == (2, 5)
    assert torch.equal(received, ctx_2d)


def test_wrapper_left_pads_heterogeneous_lengths(chronos2_pipeline):
    _install_chronos2_predict_wrapper(chronos2_pipeline)
    parts = [torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]), torch.tensor([6.0, 7.0, 8.0])]
    chronos2_pipeline.predict(parts)
    received = chronos2_pipeline.last_args[0]
    # left_pad_and_stack_1D pads to max length (5) on the left
    assert received.shape == (2, 5)
    assert torch.equal(received[0], parts[0])
    # last 3 entries of the padded row are the short tensor; first 2 are padding
    assert torch.equal(received[1, -3:], parts[1])


def test_wrapper_empty_list_raises_value_error(chronos2_pipeline):
    _install_chronos2_predict_wrapper(chronos2_pipeline)
    with pytest.raises(ValueError, match="empty"):
        chronos2_pipeline.predict([])


def test_wrapper_unwraps_point_forecast(chronos2_pipeline):
    _install_chronos2_predict_wrapper(chronos2_pipeline)
    expected_point = chronos2_pipeline.point_return  # captured before wrapper installs
    result = chronos2_pipeline.predict(torch.randn(1, 5))
    assert torch.equal(result, expected_point)  # tuple's second element is dropped


def test_wrapper_handles_context_kwarg(chronos2_pipeline):
    _install_chronos2_predict_wrapper(chronos2_pipeline)
    parts = [torch.arange(4.0), torch.arange(4.0)]
    chronos2_pipeline.predict(context=parts)
    # When passed as kwarg, the normalized tensor should land in last_kwargs["context"]
    received = chronos2_pipeline.last_kwargs["context"]
    assert received.shape == (2, 4)


# =============================================================================
# Section D — _save_metrics_json
# =============================================================================
@pytest.fixture
def sample_metrics():
    """Three non-empty metric dicts in the shape produced by evaluate_ablations."""
    return {
        "metrics": {0: {"sysA": {"mae": 0.1}}},
        "ablation_vs_original": {"za_heads_layers_0": {0: {"sysA": {"mae": 0.2}}}},
        "ablation_vs_labels": {"za_heads_layers_0": {0: {"sysA": {"mae": 0.3}}}},
    }


def test_writes_three_files_at_expected_paths(tmp_path, monkeypatch, sample_metrics):
    monkeypatch.chdir(tmp_path)
    _save_metrics_json(
        ablations_subdir="ablations_chronos_bolt_test",
        rseed=99,
        metrics_fname="metrics_test",
        ablate_n_heads_per_layer=1,
        **sample_metrics,
    )
    base = tmp_path / "outputs" / "ablations_chronos_bolt_test" / "rseed-99"
    assert (base / "original_vs_labels" / "nheads-1" / "metrics_test.json").is_file()
    assert (base / "ablations_vs_original" / "nheads-1" / "metrics_test_ablations_against_original.json").is_file()
    assert (base / "ablations_vs_labels" / "nheads-1" / "metrics_test_ablations_against_labels.json").is_file()


def test_nheads_subdir_is_all_heads_when_none(tmp_path, monkeypatch, sample_metrics):
    monkeypatch.chdir(tmp_path)
    _save_metrics_json(
        ablations_subdir="sub",
        rseed=7,
        metrics_fname="m",
        ablate_n_heads_per_layer=None,
        **sample_metrics,
    )
    base = tmp_path / "outputs" / "sub" / "rseed-7"
    assert (base / "original_vs_labels" / "all_heads" / "m.json").is_file()
    assert not (base / "original_vs_labels" / "nheads-None").exists()


def test_skips_empty_data_dicts(tmp_path, monkeypatch, sample_metrics):
    monkeypatch.chdir(tmp_path)
    _save_metrics_json(
        ablations_subdir="sub",
        rseed=7,
        metrics_fname="m",
        ablate_n_heads_per_layer=2,
        metrics={},  # empty → should be skipped
        ablation_vs_original=sample_metrics["ablation_vs_original"],
        ablation_vs_labels=sample_metrics["ablation_vs_labels"],
    )
    base = tmp_path / "outputs" / "sub" / "rseed-7"
    assert not (base / "original_vs_labels" / "nheads-2" / "m.json").exists()
    assert (base / "ablations_vs_original" / "nheads-2" / "m_ablations_against_original.json").is_file()
    assert (base / "ablations_vs_labels" / "nheads-2" / "m_ablations_against_labels.json").is_file()


def test_serializes_non_json_native_values(tmp_path, monkeypatch):
    """`make_json_serializable` handles numpy scalars and ndarrays inside the dicts."""
    monkeypatch.chdir(tmp_path)
    metrics = {0: {"sysA": {"mae": np.float64(0.5), "arr": np.array([1.0, 2.0])}}}
    _save_metrics_json(
        ablations_subdir="sub",
        rseed=7,
        metrics_fname="m",
        ablate_n_heads_per_layer=1,
        metrics=metrics,
        ablation_vs_original={"x": {0: {"sysA": {"mae": np.int64(3)}}}},
        ablation_vs_labels={"x": {0: {"sysA": {"mae": 0.7}}}},
    )
    base = tmp_path / "outputs" / "sub" / "rseed-7"
    parsed_metrics = json.loads((base / "original_vs_labels/nheads-1/m.json").read_text())
    # numpy scalar → float; numpy array → list
    assert parsed_metrics["0"]["sysA"]["mae"] == 0.5
    assert parsed_metrics["0"]["sysA"]["arr"] == [1.0, 2.0]
    parsed_orig = json.loads((base / "ablations_vs_original/nheads-1/m_ablations_against_original.json").read_text())
    assert parsed_orig["x"]["0"]["sysA"]["mae"] == 3
