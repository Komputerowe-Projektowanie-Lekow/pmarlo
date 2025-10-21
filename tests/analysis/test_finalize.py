import copy

import numpy as np
import pytest

_KB_KJ_PER_MOL = 0.00831446261815324

from pmarlo.reweight import AnalysisReweightMode
from pmarlo.workflow.finalize import AnalysisConfig, finalize_dataset


def _biased_dataset(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    low = rng.normal(loc=[-1.5, -1.0], scale=0.05, size=(60, 2))
    high = rng.normal(loc=[1.5, 1.0], scale=0.05, size=(60, 2))
    X = np.vstack([low, high]).astype(np.float64)
    energy = np.concatenate(
        [
            np.full(low.shape[0], -5.0, dtype=np.float64),
            np.full(high.shape[0], 12.0, dtype=np.float64),
        ]
    )
    bias = np.zeros_like(energy)
    beta = 1.0 / (_KB_KJ_PER_MOL * 450.0)
    feature_schema = {
        "n_features": X.shape[1],
        "names": ["feature_0", "feature_1"],
    }
    whitening_meta = {
        "shard_id": "demo-train",
        "feature_schema": feature_schema,
        "output_mean": np.zeros(X.shape[1], dtype=np.float64).tolist(),
        "output_transform": np.eye(X.shape[1], dtype=np.float64).tolist(),
        "output_transform_applied": False,
    }
    dataset = {
        "splits": {
            "train": {
                "X": X,
                "energy": energy,
                "bias": bias,
                "beta": beta,
                "feature_schema": feature_schema,
                "meta": whitening_meta,
            }
        }
    }
    return dataset


def test_finalize_reweight_changes_stationary_and_fes():
    dataset = _biased_dataset()

    cfg_none = AnalysisConfig(
        temperature_ref_K=300.0,
        lag_time=1,
        n_microstates=3,
        cluster_mode="kmeans",
        reweight=AnalysisReweightMode.NONE,
        fes_bins=16,
    )
    result_none = finalize_dataset(copy.deepcopy(dataset), cfg_none)

    cfg_weighted = AnalysisConfig(
        temperature_ref_K=300.0,
        lag_time=1,
        n_microstates=3,
        cluster_mode="kmeans",
        reweight=AnalysisReweightMode.MBAR,
        fes_bins=16,
    )
    result_weighted = finalize_dataset(copy.deepcopy(dataset), cfg_weighted)

    assert result_weighted["reweight_mode"] == AnalysisReweightMode.MBAR
    assert "frame_weights" in result_weighted
    weights_train = result_weighted["frame_weights"]["train"]
    assert np.isclose(weights_train.sum(), 1.0)

    pi_none = result_none["stationary_distribution"]
    pi_weighted = result_weighted["stationary_distribution"]
    assert pi_none.shape == pi_weighted.shape
    assert not np.allclose(pi_none, pi_weighted)

    fes_none = result_none["fes"]["free_energy"]
    fes_weighted = result_weighted["fes"]["free_energy"]
    assert fes_none.shape == fes_weighted.shape
    assert np.max(np.abs(fes_none - fes_weighted)) > 1e-6

    assert "diagnostics" in result_none
    assert isinstance(result_none["diagnostics"], dict)
    assert "diagnostics" in result_weighted


def test_finalize_requires_thermo_information():
    dataset = {
        "splits": {
            "train": {
                "X": np.random.default_rng(1).normal(size=(20, 2)),
                "meta": {"shard_id": "train"},
            }
        }
    }
    cfg = AnalysisConfig(reweight=AnalysisReweightMode.MBAR, n_microstates=3)
    with pytest.raises(ValueError):
        finalize_dataset(dataset, cfg)


def _short_trajectory_dataset(frames: int = 200) -> dict:
    rng = np.random.default_rng(7)
    X = rng.normal(loc=0.0, scale=1.0, size=(frames, 2)).astype(np.float64)
    dtraj = np.mod(np.arange(frames, dtype=int), 5)
    feature_schema = {"n_features": 2, "names": ["cv0", "cv1"]}
    whitening_meta = {
        "shard_id": "short",
        "feature_schema": feature_schema,
        "output_mean": np.zeros(2, dtype=np.float64).tolist(),
        "output_transform": np.eye(2, dtype=np.float64).tolist(),
        "output_transform_applied": False,
    }
    dataset = {
        "splits": {
            "train": {"X": X, "feature_schema": feature_schema, "meta": whitening_meta}
        },
        "__shards__": [{"id": "short", "start": 0, "stop": frames}],
        "dtrajs": [dtraj],
    }
    return dataset


def test_finalize_collects_debug_warning_for_short_dataset():
    dataset = _short_trajectory_dataset(frames=200)
    cfg = AnalysisConfig(
        temperature_ref_K=300.0,
        lag_time=10,
        n_microstates=5,
        cluster_mode="kmeans",
        collect_debug_data=True,
        fes_bins=16,
        reweight=AnalysisReweightMode.NONE,
    )

    result = finalize_dataset(copy.deepcopy(dataset), cfg)

    assert "analysis_debug" in result
    debug_data = result["analysis_debug"]
    assert debug_data.summary["tau_frames"] == cfg.lag_time
    assert 0 < debug_data.summary["total_pairs"] < 5000

    warnings = result.get("warnings", [])
    assert any("TOTAL_PAIRS_LT_5000" in warning for warning in warnings)

    fes = result["fes"]
    assert fes["free_energy"].ndim == 2
