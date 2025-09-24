import copy
import importlib
import pathlib
import sys
import types

import numpy as np

_KB_KJ_PER_MOL = 0.00831446261815324


def _prepare_modules():
    for name in list(sys.modules):
        if name.startswith("pmarlo.workflow.finalize"):
            sys.modules.pop(name)
        if name.startswith("pmarlo.reweight"):
            sys.modules.pop(name)
        if name == "pmarlo.analysis" or name.startswith("pmarlo.analysis."):
            sys.modules.pop(name)

    base = pathlib.Path("src/pmarlo")
    pmarlo_pkg = types.ModuleType("pmarlo")
    pmarlo_pkg.__path__ = [str(base)]
    sys.modules["pmarlo"] = pmarlo_pkg

    ml_pkg = sys.modules.setdefault("pmarlo.ml", types.ModuleType("pmarlo.ml"))
    if not hasattr(ml_pkg, "__path__"):
        ml_pkg.__path__ = []

    deeptica_pkg = types.ModuleType("pmarlo.ml.deeptica")
    deeptica_pkg.__path__ = []
    sys.modules["pmarlo.ml.deeptica"] = deeptica_pkg

    whitening_mod = types.ModuleType("pmarlo.ml.deeptica.whitening")

    def _identity_transform(Y, mean, W, already_applied):
        return np.asarray(Y, dtype=np.float64)

    whitening_mod.apply_output_transform = _identity_transform
    sys.modules["pmarlo.ml.deeptica.whitening"] = whitening_mod
    deeptica_pkg.apply_output_transform = _identity_transform

    finalize_mod = importlib.import_module("pmarlo.workflow.finalize")
    reweight_mod = importlib.import_module("pmarlo.reweight")
    AnalysisConfig = finalize_mod.AnalysisConfig
    finalize_dataset = finalize_mod.finalize_dataset
    AnalysisReweightMode = reweight_mod.AnalysisReweightMode
    return AnalysisConfig, finalize_dataset, AnalysisReweightMode


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
    dataset = {
        "splits": {
            "train": {
                "X": X,
                "energy": energy,
                "bias": bias,
                "beta": beta,
                "meta": {"shard_id": "demo-train"},
            }
        }
    }
    return dataset


def test_finalize_reweight_changes_stationary_and_fes():
    AnalysisConfig, finalize_dataset, AnalysisReweightMode = _prepare_modules()
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


def test_finalize_falls_back_without_thermo_information():
    AnalysisConfig, finalize_dataset, AnalysisReweightMode = _prepare_modules()
    dataset = {
        "splits": {
            "train": {
                "X": np.random.default_rng(1).normal(size=(20, 2)),
                "meta": {"shard_id": "train"},
            }
        }
    }
    cfg = AnalysisConfig(reweight=AnalysisReweightMode.MBAR, n_microstates=3)
    result = finalize_dataset(dataset, cfg)

    assert result["reweight_mode"] == AnalysisReweightMode.NONE
    assert "frame_weights" not in result
