from __future__ import annotations

import sys
import types

import numpy as np
import pytest


if "openmm" not in sys.modules:
    openmm_stub = types.ModuleType("openmm")

    class _OpenMMException(Exception):
        pass

    class _Platform:
        @staticmethod
        def getPlatformByName(name: str) -> None:  # pragma: no cover - simple stub
            raise _OpenMMException(f"platform {name!r} unavailable in tests")

    openmm_stub.OpenMMException = _OpenMMException
    openmm_stub.Platform = _Platform
    sys.modules["openmm"] = openmm_stub

if "scipy" not in sys.modules:
    scipy_stub = types.ModuleType("scipy")
    ndimage_stub = types.ModuleType("scipy.ndimage")
    stats_stub = types.ModuleType("scipy.stats")
    mstats_stub = types.ModuleType("scipy.stats.mstats")
    constants_stub = types.ModuleType("scipy.constants")

    def _gaussian_filter(data, sigma, mode="nearest"):
        return data

    def _iqr(data, axis=None, rng=(25, 75), scale=1.0, nan_policy="propagate"):
        return 0.0

    ndimage_stub.gaussian_filter = _gaussian_filter
    def _mquantiles(data, prob=(0.25, 0.5, 0.75), alphap=0.5, betap=0.5):
        return data

    stats_stub.iqr = _iqr
    mstats_stub.mquantiles = _mquantiles
    constants_stub.k = 1.380649e-23
    constants_stub.Avogadro = 6.02214076e23
    scipy_stub.ndimage = ndimage_stub
    scipy_stub.stats = stats_stub
    scipy_stub.constants = constants_stub
    stats_stub.mstats = mstats_stub
    sys.modules["scipy"] = scipy_stub
    sys.modules["scipy.ndimage"] = ndimage_stub
    sys.modules["scipy.stats"] = stats_stub
    sys.modules["scipy.stats.mstats"] = mstats_stub
    sys.modules["scipy.constants"] = constants_stub

if "sklearn" not in sys.modules:
    sklearn_stub = types.ModuleType("sklearn")
    cross_decomposition_stub = types.ModuleType("sklearn.cross_decomposition")
    cluster_stub = types.ModuleType("sklearn.cluster")

    class _CCA:
        def __init__(self, *args, **kwargs) -> None:  # pragma: no cover - stub
            pass

    class _KMeans:
        def __init__(self, *args, **kwargs) -> None:  # pragma: no cover - stub
            pass

        def fit(self, X):  # pragma: no cover - stub
            return self

    class _MiniBatchKMeans(_KMeans):
        pass

    cross_decomposition_stub.CCA = _CCA
    cluster_stub.KMeans = _KMeans
    cluster_stub.MiniBatchKMeans = _MiniBatchKMeans
    sklearn_stub.cross_decomposition = cross_decomposition_stub
    sklearn_stub.cluster = cluster_stub
    sys.modules["sklearn"] = sklearn_stub
    sys.modules["sklearn.cross_decomposition"] = cross_decomposition_stub
    sys.modules["sklearn.cluster"] = cluster_stub

if "deeptime" not in sys.modules:
    deeptime_stub = types.ModuleType("deeptime")
    markov_stub = types.ModuleType("deeptime.markov")
    msm_stub = types.ModuleType("deeptime.markov.msm")
    pcca_stub = types.ModuleType("deeptime.markov.pcca")
    tools_stub = types.ModuleType("deeptime.markov.tools")
    analysis_stub = types.ModuleType("deeptime.markov.tools.analysis")
    estimation_stub = types.ModuleType("deeptime.markov.tools.estimation")
    dense_stub = types.ModuleType("deeptime.markov.tools.estimation.dense")
    transition_matrix_stub = types.ModuleType(
        "deeptime.markov.tools.estimation.dense.transition_matrix"
    )

    class _MaximumLikelihoodMSM:
        def __init__(self, *args, **kwargs) -> None:  # pragma: no cover - stub
            pass

        def fit_fetch(self, *args, **kwargs):  # pragma: no cover - stub
            class _Result:
                def stationary_distribution(self):
                    return np.asarray([1.0])

            return np.asarray([[1.0]]), _Result()

    def _stationary_distribution(arr, check_inputs=False):
        if arr.size == 0:
            return np.asarray([], dtype=float)
        n = arr.shape[0]
        return np.full((n,), 1.0 / n, dtype=float)

    def _transition_matrix_non_reversible(arr):
        return np.asarray(arr, dtype=float)

    msm_stub.MaximumLikelihoodMSM = _MaximumLikelihoodMSM
    analysis_stub.stationary_distribution = _stationary_distribution
    transition_matrix_stub.transition_matrix_non_reversible = (
        _transition_matrix_non_reversible
    )

    deeptime_stub.markov = markov_stub
    markov_stub.msm = msm_stub
    markov_stub.pcca = pcca_stub
    markov_stub.tools = tools_stub
    tools_stub.analysis = analysis_stub
    tools_stub.estimation = estimation_stub
    estimation_stub.dense = dense_stub
    dense_stub.transition_matrix = transition_matrix_stub

    sys.modules["deeptime"] = deeptime_stub
    sys.modules["deeptime.markov"] = markov_stub
    sys.modules["deeptime.markov.msm"] = msm_stub
    sys.modules["deeptime.markov.pcca"] = pcca_stub
    sys.modules["deeptime.markov.tools"] = tools_stub
    sys.modules["deeptime.markov.tools.analysis"] = analysis_stub
    sys.modules["deeptime.markov.tools.estimation"] = estimation_stub
    sys.modules["deeptime.markov.tools.estimation.dense"] = dense_stub
    sys.modules[
        "deeptime.markov.tools.estimation.dense.transition_matrix"
    ] = transition_matrix_stub

if "pmarlo.analysis" not in sys.modules:
    analysis_stub = types.ModuleType("pmarlo.analysis")

    def _compute_diagnostics(*args, **kwargs):  # pragma: no cover - stub
        return {}

    analysis_stub.compute_diagnostics = _compute_diagnostics
    sys.modules["pmarlo.analysis"] = analysis_stub

if "pmarlo.analysis.fes" not in sys.modules:
    analysis_fes_stub = types.ModuleType("pmarlo.analysis.fes")

    def _ensure_fes_inputs_whitened(dataset, names):  # pragma: no cover - stub
        return dataset

    analysis_fes_stub.ensure_fes_inputs_whitened = _ensure_fes_inputs_whitened
    sys.modules["pmarlo.analysis.fes"] = analysis_fes_stub
    if "pmarlo.analysis" in sys.modules:
        sys.modules["pmarlo.analysis"].fes = analysis_fes_stub

if "pmarlo.analysis.msm" not in sys.modules:
    analysis_msm_stub = types.ModuleType("pmarlo.analysis.msm")

    def _ensure_msm_inputs_whitened(dataset):  # pragma: no cover - stub
        return dataset

    analysis_msm_stub.ensure_msm_inputs_whitened = _ensure_msm_inputs_whitened
    sys.modules["pmarlo.analysis.msm"] = analysis_msm_stub
    if "pmarlo.analysis" in sys.modules:
        sys.modules["pmarlo.analysis"].msm = analysis_msm_stub

if "pmarlo.markov_state_model" not in sys.modules:
    markov_state_model_stub = types.ModuleType("pmarlo.markov_state_model")
    sys.modules["pmarlo.markov_state_model"] = markov_state_model_stub

if "pmarlo.markov_state_model._msm_utils" not in sys.modules:
    msm_utils_stub = types.ModuleType("pmarlo.markov_state_model._msm_utils")

    def _build_simple_msm(*args, **kwargs):  # pragma: no cover - stub
        return np.empty((0, 0)), np.empty((0,))

    msm_utils_stub.build_simple_msm = _build_simple_msm
    sys.modules["pmarlo.markov_state_model._msm_utils"] = msm_utils_stub
    if "pmarlo.markov_state_model" in sys.modules:
        sys.modules["pmarlo.markov_state_model"]._msm_utils = msm_utils_stub

from pmarlo.transform.build import BuildOpts, estimate_memory_usage


class _Dataset:
    def __init__(self, n_frames: int, feature_names: list[str]):
        self.n_frames = n_frames
        self.feature_names = feature_names

    def __len__(self) -> int:
        return self.n_frames


def test_estimate_memory_usage_calculates_expected_gb() -> None:
    dataset = _Dataset(n_frames=1200, feature_names=["f1", "f2", "f3", "f4"])
    opts = BuildOpts(n_clusters=150, n_states=30, enable_fes=True)

    n_frames = dataset.n_frames
    n_features = len(dataset.feature_names)

    dataset_gb = (n_frames * n_features * 8) / (1024**3)
    msm_gb = (opts.n_clusters * n_features * 8) / (1024**3)
    msm_gb += (opts.n_states * opts.n_states * 8) / (1024**3)
    fes_gb = (100 * 100 * 8) / (1024**3)

    expected = dataset_gb + msm_gb + fes_gb

    assert estimate_memory_usage(dataset, opts) == pytest.approx(expected)


def test_estimate_memory_usage_requires_frames() -> None:
    dataset = _Dataset(n_frames=0, feature_names=["f1"])
    opts = BuildOpts()

    with pytest.raises(ValueError, match="has no frames"):
        estimate_memory_usage(dataset, opts)


def test_estimate_memory_usage_requires_features() -> None:
    dataset = _Dataset(n_frames=10, feature_names=[])
    opts = BuildOpts()

    with pytest.raises(ValueError, match="has no feature names"):
        estimate_memory_usage(dataset, opts)
