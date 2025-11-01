from __future__ import annotations

"""Performance benchmarks for TICA routines and linear algebra kernels.

These benchmarks cover both high-level TICA workflows and low-level algorithmic
components used in Time-lagged Independent Component Analysis:
- End-to-end TICA fitting via the MSM feature mixin hooks
- Construction of time-lagged and instantaneous covariance matrices
- Solving the generalized eigenvalue problem for slow mode estimation
- Projecting high-dimensional features onto learned TICA components

Run with: pytest -m benchmark tests/perf/test_tica_perf.py
"""

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, List

import numpy as np
import pytest

from pmarlo.markov_state_model._features import FeaturesMixin

pytestmark = [
    pytest.mark.perf,
    pytest.mark.benchmark,
    pytest.mark.msm,
    pytest.mark.tica,
]

# Optional dependencies
pytest.importorskip("pytest_benchmark", reason="pytest-benchmark not installed")
pytest.importorskip("deeptime", reason="deeptime is required for TICA benchmarks")

if not os.getenv("PMARLO_RUN_PERF"):
    pytest.skip(
        "perf tests disabled; set PMARLO_RUN_PERF=1 to run", allow_module_level=True
    )


def _generate_correlated_trajectories(
    n_trajs: int, n_frames: int, n_features: int, *, seed: int = 42
) -> list[np.ndarray]:
    """Generate synthetic trajectories with mild temporal correlations."""

    rng = np.random.default_rng(seed)
    trajectories: list[np.ndarray] = []

    for traj_idx in range(n_trajs):
        data = np.zeros((n_frames, n_features), dtype=np.float64)
        base = rng.standard_normal(n_features)
        data[0] = base
        for t in range(1, n_frames):
            noise = rng.standard_normal(n_features) * 0.2
            data[t] = 0.85 * data[t - 1] + 0.1 * base + noise
        # Add a trajectory-specific offset so the covariance calculation has
        # meaningful structure while staying deterministic.
        data += rng.normal(scale=0.05, size=(1, n_features))
        trajectories.append(data)

    return trajectories


def _generate_correlated_time_series(
    n_frames: int, n_features: int, *, seed: int = 13
) -> np.ndarray:
    """Synthesize correlated trajectories with multiple timescales."""

    rng = np.random.default_rng(seed)
    latent = np.zeros((n_frames, 3), dtype=np.float64)
    # Slow AR(1) processes to ensure clear spectral separation
    for t in range(1, n_frames):
        latent[t, 0] = 0.985 * latent[t - 1, 0] + rng.normal(scale=0.05)
        latent[t, 1] = 0.950 * latent[t - 1, 1] + rng.normal(scale=0.08)
        latent[t, 2] = rng.normal(scale=0.5)

    mixing = rng.normal(scale=0.7, size=(3, n_features))
    noise = rng.normal(scale=0.05, size=(n_frames, n_features))
    series = latent @ mixing + noise
    return series.astype(np.float32, copy=False)


class _TICAHarness(FeaturesMixin):
    """Minimal harness that exposes the TICA mixin hooks for benchmarks."""

    def __init__(
        self,
        trajectories: Iterable[np.ndarray],
        *,
        output_dir: Path,
    ) -> None:
        self.trajectories = [
            SimpleNamespace(n_frames=len(traj)) for traj in trajectories
        ]
        self._trajectory_arrays = [
            np.asarray(traj, dtype=np.float32) for traj in trajectories
        ]
        self.features = np.vstack(self._trajectory_arrays)
        self.feature_stride = 1
        self.tica_lag = 0
        self.tica_components = None
        self.raw_frames = int(self.features.shape[0])
        self.effective_frames = self.raw_frames
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._original = self.features.copy()

    def reset(self) -> None:
        """Restore the original feature matrix for repeated benchmark runs."""

        self.features = self._original.copy()
        self.tica_lag = 0
        self.tica_components = None
        if hasattr(self, "tica_components_"):
            delattr(self, "tica_components_")


@pytest.fixture(scope="module")
def synthetic_trajectories() -> list[np.ndarray]:
    """Provide a reusable set of synthetic trajectories."""

    return _generate_correlated_trajectories(n_trajs=3, n_frames=600, n_features=12)


@pytest.fixture
def tica_harness(
    synthetic_trajectories: list[np.ndarray], tmp_path: Path
) -> _TICAHarness:
    harness = _TICAHarness(
        synthetic_trajectories, output_dir=tmp_path / "tica_artifacts"
    )
    harness.reset()
    return harness


def test_tica_full_fitting_process(benchmark, tica_harness: _TICAHarness) -> None:
    """Benchmark the full TICA fitting pipeline implemented by the mixin."""

    lag = 10
    hint_components = 4
    expected_components = int(max(2, min(5, hint_components)))
    expected_rows = sum(
        max(0, traj.n_frames - lag) for traj in tica_harness.trajectories
    )

    def _run_tica() -> np.ndarray:
        tica_harness.reset()
        tica_harness.tica_lag = lag
        tica_harness._maybe_apply_tica(hint_components, lag)
        assert tica_harness.features is not None
        return tica_harness.features

    transformed = benchmark(_run_tica)

    assert transformed.shape == (expected_rows, expected_components)
    assert not np.isnan(transformed).any()
    assert getattr(tica_harness, "tica_components_", None) == expected_components


def _build_tica_covariances(
    trajectories: Iterable[np.ndarray], lag: int
) -> tuple[np.ndarray, np.ndarray]:
    """Construct instantaneous and time-lagged covariance matrices."""

    traj_arrays: List[np.ndarray] = [
        np.asarray(traj, dtype=np.float64) for traj in trajectories
    ]
    if not traj_arrays:
        return np.empty((0, 0)), np.empty((0, 0))

    stacked = np.vstack(traj_arrays)
    if stacked.size == 0:
        return np.empty((0, 0)), np.empty((0, 0))

    mean = stacked.mean(axis=0, keepdims=True)
    n_features = stacked.shape[1]
    c00 = np.zeros((n_features, n_features), dtype=np.float64)
    c0t = np.zeros_like(c00)
    sample_count = 0

    for traj in traj_arrays:
        if traj.shape[0] <= lag:
            continue
        x0 = traj[:-lag] - mean
        xt = traj[lag:] - mean
        c00 += x0.T @ x0
        c0t += x0.T @ xt
        sample_count += x0.shape[0]

    if sample_count <= 1:
        return c00, c0t

    denom = float(sample_count - 1)
    return c00 / denom, c0t / denom


def test_tica_covariance_construction(benchmark, synthetic_trajectories):
    """Benchmark construction of lagged and instantaneous covariance matrices."""

    lag = 8

    def _run_covariances() -> tuple[np.ndarray, np.ndarray]:
        return _build_tica_covariances(synthetic_trajectories, lag)

    c00, c0t = benchmark(_run_covariances)

    assert c00.shape == (synthetic_trajectories[0].shape[1],) * 2
    assert c0t.shape == c00.shape
    assert np.allclose(c00, c00.T, atol=1e-9)
    eigvals = np.linalg.eigvalsh(0.5 * (c00 + c00.T))
    assert np.all(eigvals >= -1e-12)
    assert np.linalg.norm(c0t) > 0.0


def test_tica_eigenvalue_solver_benchmark(benchmark: pytest.BenchmarkSession) -> None:
    """Benchmark solving the generalized eigenproblem for TICA components."""

    from pmarlo.features.deeptica.core import trainer_api

    outputs = _generate_correlated_time_series(20_000, 8, seed=21)
    lag = 10
    idx_t = np.arange(0, outputs.shape[0] - lag, dtype=np.int64)
    idx_tau = idx_t + lag
    cfg = SimpleNamespace(n_out=4)

    def _solve() -> list[float] | None:
        return trainer_api._estimate_top_eigenvalues(outputs, idx_t, idx_tau, cfg)

    eigenvalues = benchmark(_solve)
    assert eigenvalues is not None
    eig_arr = np.asarray(eigenvalues, dtype=np.float64)
    assert eig_arr.shape == (min(cfg.n_out, idx_t.size),)
    assert np.all(eig_arr[:-1] >= eig_arr[1:] - 1e-9)
    assert np.all(np.abs(eig_arr) <= 1.0 + 1e-6)


def test_tica_transformation_benchmark(benchmark: pytest.BenchmarkSession) -> None:
    """Benchmark projecting features onto learned TICA components."""

    pytest.importorskip("deeptime", reason="deeptime is required for TICA benchmarks")
    from pmarlo.markov_state_model.reduction import tica_reduce

    features = _generate_correlated_time_series(12_000, 24, seed=7)
    lag = 8
    n_components = 3

    def _transform() -> np.ndarray:
        return tica_reduce(features, lag=lag, n_components=n_components, scale=True)

    projected = benchmark(_transform)
    assert projected.shape == (features.shape[0], n_components)
    assert np.isfinite(projected).all()
