from __future__ import annotations

"""Performance benchmarks for CK diagnostics and stationary distribution."""

import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from pmarlo.markov_state_model._base import CKTestResult

pytestmark = [pytest.mark.perf, pytest.mark.benchmark, pytest.mark.msm]

pytest.importorskip("pytest_benchmark", reason="pytest-benchmark not installed")

if not os.getenv("PMARLO_RUN_PERF"):
    pytest.skip(
        "perf tests disabled; set PMARLO_RUN_PERF=1 to run", allow_module_level=True
    )

if TYPE_CHECKING:
    from pmarlo.markov_state_model.enhanced_msm import EnhancedMSM


def _build_cycle_msm(tmp_path: Path) -> "EnhancedMSM":
    """Create a simple three-state MSM cycling 0â†’1â†’2â†’0."""
    from pmarlo.markov_state_model.enhanced_msm import EnhancedMSM

    traj = np.tile(np.array([0, 1, 2], dtype=int), 1_000)
    msm = EnhancedMSM(output_dir=str(tmp_path))
    msm.dtrajs = [traj]
    msm.n_states = 3
    msm.lag_time = 1

    counts = np.zeros((3, 3), dtype=float)
    for i in range(traj.size - 1):
        counts[traj[i], traj[i + 1]] += 1.0
    row_sums = counts.sum(axis=1)
    row_sums[row_sums == 0] = 1.0
    msm.transition_matrix = counts / row_sums[:, None]
    return msm


def test_ck_micro_benchmark(tmp_path: Path, benchmark: pytest.BenchmarkFixture) -> None:
    """Benchmark the Chapmanâ€“Kolmogorov microstate test."""
    msm = _build_cycle_msm(tmp_path)

    def _compute() -> CKTestResult:
        return msm.compute_ck_test_micro()

    result = benchmark(_compute)
    assert not result.insufficient_data
    assert set(result.mse) == {2, 3, 4, 5}
    assert all(value >= 0.0 for value in result.mse.values())
    assert result.mse[2] < 1e-6


def _random_transition_matrix(n_states: int, seed: int = 123) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mat = rng.random((n_states, n_states), dtype=float)
    row_sums = mat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0
    return mat / row_sums


def test_stationary_distribution_benchmark(benchmark: pytest.BenchmarkFixture) -> None:
    """Benchmark stationary distribution extraction from a dense MSM."""
    from pmarlo.markov_state_model._msm_utils import _stationary_from_T

    transition_matrix = _random_transition_matrix(75)

    def _compute() -> np.ndarray:
        return _stationary_from_T(transition_matrix)

    stationary = benchmark(_compute)
    assert stationary.shape == (transition_matrix.shape[0],)
    np.testing.assert_allclose(stationary.sum(), 1.0, atol=1e-12)
    np.testing.assert_array_less(-1e-12, stationary)
    np.testing.assert_allclose(
        transition_matrix.T @ stationary,
        stationary,
        atol=1e-7,
    )
