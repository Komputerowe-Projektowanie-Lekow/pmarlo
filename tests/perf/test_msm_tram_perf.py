from __future__ import annotations

"""TRAM MSM construction performance benchmarks.

The Transition-based Reweighting Analysis Method (TRAM) combines data from
multiple thermodynamic ensembles to produce a single unbiased MSM. These
benchmarks construct synthetic multi-temperature trajectories with consistent
bias matrices to exercise the full TRAM pipeline exposed by :class:`TRAMMixin`.

Run with: pytest -m benchmark tests/perf/test_msm_tram_perf.py
"""

import os
from dataclasses import dataclass

import numpy as np
import pytest

pytestmark = [pytest.mark.perf, pytest.mark.benchmark, pytest.mark.msm]

pytest.importorskip("pytest_benchmark", reason="pytest-benchmark not installed")
pytest.importorskip("deeptime", reason="deeptime is required for TRAM benchmarks")

if not os.getenv("PMARLO_RUN_PERF"):
    pytest.skip(
        "perf tests disabled; set PMARLO_RUN_PERF=1 to run", allow_module_level=True
    )


@dataclass
class _TRAMHarness:
    """Lightweight object exposing the attributes expected by :class:`TRAMMixin`."""

    temperatures: list[float]
    dtrajs: list[np.ndarray]
    bias_matrices: list[np.ndarray]
    tram_reference_index: int = 0
    transition_matrix: np.ndarray | None = None
    count_matrix: np.ndarray | None = None
    stationary_distribution: np.ndarray | None = None

    def build(self, lag_time: int) -> "_TRAMHarness":
        from pmarlo.markov_state_model._tram import TRAMMixin

        class _Builder(TRAMMixin):
            pass

        builder = _Builder()
        builder.temperatures = self.temperatures
        builder.dtrajs = [np.asarray(dt, dtype=int) for dt in self.dtrajs]
        builder.bias_matrices = [
            np.asarray(bias, dtype=float) for bias in self.bias_matrices
        ]
        builder.transition_matrix = None
        builder.count_matrix = None
        builder.stationary_distribution = None
        builder.tram_reference_index = int(self.tram_reference_index)
        builder._build_tram_msm(lag_time=lag_time)
        self.transition_matrix = builder.transition_matrix
        self.count_matrix = builder.count_matrix
        self.stationary_distribution = builder.stationary_distribution
        return self


def _synthetic_tram_inputs(
    n_therm_states: int,
    n_markov_states: int,
    trajectory_length: int,
    *,
    seed: int = 11,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Generate discrete trajectories and bias matrices for TRAM benchmarking."""
    rng = np.random.default_rng(seed)
    dtrajs: list[np.ndarray] = []
    bias_matrices: list[np.ndarray] = []

    base_cycle = np.arange(n_markov_states, dtype=int)
    tiled = np.tile(base_cycle, trajectory_length // n_markov_states + 1)

    for therm_state in range(n_therm_states):
        noise = rng.integers(0, n_markov_states, size=trajectory_length)
        dtraj = np.mod(tiled[:trajectory_length] + noise + therm_state, n_markov_states)
        dtrajs.append(dtraj.astype(int))

        bias = np.empty((trajectory_length, n_therm_states), dtype=float)
        for other_state in range(n_therm_states):
            penalty = abs(therm_state - other_state)
            bias[:, other_state] = penalty + 0.2 * rng.normal(size=trajectory_length)
        bias_matrices.append(bias)

    return dtrajs, bias_matrices


def test_tram_constructs_reference_msm(benchmark: pytest.BenchmarkFixture) -> None:
    """Benchmark TRAM MSM construction for a moderate 3x60x6 dataset."""
    n_therm = 3
    n_states = 6
    traj_len = 60
    dtrajs, bias = _synthetic_tram_inputs(n_therm, n_states, traj_len)
    harness = _TRAMHarness(
        temperatures=[290.0, 300.0, 310.0],
        dtrajs=dtrajs,
        bias_matrices=bias,
        tram_reference_index=1,
    )

    def _run() -> _TRAMHarness:
        return _TRAMHarness(
            temperatures=harness.temperatures,
            dtrajs=harness.dtrajs,
            bias_matrices=harness.bias_matrices,
            tram_reference_index=harness.tram_reference_index,
        ).build(lag_time=2)

    built = benchmark(_run)
    assert built.transition_matrix is not None
    assert built.transition_matrix.shape == (n_states, n_states)
    np.testing.assert_allclose(built.transition_matrix.sum(axis=1), 1.0, atol=1e-6)
    assert built.count_matrix is not None
    assert built.count_matrix.shape == (n_states, n_states)
    assert built.stationary_distribution is not None
    np.testing.assert_allclose(built.stationary_distribution.sum(), 1.0, atol=1e-6)


def test_tram_handles_longer_trajectories(benchmark: pytest.BenchmarkFixture) -> None:
    """Benchmark TRAM MSM construction for long trajectories and more thermodynamic states."""
    n_therm = 4
    n_states = 5
    traj_len = 150
    dtrajs, bias = _synthetic_tram_inputs(n_therm, n_states, traj_len, seed=27)
    temperatures = [285.0, 295.0, 305.0, 315.0]

    def _run() -> _TRAMHarness:
        return _TRAMHarness(
            temperatures=temperatures,
            dtrajs=dtrajs,
            bias_matrices=bias,
            tram_reference_index=2,
        ).build(lag_time=3)

    built = benchmark(_run)
    assert built.transition_matrix is not None
    assert built.transition_matrix.shape == (n_states, n_states)
    assert built.count_matrix is not None
    assert built.count_matrix.shape == (n_states, n_states)
    assert built.stationary_distribution is not None
    assert np.all(built.stationary_distribution >= 0.0)
    np.testing.assert_allclose(built.stationary_distribution.sum(), 1.0, atol=1e-6)
