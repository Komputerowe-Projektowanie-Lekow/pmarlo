"""Performance benchmarks for MSM estimation and ITS calculations."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List

import numpy as np
import pytest

from pmarlo import constants as const
from pmarlo.utils.msm_utils import ensure_connected_counts

pytestmark = [pytest.mark.perf, pytest.mark.benchmark, pytest.mark.msm]

# Optional plugin
pytest.importorskip("pytest_benchmark", reason="pytest-benchmark not installed")

if not os.getenv("PMARLO_RUN_PERF"):
    pytest.skip(
        "perf tests disabled; set PMARLO_RUN_PERF=1 to run", allow_module_level=True
    )

if TYPE_CHECKING:
    from pmarlo.markov_state_model._estimation import EstimationMixin
    from pmarlo.markov_state_model._its import ITSMixin


@dataclass
class _ToyMSMState:
    dtrajs: List[np.ndarray]
    n_states: int
    count_mode: str


class _ToyMSMEstimator:
    """Lightweight harness around :class:`EstimationMixin` for benchmarks."""

    def __init__(self, state: _ToyMSMState, *, output_dir: str):
        from pmarlo.markov_state_model._estimation import EstimationMixin

        class _Estimator(EstimationMixin):
            def __init__(self, outer_state: _ToyMSMState, *, output_dir: str):
                self.features = None
                self.dtrajs = [np.asarray(dt, dtype=int) for dt in outer_state.dtrajs]
                self.n_states = int(outer_state.n_states)
                self.count_mode = str(outer_state.count_mode)
                self.effective_frames = None
                self.output_dir = Path(output_dir)
                self.output_dir.mkdir(parents=True, exist_ok=True)
                self.tica_lag = 0
                self.stationary_distribution = None
                self.free_energies = None
                self.lag_time = 0
                self.count_matrix = None
                self.transition_matrix = None
                self.estimator_backend = "local"

            # Protocol hooks that are unused in this benchmark
            def _maybe_apply_tica(
                self, n_components: int, lag: int
            ) -> None:  # pragma: no cover - perf harness
                return None

            def _build_tram_msm(
                self, lag_time: int
            ) -> None:  # pragma: no cover - perf harness
                raise NotImplementedError("TRAM method not supported in toy benchmark")

            def _should_use_deeptime(self) -> bool:
                return False

        self._impl = _Estimator(state, output_dir=output_dir)

    def run(self, *, lag_time: int) -> "EstimationMixin":
        self._impl.build_msm(lag_time=lag_time, method="standard")
        return self._impl


class _ToyITSEngine:
    """Deterministic harness around :class:`ITSMixin` for benchmarking ITS."""

    def __init__(
        self, counts_by_lag: Dict[int, np.ndarray], *, dtrajs: List[np.ndarray]
    ):
        from pmarlo.markov_state_model._its import ITSMixin

        class _Engine(ITSMixin):
            def __init__(
                self, counts: Dict[int, np.ndarray], trajectories: List[np.ndarray]
            ):
                self.random_state = 13
                self.dtrajs = [np.asarray(dt, dtype=int) for dt in trajectories]
                self.count_mode = "sliding"
                self.lag_time = min(counts)
                self.n_states = int(next(iter(counts.values())).shape[0])
                self.time_per_frame_ps = 1.0
                self.implied_timescales = None
                self.counts_by_lag = {
                    int(k): np.asarray(v, dtype=float) for k, v in counts.items()
                }
                self.effective_frames = None

            def _counts_from_deeptime_backend(self, lag: int) -> np.ndarray:
                key = int(lag)
                if key not in self.counts_by_lag:
                    raise KeyError(f"No synthetic counts prepared for lag {lag}")
                return self.counts_by_lag[key]

        self._impl = _Engine(counts_by_lag, dtrajs)

    def run(
        self,
        *,
        lag_times: List[int],
        n_timescales: int,
        n_samples: int,
        ci: float,
        dirichlet_alpha: float,
    ) -> "ITSMixin":
        self._impl.compute_implied_timescales(
            lag_times=lag_times,
            n_timescales=n_timescales,
            n_samples=n_samples,
            ci=ci,
            dirichlet_alpha=dirichlet_alpha,
        )
        return self._impl


@pytest.fixture
def synthetic_state() -> _ToyMSMState:
    rng = np.random.default_rng(7)
    traj_a = rng.integers(0, 3, size=800, dtype=int)
    traj_b = rng.integers(0, 3, size=800, dtype=int)
    return _ToyMSMState(dtrajs=[traj_a, traj_b], n_states=3, count_mode="sliding")


def _expected_transition_matrix(
    dtrajs: List[np.ndarray], n_states: int, *, lag: int, mode: str
) -> np.ndarray:
    counts = np.zeros((n_states, n_states), dtype=float)
    step = lag if mode == "strided" else 1
    for dtraj in dtrajs:
        arr = np.asarray(dtraj, dtype=int)
        if arr.size <= lag:
            continue
        i_states = arr[:-lag:step]
        j_states = arr[lag::step]
        valid = (
            (i_states >= 0)
            & (j_states >= 0)
            & (i_states < n_states)
            & (j_states < n_states)
        )
        if not np.any(valid):
            continue
        np.add.at(counts, (i_states[valid], j_states[valid]), 1.0)
    res = ensure_connected_counts(
        counts,
        alpha=const.NUMERIC_DIRICHLET_ALPHA,
        epsilon=const.NUMERIC_MIN_POSITIVE,
    )
    transitions = np.eye(n_states, dtype=float)
    if res.counts.size:
        row_sums = res.counts.sum(axis=1, keepdims=True)
        T_active = np.divide(res.counts, row_sums, where=row_sums > 0.0)
        transitions[np.ix_(res.active, res.active)] = T_active
    return transitions


def test_standard_msm_transition_estimation(benchmark, synthetic_state, tmp_path):
    """Benchmark standard MSM estimation using synthetic trajectories."""

    def _build():
        estimator = _ToyMSMEstimator(
            synthetic_state, output_dir=str(tmp_path / "estimation")
        )
        return estimator.run(lag_time=1)

    result = benchmark(_build)

    expected_T = _expected_transition_matrix(
        synthetic_state.dtrajs, synthetic_state.n_states, lag=1, mode="sliding"
    )

    assert result.transition_matrix is not None
    assert np.allclose(result.transition_matrix, expected_T, atol=1e-6)
    assert result.stationary_distribution is not None
    assert np.isclose(np.sum(result.stationary_distribution), 1.0, atol=1e-6)
    assert np.all(result.stationary_distribution >= 0.0)
    assert result.free_energies is not None
    assert np.isclose(float(np.min(result.free_energies)), 0.0, atol=1e-8)


def test_strided_msm_transition_estimation(benchmark, tmp_path):
    """Benchmark strided MSM estimation to validate lag-aware counting."""

    dtrajs = [np.array([0, 1, 2, 0, 1, 2, 0, 1, 2], dtype=int)]
    state = _ToyMSMState(dtrajs=dtrajs, n_states=3, count_mode="strided")

    def _build():
        estimator = _ToyMSMEstimator(state, output_dir=str(tmp_path / "strided"))
        return estimator.run(lag_time=2)

    result = benchmark(_build)

    expected_T = _expected_transition_matrix(dtrajs, 3, lag=2, mode="strided")
    assert np.allclose(result.transition_matrix, expected_T, atol=1e-6)


def test_its_timescale_computation(benchmark):
    """Benchmark ITS computation on a well-behaved synthetic MSM."""

    counts = np.array(
        [
            [400.0, 50.0, 10.0],
            [30.0, 360.0, 30.0],
            [20.0, 40.0, 300.0],
        ],
        dtype=float,
    )
    lag = 5
    counts_by_lag = {lag: counts}
    dtrajs = [np.tile(np.array([0, 1, 2, 1, 0], dtype=int), 200)]

    def _compute():
        engine = _ToyITSEngine(counts_by_lag, dtrajs=dtrajs)
        return engine.run(
            lag_times=[lag],
            n_timescales=2,
            n_samples=128,
            ci=0.9,
            dirichlet_alpha=1.0,
        )

    result = benchmark(_compute)

    its_result = result.implied_timescales
    assert its_result is not None
    assert its_result.timescales.shape == (1, 2)
    assert its_result.eigenvalues.shape == (1, 2)

    transition_matrix = counts / counts.sum(axis=1, keepdims=True)
    evals, _ = np.linalg.eig(transition_matrix.T)
    evals = np.sort(np.real(evals))[::-1]
    expected_eigs = evals[1:3]
    expected_ts = -lag / np.log(np.clip(expected_eigs, 1e-12, 1 - 1e-12))

    assert np.allclose(its_result.eigenvalues[0, :2], expected_eigs, atol=0.05)
    assert np.allclose(its_result.timescales[0, :2], expected_ts, rtol=0.1)
    assert np.all(np.isfinite(its_result.rates[0, :2]))
    assert np.all(
        its_result.timescales_ci[0, :, 0] <= its_result.timescales_ci[0, :, 1]
    )
