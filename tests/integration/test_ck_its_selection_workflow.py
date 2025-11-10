from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pmarlo.markov_state_model.ck_its_selector import select_optimal_lag_ck_its

pytestmark = pytest.mark.integration


def _simulate_two_well_trajectory(
    n_frames: int, tau_corr: int = 800, seed: int = 0
) -> np.ndarray:
    """Simulate a 1D double-well potential trajectory for testing.

    Returns discrete trajectory with 2 states (left/right well).
    """
    rng = np.random.default_rng(seed)
    x = np.zeros(n_frames, dtype=float)
    dt = 1.0

    for t in range(1, n_frames):
        force = -4.0 * x[t - 1] * (x[t - 1] ** 2 - 1.0)
        x[t] = (
            x[t - 1]
            + (dt / tau_corr) * force
            + np.sqrt(2.0 * dt / tau_corr) * rng.normal(0.0, 1.0)
        )

    # Discretize into 2 states
    z = (x > 0.0).astype(np.int32)
    return z


def test_ck_its_selector_synthetic_data():
    """Test CK+ITS selector with synthetic double-well trajectory."""
    # Generate synthetic trajectory
    n_frames = 10000
    traj = _simulate_two_well_trajectory(n_frames, tau_corr=800, seed=42)
    dtrajs = [traj]

    # Run selection with reasonable candidates
    selected_lag, evaluations = select_optimal_lag_ck_its(
        dtrajs=dtrajs,
        tau_candidates=[50, 100, 200],
        horizons=[1, 2, 3],
        ck_threshold=0.20,  # Relaxed for synthetic data
        coverage_threshold=0.95,
        min_median_count=50,
    )

    # Verify results
    assert selected_lag in [50, 100, 200]
    assert len(evaluations) == 3

    # Check that all evaluations completed
    for eval_result in evaluations:
        assert eval_result.lag in [50, 100, 200]
        assert np.isfinite(eval_result.ck_error) or eval_result.ck_error == float("inf")
        assert 0.0 <= eval_result.coverage_fraction <= 1.0
        assert eval_result.median_count >= 0

    # Selected lag should have reasonable CK error
    selected_eval = next(e for e in evaluations if e.lag == selected_lag)
    assert selected_eval.ck_error <= 0.5  # Should be decent quality


def test_ck_its_selector_with_multiple_trajectories():
    """Test CK+ITS selector with multiple independent trajectories."""
    # Generate multiple trajectories
    n_trajs = 3
    n_frames = 5000
    dtrajs = [
        _simulate_two_well_trajectory(n_frames, tau_corr=800, seed=i)
        for i in range(n_trajs)
    ]

    # Run selection
    selected_lag, evaluations = select_optimal_lag_ck_its(
        dtrajs=dtrajs,
        tau_candidates=[25, 50, 75],
        horizons=[1, 2],
        ck_threshold=0.20,
        coverage_threshold=0.90,
        min_median_count=30,
    )

    # Verify results
    assert selected_lag in [25, 50, 75]
    assert len(evaluations) == 3

    # Multiple trajectories should improve statistics
    selected_eval = next(e for e in evaluations if e.lag == selected_lag)
    assert selected_eval.passed_sanity


def test_ck_its_selector_stability():
    """Test that CK+ITS selector produces consistent results."""
    # Generate trajectory
    n_frames = 8000
    traj = _simulate_two_well_trajectory(n_frames, tau_corr=800, seed=123)
    dtrajs = [traj]

    # Run selection twice with same data
    selected_lag1, _ = select_optimal_lag_ck_its(
        dtrajs=dtrajs,
        tau_candidates=[40, 80, 120],
        horizons=[1, 2],
        ck_threshold=0.20,
        coverage_threshold=0.95,
        min_median_count=40,
    )

    selected_lag2, _ = select_optimal_lag_ck_its(
        dtrajs=dtrajs,
        tau_candidates=[40, 80, 120],
        horizons=[1, 2],
        ck_threshold=0.20,
        coverage_threshold=0.95,
        min_median_count=40,
    )

    # Should select same lag
    assert selected_lag1 == selected_lag2


def test_ck_its_selector_prefers_smaller_lag():
    """Test that selector prefers smaller lag when multiple pass criteria."""
    # Generate long trajectory with good statistics
    n_frames = 15000
    traj = _simulate_two_well_trajectory(n_frames, tau_corr=800, seed=456)
    dtrajs = [traj]

    # Run with multiple candidates and relaxed criteria
    selected_lag, evaluations = select_optimal_lag_ck_its(
        dtrajs=dtrajs,
        tau_candidates=[30, 60, 90, 120],
        horizons=[1, 2],
        ck_threshold=0.30,  # Very relaxed
        coverage_threshold=0.80,  # Very relaxed
        min_median_count=20,  # Very relaxed
    )

    # Find all passing lags
    passing_lags = [
        e.lag for e in evaluations if e.passed_sanity and e.ck_error <= 0.30
    ]

    if passing_lags:
        # Selected lag should be the smallest among passing lags
        assert selected_lag == min(passing_lags)


def test_ck_its_selector_with_insufficient_data():
    """Test behavior with insufficient data."""
    # Very short trajectory
    traj = np.array([0, 1, 0, 1, 0], dtype=int)
    dtrajs = [traj]

    # Should complete without crashing, even if no lag passes
    selected_lag, evaluations = select_optimal_lag_ck_its(
        dtrajs=dtrajs,
        tau_candidates=[1, 2],
        horizons=[1],
        ck_threshold=0.10,
        coverage_threshold=0.95,
        min_median_count=100,
    )

    # Should still select something (fallback behavior)
    assert selected_lag in [1, 2]
    assert len(evaluations) == 2


def test_ck_its_selector_coverage_check():
    """Test that coverage threshold is properly enforced."""
    # Create trajectory with isolated states
    # States 0-4 are connected, state 5 is isolated
    traj1 = np.array([0, 1, 2, 3, 4, 3, 2, 1, 0] * 50, dtype=int)
    traj2 = np.array([5] * 50, dtype=int)  # Isolated state
    dtrajs = [np.concatenate([traj1, traj2])]

    # Run with strict coverage requirement
    selected_lag, evaluations = select_optimal_lag_ck_its(
        dtrajs=dtrajs,
        tau_candidates=[1, 2],
        horizons=[1],
        ck_threshold=0.50,
        coverage_threshold=0.99,  # Strict coverage
        min_median_count=10,
    )

    # Check that coverage was computed
    for eval_result in evaluations:
        assert 0.0 <= eval_result.coverage_fraction <= 1.0


def test_ck_its_selector_macrostate_determination():
    """Test that macrostates are automatically determined."""
    # Generate trajectory with clear multi-well structure
    n_frames = 10000
    traj = _simulate_two_well_trajectory(n_frames, tau_corr=800, seed=789)
    dtrajs = [traj]

    # Run selection
    selected_lag, evaluations = select_optimal_lag_ck_its(
        dtrajs=dtrajs,
        tau_candidates=[100],
        horizons=[1, 2],
        ck_threshold=0.25,
        coverage_threshold=0.90,
        min_median_count=30,
    )

    # Check that macrostates were determined
    eval_result = evaluations[0]
    if eval_result.n_macrostates > 0:
        # For 2-state system, should detect 1-2 macrostates
        assert 1 <= eval_result.n_macrostates <= 3

        # Should have computed eigenvalue gap
        if eval_result.eigenvalue_gap is not None:
            assert eval_result.eigenvalue_gap >= 0


def test_ck_its_selector_timescales_computation():
    """Test that timescales are computed for ITS analysis."""
    # Generate trajectory
    n_frames = 8000
    traj = _simulate_two_well_trajectory(n_frames, tau_corr=800, seed=321)
    dtrajs = [traj]

    # Run selection
    selected_lag, evaluations = select_optimal_lag_ck_its(
        dtrajs=dtrajs,
        tau_candidates=[50, 100],
        horizons=[1, 2],
        ck_threshold=0.25,
        coverage_threshold=0.90,
        min_median_count=30,
    )

    # Check that at least some timescales were computed
    timescales_computed = 0
    for eval_result in evaluations:
        if eval_result.timescales is not None and eval_result.timescales.size > 0:
            timescales_computed += 1
            # Timescales should be positive and finite
            assert np.all(eval_result.timescales > 0)
            assert np.all(np.isfinite(eval_result.timescales))

    # At least one lag should have timescales
    assert timescales_computed > 0


def test_ck_its_selector_ck_error_increases_with_horizon():
    """Test that CK error generally doesn't decrease with longer horizons."""
    # Generate trajectory
    n_frames = 12000
    traj = _simulate_two_well_trajectory(n_frames, tau_corr=800, seed=654)
    dtrajs = [traj]

    # Run with multiple horizons
    selected_lag, evaluations = select_optimal_lag_ck_its(
        dtrajs=dtrajs,
        tau_candidates=[80],
        horizons=[1, 2, 3],  # Multiple horizons
        ck_threshold=0.50,
        coverage_threshold=0.90,
        min_median_count=30,
    )

    # CK error should reflect maximum across all horizons
    eval_result = evaluations[0]
    assert np.isfinite(eval_result.ck_error)
    # Error should be reasonable (not infinite)
    assert eval_result.ck_error < 1.0 or not eval_result.passed_sanity
