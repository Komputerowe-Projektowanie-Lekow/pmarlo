from __future__ import annotations

import numpy as np
import pytest

from pmarlo.markov_state_model.ck_its_selector import (
    _auto_determine_macrostates,
    _check_sanity_criteria,
    _compute_ck_error,
    _compute_coverage_fraction,
    _compute_median_count,
    _compute_observed_macro_kinetics,
    _compute_predicted_macro_kinetics,
    _count_transitions,
    _evaluate_single_lag,
    select_optimal_lag_ck_its,
)


@pytest.fixture
def simple_dtrajs():
    """Simple discrete trajectories for testing."""
    # Two trajectories with 5 states each
    return [
        np.array([0, 1, 2, 1, 0, 1, 2, 3, 2, 1], dtype=int),
        np.array([1, 2, 3, 2, 1, 0, 1, 2, 1, 0], dtype=int),
    ]


def test_count_transitions(simple_dtrajs):
    """Test transition counting at different lags."""
    C = _count_transitions(simple_dtrajs, n_states=4, lag=1)

    assert C.shape == (4, 4)
    assert np.all(C >= 0)
    # Should have some transitions
    assert np.sum(C) > 0


def test_count_transitions_lag_too_long(simple_dtrajs):
    """Test that long lag returns zero transitions."""
    C = _count_transitions(simple_dtrajs, n_states=4, lag=100)

    assert C.shape == (4, 4)
    assert np.sum(C) == 0


def test_compute_coverage_fraction():
    """Test coverage fraction computation."""
    # Fully connected 3x3 matrix
    C_full = np.ones((3, 3), dtype=float)
    coverage_full = _compute_coverage_fraction(C_full)
    assert coverage_full == pytest.approx(1.0)

    # Partially connected (state 2 isolated)
    C_partial = np.array(
        [
            [1, 1, 0],
            [1, 1, 0],
            [0, 0, 0],
        ],
        dtype=float,
    )
    coverage_partial = _compute_coverage_fraction(C_partial)
    assert coverage_partial == pytest.approx(2.0 / 3.0)

    # Empty matrix
    C_empty = np.zeros((0, 0), dtype=float)
    coverage_empty = _compute_coverage_fraction(C_empty)
    assert coverage_empty == 0.0


def test_compute_median_count():
    """Test median count computation."""
    C = np.array(
        [
            [10, 5, 0],
            [5, 20, 3],
            [0, 3, 15],
        ],
        dtype=float,
    )

    median = _compute_median_count(C)
    # State counts (incoming + outgoing): [30, 56, 36] -> median = 36
    assert median == 36

    # Empty matrix
    C_empty = np.zeros((0, 0), dtype=float)
    median_empty = _compute_median_count(C_empty)
    assert median_empty == 0


def test_auto_determine_macrostates():
    """Test automatic macrostate determination via eigenvalue gap."""
    # Simple 5x5 transition matrix with clear structure
    T = np.array(
        [
            [0.8, 0.2, 0.0, 0.0, 0.0],
            [0.2, 0.7, 0.1, 0.0, 0.0],
            [0.0, 0.1, 0.7, 0.2, 0.0],
            [0.0, 0.0, 0.2, 0.6, 0.2],
            [0.0, 0.0, 0.0, 0.2, 0.8],
        ],
        dtype=float,
    )

    n_macro = _auto_determine_macrostates(T, min_macro=2, max_macro=4)

    # Should be between min and max
    assert 2 <= n_macro <= 4

    # Too small matrix
    T_small = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float)
    n_macro_small = _auto_determine_macrostates(T_small, min_macro=3, max_macro=5)
    assert n_macro_small == 3  # Falls back to min_macro


def test_compute_predicted_macro_kinetics():
    """Test predicted macrostate kinetics computation."""
    # Simple 4-state system
    T_tau = np.array(
        [
            [0.7, 0.3, 0.0, 0.0],
            [0.3, 0.5, 0.2, 0.0],
            [0.0, 0.2, 0.6, 0.2],
            [0.0, 0.0, 0.2, 0.8],
        ],
        dtype=float,
    )

    pi = np.array([0.3, 0.3, 0.2, 0.2], dtype=float)

    # Two macrostates: {0,1} and {2,3}
    chi = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=float,
    )

    T_macro = _compute_predicted_macro_kinetics(T_tau, pi, chi, k=1)

    assert T_macro.shape == (2, 2)
    assert np.all(T_macro >= 0)
    assert np.all(T_macro <= 1)
    # Rows should sum to ~1
    assert np.allclose(T_macro.sum(axis=1), 1.0, atol=0.1)


def test_compute_observed_macro_kinetics(simple_dtrajs):
    """Test observed macrostate kinetics computation."""
    # Map states to macrostates
    macro_labels = np.array([0, 0, 1, 1], dtype=int)

    T_obs = _compute_observed_macro_kinetics(
        simple_dtrajs, macro_labels, n_macro=2, lag=1
    )

    assert T_obs.shape == (2, 2)
    assert np.all(T_obs >= 0)
    assert np.all(T_obs <= 1)
    # Rows should sum to ~1
    row_sums = T_obs.sum(axis=1)
    valid_rows = row_sums > 0
    assert np.allclose(row_sums[valid_rows], 1.0)


def test_compute_ck_error():
    """Test CK error computation."""
    T_pred = np.array([[0.8, 0.2], [0.3, 0.7]], dtype=float)
    T_obs = np.array([[0.75, 0.25], [0.35, 0.65]], dtype=float)

    error = _compute_ck_error(T_pred, T_obs)

    assert error > 0
    assert np.isfinite(error)

    # Identical matrices should have zero error
    error_zero = _compute_ck_error(T_pred, T_pred)
    assert error_zero == pytest.approx(0.0, abs=1e-10)


def test_compute_ck_error_shape_mismatch():
    """Test that shape mismatch raises error."""
    T_pred = np.array([[0.8, 0.2], [0.3, 0.7]], dtype=float)
    T_obs = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=float)

    with pytest.raises(ValueError, match="shape mismatch"):
        _compute_ck_error(T_pred, T_obs)


def test_check_sanity_criteria():
    """Test sanity criteria checking."""
    # Pass both criteria
    passed, reason = _check_sanity_criteria(
        coverage=0.99, median_count=150, coverage_threshold=0.98, min_median_count=100
    )
    assert passed is True
    assert reason is None

    # Fail coverage
    passed, reason = _check_sanity_criteria(
        coverage=0.95, median_count=150, coverage_threshold=0.98, min_median_count=100
    )
    assert passed is False
    assert "Coverage" in reason

    # Fail median count
    passed, reason = _check_sanity_criteria(
        coverage=0.99, median_count=50, coverage_threshold=0.98, min_median_count=100
    )
    assert passed is False
    assert "Median count" in reason


def test_evaluate_single_lag(simple_dtrajs):
    """Test evaluation of a single lag time."""
    result = _evaluate_single_lag(
        dtrajs=simple_dtrajs,
        lag=1,
        horizons=[1, 2],
        n_states=4,
        coverage_threshold=0.5,  # Relaxed for test
        min_median_count=1,  # Relaxed for test
        diag_mass_threshold=0.0,
    )

    assert result.lag == 1
    assert np.isfinite(result.ck_error)
    assert 0.0 <= result.coverage_fraction <= 1.0
    assert result.median_count >= 0
    assert result.n_macrostates >= 0
    assert result.n_microstates == 4
    if result.diag_mass is not None and np.isfinite(result.diag_mass):
        assert 0.0 <= result.diag_mass <= 1.0


def test_select_optimal_lag_ck_its():
    """Test full optimal lag selection."""
    # Generate longer trajectories for realistic test
    np.random.seed(42)
    n_states = 10
    traj_length = 200

    # Generate Markov chain trajectories
    dtrajs = []
    for _ in range(3):
        traj = [np.random.randint(0, n_states)]
        for _ in range(traj_length - 1):
            # Simple random walk
            current = traj[-1]
            next_state = (current + np.random.choice([-1, 0, 1])) % n_states
            traj.append(next_state)
        dtrajs.append(np.array(traj, dtype=int))

    selected_lag, evaluations = select_optimal_lag_ck_its(
        dtrajs=dtrajs,
        tau_candidates=[5, 10, 15],
        horizons=[1, 2],
        ck_threshold=0.5,  # Relaxed for test
        coverage_threshold=0.5,
        min_median_count=10,
        diag_mass_threshold=0.0,
    )

    assert selected_lag in [5, 10, 15]
    assert len(evaluations) == 3

    # Check that evaluations are sorted by lag
    lags = [e.lag for e in evaluations]
    assert lags == sorted(lags)

    # Each evaluation should have valid data
    for ev in evaluations:
        assert ev.lag in [5, 10, 15]
        assert np.isfinite(ev.ck_error) or ev.ck_error == float("inf")
        assert 0.0 <= ev.coverage_fraction <= 1.0


def test_select_optimal_lag_empty_dtrajs():
    """Test that empty trajectories raise error."""
    with pytest.raises(ValueError, match="No discrete trajectories"):
        select_optimal_lag_ck_its(dtrajs=[])


def test_select_optimal_lag_fallback():
    """Test fallback behavior when no lag passes criteria."""
    # Very strict criteria that likely won't pass
    np.random.seed(42)
    short_traj = np.array([0, 1, 2, 1, 0], dtype=int)

    selected_lag, evaluations = select_optimal_lag_ck_its(
        dtrajs=[short_traj],
        tau_candidates=[1, 2],
        horizons=[1],
        ck_threshold=0.001,  # Very strict
        coverage_threshold=0.99,
        min_median_count=1000,  # Very high
        diag_mass_threshold=0.0,
    )

    # Should still select something (fallback to smallest)
    assert selected_lag in [1, 2]
    assert len(evaluations) == 2


def test_select_optimal_lag_filters_long_candidates(simple_dtrajs):
    """Ensure candidates longer than the trajectories are ignored."""
    selected_lag, evaluations = select_optimal_lag_ck_its(
        dtrajs=simple_dtrajs,
        tau_candidates=[1, 5, 500],
        horizons=[1],
        ck_threshold=1.0,
        coverage_threshold=0.3,
        min_median_count=1,
        diag_mass_threshold=0.0,
    )

    lags = [ev.lag for ev in evaluations]
    assert 500 not in lags
    assert selected_lag in lags


def test_select_optimal_lag_all_candidates_filtered(simple_dtrajs):
    """Raise when every candidate exceeds the supported lag."""
    max_len = max(traj.size for traj in simple_dtrajs)
    too_large = max_len + 10

    with pytest.raises(ValueError, match="max supported lag"):
        select_optimal_lag_ck_its(
            dtrajs=simple_dtrajs,
            tau_candidates=[too_large],
            horizons=[1],
            ck_threshold=1.0,
            coverage_threshold=0.3,
            min_median_count=1,
            diag_mass_threshold=0.0,
        )
