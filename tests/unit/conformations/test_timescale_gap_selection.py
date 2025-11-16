"""Tests for selecting k_slow via implied timescale gaps."""

import numpy as np
import pytest

from pmarlo.conformations.kinetic_importance import KineticImportanceScore


def make_dummy_kis(n_states: int = 4) -> KineticImportanceScore:
    """Build a simple MSM just to access the selection method."""
    T = np.full((n_states, n_states), 1.0 / n_states, dtype=float)
    pi = np.full(n_states, 1.0 / n_states, dtype=float)
    return KineticImportanceScore(T=T, pi=pi)


def test_timescale_gap_clear_large_gap_selects_modes_before_gap():
    """When there is a single strong gap in timescales we cut at that gap."""
    kis = make_dummy_kis()

    its = np.array([100.0, 50.0, 5.0, 4.0, 3.0], dtype=float)
    gap_threshold = 5.0

    k_slow = kis._select_by_timescale_gap(its, gap_threshold=gap_threshold)

    assert k_slow == 2


def test_timescale_gap_no_gap_uses_small_default_number():
    """Smooth decays without a gap should use the bounded default."""
    kis = make_dummy_kis()

    its = np.array([100.0, 50.0, 25.0, 12.5, 6.25, 3.125], dtype=float)
    gap_threshold = 10.0

    k_slow = kis._select_by_timescale_gap(its, gap_threshold=gap_threshold)

    assert k_slow == 5


def test_timescale_gap_threshold_controls_detection():
    """The threshold parameter must influence gap detection."""
    kis = make_dummy_kis()
    its = np.array([100.0, 50.0, 10.0, 9.0, 8.0], dtype=float)

    k_slow_low = kis._select_by_timescale_gap(its, gap_threshold=3.0)
    assert k_slow_low == 2

    k_slow_high = kis._select_by_timescale_gap(its, gap_threshold=10.0)
    assert k_slow_high == 5


@pytest.mark.parametrize(
    "timescales",
    [
        np.array([], dtype=float),
        np.array([100.0], dtype=float),
    ],
)
def test_timescale_gap_short_input_respects_minimum_two_modes(timescales: np.ndarray):
    """If there are too few valid timescales we still return at least two."""
    kis = make_dummy_kis()

    k_slow = kis._select_by_timescale_gap(timescales, gap_threshold=5.0)

    assert k_slow == 2


def test_timescale_gap_ignores_non_positive_and_nan_timescales():
    """Nonphysical entries should not influence the selection logic."""
    kis = make_dummy_kis()

    its = np.array(
        [100.0, 50.0, np.nan, -10.0, 10.0, 0.0, np.inf],
        dtype=float,
    )

    gap_threshold = 3.0

    k_slow = kis._select_by_timescale_gap(its, gap_threshold=gap_threshold)

    assert k_slow == 2


def test_timescale_gap_all_invalid_returns_minimum_two():
    """If everything is invalid we fall back to the minimum safe value."""
    kis = make_dummy_kis()

    its = np.array([np.nan, -1.0, 0.0, np.inf], dtype=float)

    k_slow = kis._select_by_timescale_gap(its, gap_threshold=5.0)
    assert k_slow == 2
