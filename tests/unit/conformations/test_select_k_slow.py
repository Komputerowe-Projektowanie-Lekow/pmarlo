"""Behavioral contract tests for select_k_slow."""

import numpy as np
import pytest

from pmarlo.conformations.kinetic_importance import KineticImportanceScore


class DummySelector:
    """Minimal stub harness to exercise select_k_slow in isolation."""

    select_k_slow = KineticImportanceScore.select_k_slow

    def __init__(self, n_states, timescale_result=None, variance_result=None):
        self.n_states = n_states
        self._timescale_result = timescale_result
        self._variance_result = variance_result
        self.last_timescale_args = None
        self.timescale_called = False
        self.variance_called = False

    def _select_by_timescale_gap(self, its, gap_threshold):
        """Stubbed timescale-gap selector."""
        self.timescale_called = True
        self.last_timescale_args = (np.array(its, copy=True), gap_threshold)
        return self._timescale_result

    def _select_by_variance_explained(self):
        """Stubbed variance selector."""
        self.variance_called = True
        return self._variance_result


def test_timescale_gap_forwards_result_and_clamps_to_n_states():
    """timescale_gap should forward arguments/results and clamp to [2, n_states]."""
    its = np.array([10.0, 5.0, 2.5])
    gap_threshold = 2.0

    sel = DummySelector(n_states=10, timescale_result=4)
    k = sel.select_k_slow(its=its, method="timescale_gap", gap_threshold=gap_threshold)
    assert k == 4
    assert sel.timescale_called is True
    np.testing.assert_allclose(sel.last_timescale_args[0], its)
    assert sel.last_timescale_args[1] == gap_threshold

    sel = DummySelector(n_states=5, timescale_result=10)
    k = sel.select_k_slow(its=its, method="timescale_gap", gap_threshold=gap_threshold)
    assert k == 5

    sel = DummySelector(n_states=5, timescale_result=1)
    k = sel.select_k_slow(its=its, method="timescale_gap", gap_threshold=gap_threshold)
    assert k == 2


def test_variance_method_forwards_result_and_clamps():
    """variance method should call stub and clamp results."""
    sel = DummySelector(n_states=10, variance_result=3)
    k = sel.select_k_slow(method="variance")
    assert sel.variance_called is True
    assert k == 3

    sel = DummySelector(n_states=4, variance_result=10)
    k = sel.select_k_slow(method="variance")
    assert k == 4

    sel = DummySelector(n_states=4, variance_result=1)
    k = sel.select_k_slow(method="variance")
    assert k == 2


@pytest.mark.parametrize(
    ("n_states", "expected"),
    [
        (10, 2),
        (3, 2),
        (20, 2),
        (30, 3),
        (40, 4),
        (50, 5),
        (100, 5),
    ],
)
def test_timescale_gap_without_valid_its_uses_fallback_heuristic(n_states, expected):
    """If ITS insufficient, method should use fallback heuristic rather than stub."""
    sel = DummySelector(n_states=n_states, timescale_result=999)
    k = sel.select_k_slow(its=None, method="timescale_gap")
    assert k == expected
    assert sel.timescale_called is False

    sel = DummySelector(n_states=n_states, timescale_result=999)
    k = sel.select_k_slow(its=np.array([1.0]), method="timescale_gap")
    assert k == expected
    assert sel.timescale_called is False


def test_unknown_method_raises_value_error():
    """Selecting unknown method should raise ValueError."""
    sel = DummySelector(n_states=10, timescale_result=3, variance_result=3)
    its = np.array([10.0, 5.0])

    with pytest.raises(ValueError):
        sel.select_k_slow(its=its, method="not_a_method")


def test_n_states_less_than_two_raises_value_error():
    """Classes with <2 states cannot select slow modes."""
    sel = DummySelector(n_states=1, timescale_result=3, variance_result=3)
    its = np.array([10.0, 5.0])

    with pytest.raises(ValueError):
        sel.select_k_slow(its=its, method="timescale_gap")

    with pytest.raises(ValueError):
        sel.select_k_slow(method="variance")
