"""Behavioral tests for the variance-explained selector."""

from __future__ import annotations

import numpy as np
import pytest

from pmarlo.conformations.kinetic_importance import KineticImportanceScore


class DummyKIS:
    """Minimal helper exposing the API used by the selector."""

    def __init__(self, eigenvalues):
        self.n_states = len(eigenvalues)
        self._eigenvalues = np.asarray(eigenvalues, dtype=complex)

    def _compute_eigenvectors(self, n_vecs):
        """Return the provided spectrum and dummy eigenvectors."""
        assert n_vecs == self.n_states
        return self._eigenvalues, np.eye(self.n_states)


def _select(dummy: DummyKIS, threshold: float) -> int:
    """Invoke the real selection logic on a dummy object."""
    return KineticImportanceScore._select_by_variance_explained(dummy, threshold)


def test_small_two_state_system_always_returns_two():
    dummy = DummyKIS([1.0, 0.4])
    for thr in [0.0, 0.1, 0.5, 0.9, 1.0]:
        assert _select(dummy, thr) == 2


def test_three_state_fraction_thresholds():
    dummy = DummyKIS([1.0, 0.8, 0.4])
    assert _select(dummy, 0.0) == 2
    assert _select(dummy, 0.1) == 2
    assert _select(dummy, 0.5) == 2
    assert _select(dummy, 0.66) == 2
    assert _select(dummy, 2.0 / 3.0) == 3
    assert _select(dummy, 0.7) == 3
    assert _select(dummy, 0.9) == 3
    assert _select(dummy, 1.0) == 3


def test_equal_slow_modes_symmetry():
    dummy = DummyKIS([1.0, 0.5, 0.5])
    assert _select(dummy, 0.0) == 2
    assert _select(dummy, 0.49) == 2
    assert _select(dummy, 0.5) == 2
    assert _select(dummy, 0.51) == 3
    assert _select(dummy, 0.9) == 3
    assert _select(dummy, 1.0) == 3


def test_complex_eigenvalues_use_magnitudes():
    dummy = DummyKIS([1.0, 0.5 + 0.5j, 0.0 + 0.3j])
    assert _select(dummy, 0.2) == 2
    assert _select(dummy, 0.6) == 2
    assert _select(dummy, 0.8) == 3
    assert _select(dummy, 0.95) == 3


def test_negative_slow_eigenvalues_contribute_positive_weight():
    dummy = DummyKIS([1.0, -0.9, 0.3])
    assert _select(dummy, 0.0) == 2
    assert _select(dummy, 0.5) == 2
    assert _select(dummy, 0.8) == 3
    assert _select(dummy, 0.9) == 3
    assert _select(dummy, 1.0) == 3


def test_monotonicity_in_threshold():
    dummy = DummyKIS([1.0, 0.8, 0.4, 0.2])
    thresholds = np.linspace(0.0, 1.0, 11)
    k_values = [_select(dummy, thr) for thr in thresholds]
    assert k_values == sorted(k_values)
    assert all(k >= 2 for k in k_values)
    assert all(k <= dummy.n_states for k in k_values)


def test_high_threshold_can_use_all_slow_modes():
    eigenvalues = [1.0, 0.7, 0.5, 0.3, 0.1]
    dummy = DummyKIS(eigenvalues)
    k_low = _select(dummy, 0.1)
    k_high = _select(dummy, 0.999)
    assert 2 <= k_low <= dummy.n_states
    assert k_high == dummy.n_states
