"""Conceptual contract tests for KineticImportanceScore.compute."""

from __future__ import annotations

import numpy as np
import pytest

from pmarlo.conformations.kinetic_importance import KineticImportanceScore


def make_two_state_symmetric_chain():
    """
    Build a simple 2 state reversible MSM with known eigenvectors.

    Transition matrix:
        [[0.9, 0.1],
         [0.1, 0.9]]

    Stationary distribution is uniform. Eigenvectors are known analytically.
    """
    T = np.array([[0.9, 0.1], [0.1, 0.9]], dtype=float)
    pi = np.array([0.5, 0.5], dtype=float)
    return T, pi


def make_random_msm(n_states: int, seed: int = 0):
    """
    Build a dense MSM plus stationary distribution for property-based tests.
    """
    rng = np.random.default_rng(seed)
    X = rng.random((n_states, n_states))
    T = X / X.sum(axis=1, keepdims=True)

    # Stationary distribution from eigenvector of T.T with eigenvalue 1
    evals, evecs = np.linalg.eig(T.T)
    idx = np.argmin(np.abs(evals - 1.0))
    pi = np.real(evecs[:, idx])
    pi = np.maximum(pi, 0.0)
    pi = pi / pi.sum()

    return T, pi


def test_kis_two_state_chain_matches_analytic_value():
    """KIS matches analytic result for a two-state symmetric MSM."""
    T, pi = make_two_state_symmetric_chain()
    kis = KineticImportanceScore(T, pi)

    result = kis.compute(k_slow=1)

    assert result.k_slow == 1
    assert result.kis_scores.shape == (2,)
    assert np.allclose(result.kis_scores, np.array([0.25, 0.25]), atol=1e-7)
    assert set(result.ranked_states.tolist()) == {0, 1}


def test_kis_scores_are_nonnegative_and_rank_uses_scores():
    """Generic MSM should produce nonnegative scores and consistent ranking."""
    T, pi = make_random_msm(n_states=5, seed=1)
    kis = KineticImportanceScore(T, pi)

    result = kis.compute(k_slow=2)

    scores = result.kis_scores
    ranks = result.ranked_states

    assert np.all(scores >= -1e-12)
    assert set(ranks.tolist()) == set(range(T.shape[0]))
    sorted_idx = np.argsort(scores)[::-1]
    assert np.array_equal(ranks, sorted_idx)


def test_kis_is_monotonic_in_k_slow():
    """Including more slow modes should not decrease any KIS component."""
    T, pi = make_random_msm(n_states=4, seed=2)
    kis = KineticImportanceScore(T, pi)

    result_k1 = kis.compute(k_slow=1)
    result_k2 = kis.compute(k_slow=2)
    result_k3 = kis.compute(k_slow=3)

    scores1 = result_k1.kis_scores
    scores2 = result_k2.kis_scores
    scores3 = result_k3.kis_scores

    assert np.all(scores2 + 1e-12 >= scores1)
    assert np.all(scores3 + 1e-12 >= scores2)


class DummyKISAuto(KineticImportanceScore):
    """Helper subclass to assert that `k_slow="auto"` defers to select_k_slow."""

    def __init__(self, T, pi, k_slow_to_return: int):
        super().__init__(T, pi)
        self._k_slow_to_return = k_slow_to_return
        self.select_called_with = None

    def select_k_slow(self, its=None):
        self.select_called_with = its
        return self._k_slow_to_return


def test_kis_k_slow_auto_uses_selector():
    """k_slow='auto' must rely on select_k_slow for the resolved value."""
    T, pi = make_random_msm(n_states=4, seed=3)
    kis = DummyKISAuto(T, pi, k_slow_to_return=2)

    dummy_its = np.array([10.0, 5.0, 1.0])
    result = kis.compute(k_slow="auto", its=dummy_its)

    assert result.k_slow == 2
    assert kis.select_called_with is dummy_its


def test_kis_accepts_integer_like_string_for_k_slow():
    """Stringified integers should be accepted and behave like ints."""
    T, pi = make_random_msm(n_states=4, seed=4)
    kis = KineticImportanceScore(T, pi)

    res_int = kis.compute(k_slow=2)
    res_str = kis.compute(k_slow="2")

    assert res_str.k_slow == 2
    assert np.allclose(res_int.kis_scores, res_str.kis_scores)


def test_kis_invalid_k_slow_string_raises():
    """Invalid non-integer strings for k_slow should raise ValueError."""
    T, pi = make_random_msm(n_states=3, seed=5)
    kis = KineticImportanceScore(T, pi)

    with pytest.raises(ValueError):
        kis.compute(k_slow="not-an-int")


def test_kis_out_of_range_k_slow_is_handled_gracefully():
    """k_slow values outside [1, n_states - 1] must be clamped into range."""
    T, pi = make_random_msm(n_states=5, seed=6)
    kis = KineticImportanceScore(T, pi)

    res_small = kis.compute(k_slow=0)
    assert 1 <= res_small.k_slow <= 4

    res_large = kis.compute(k_slow=100)
    assert 1 <= res_large.k_slow <= 4
