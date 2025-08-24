import numpy as np
import pytest

from pmarlo.states.msm_bridge import _stationary_from_T
from pmarlo.utils.msm_utils import check_transition_matrix


def _random_stochastic_matrix(rng: np.random.Generator, n: int) -> np.ndarray:
    T = rng.random((n, n))
    T /= T.sum(axis=1, keepdims=True)
    return T


def test_check_transition_matrix_passes():
    rng = np.random.default_rng(0)
    T = _random_stochastic_matrix(rng, 5)
    pi = _stationary_from_T(T)
    check_transition_matrix(T, pi)


def test_check_transition_matrix_catches_tiny_negative():
    rng = np.random.default_rng(1)
    for _ in range(5):
        n = int(rng.integers(2, 6))
        T = _random_stochastic_matrix(rng, n)
        pi = _stationary_from_T(T)
        check_transition_matrix(T, pi)  # baseline
        i = int(rng.integers(0, n))
        j = int(rng.integers(0, n))
        eps = 1e-13
        original = T[i, j]
        T[i, j] = -eps
        T[i, (j + 1) % n] += original + eps
        with pytest.raises(ValueError, match="Negative probabilities"):
            check_transition_matrix(T, pi)
