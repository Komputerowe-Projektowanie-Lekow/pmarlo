"""Mathematical contract tests for `_compute_eigenvectors`."""

from __future__ import annotations

import numpy as np
import pytest

from pmarlo.conformations.kinetic_importance import KineticImportanceScore


def make_two_state_chain(p: float = 0.8, q: float = 0.6):
    """
    Simple 2-state ergodic Markov chain:

        [ p      1-p   ]
    T = [ 1-q    q     ]

    Eigenvalues: 1 and (p + q - 1)
    Stationary distribution:
        pi_1 = (1 - q) / (2 - p - q)
        pi_2 = (1 - p) / (2 - p - q)
    """
    T = np.array(
        [
            [p, 1.0 - p],
            [1.0 - q, q],
        ],
        dtype=float,
    )

    denom = 2.0 - p - q
    pi = np.array(
        [
            (1.0 - q) / denom,
            (1.0 - p) / denom,
        ],
        dtype=float,
    )

    lambda1 = 1.0
    lambda2 = p + q - 1.0

    return T, pi, np.array([lambda1, lambda2], dtype=float)


def make_cyclic_three_state_chain():
    """
    3-state cyclic Markov chain:

        [0 1 0]
    T = [0 0 1]
        [1 0 0]

    This is a valid row-stochastic matrix.
    Eigenvalues: 1, exp(2πi/3), exp(4πi/3) (one real, two complex).
    Stationary distribution is uniform (1/3, 1/3, 1/3).
    """
    T = np.array(
        [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    pi = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=float)
    return T, pi


def test_two_state_chain_eigenvalues_and_left_eigenvectors():
    """
    The function should return eigenvalues of the transition matrix and
    left eigenvectors v such that v @ T ≈ lambda * v, sorted by |lambda|.
    """
    T, pi, expected_evals = make_two_state_chain()
    kis = KineticImportanceScore(T=T, pi=pi)

    eigenvalues, eigenvectors = kis._compute_eigenvectors(n_vecs=2)

    # Shapes
    assert eigenvalues.shape == (2,)
    assert eigenvectors.shape == (2, 2)

    # Eigenvalues: one should be 1, the other p + q - 1 (here 0.4)
    # Independent of order we expect the set {1, lambda2}.
    assert np.isclose(eigenvalues[0], 1.0, atol=1e-12)
    assert np.isclose(eigenvalues[1], expected_evals[1], atol=1e-12)

    # Sorting by magnitude: |lambda_0| >= |lambda_1|
    assert abs(eigenvalues[0]) >= abs(eigenvalues[1]) - 1e-12

    # Check left eigenvector property v @ T ≈ lambda v for each returned pair
    for val, vec in zip(eigenvalues, eigenvectors):
        lhs = vec @ T
        rhs = val * vec
        np.testing.assert_allclose(lhs, rhs, atol=1e-10, rtol=1e-10)


def test_stationary_eigenvector_matches_stationary_distribution_up_to_scale():
    """
    Leading eigenpair (largest |lambda|) should correspond to the stationary mode.
    After normalization the first left eigenvector should match pi.
    """
    T, pi, _ = make_two_state_chain()
    kis = KineticImportanceScore(T=T, pi=pi)

    eigenvalues, eigenvectors = kis._compute_eigenvectors(n_vecs=1)

    # Leading eigenvalue is 1
    assert np.isclose(eigenvalues[0], 1.0, atol=1e-12)

    # Normalize leading left eigenvector to compare with stationary distribution.
    v = eigenvectors[0].real.copy()

    # Fix global sign ambiguity
    if v[0] < 0:
        v = -v

    # Normalize to sum 1
    v /= np.sum(v)

    np.testing.assert_allclose(v, pi, atol=1e-10, rtol=1e-10)


def test_n_vecs_is_capped_by_number_of_states():
    """Requesting more eigenvectors than states should just cap the output."""
    T, pi, _ = make_two_state_chain()
    kis = KineticImportanceScore(T=T, pi=pi)

    eigenvalues, eigenvectors = kis._compute_eigenvectors(n_vecs=10)

    # Only 2 states, so at most 2 eigenvectors
    assert eigenvalues.shape == (2,)
    assert eigenvectors.shape == (2, 2)


def test_n_vecs_must_be_positive():
    """n_vecs must be strictly positive."""
    T, pi, _ = make_two_state_chain()
    kis = KineticImportanceScore(T=T, pi=pi)

    with pytest.raises(ValueError):
        kis._compute_eigenvectors(0)

    with pytest.raises(ValueError):
        kis._compute_eigenvectors(-1)


def test_cyclic_three_state_chain_yields_complex_eigenpairs():
    """
    For a 3-cycle Markov chain, there are complex eigenvalues.
    The routine should preserve complex eigenpairs and keep left-eig property.
    """
    T, pi = make_cyclic_three_state_chain()
    kis = KineticImportanceScore(T=T, pi=pi)

    eigenvalues, eigenvectors = kis._compute_eigenvectors(n_vecs=3)

    # There should be at least one eigenvalue with a non-negligible imaginary part.
    assert np.any(np.abs(np.imag(eigenvalues)) > 1e-6)

    # Eigenvectors should be complex in this case
    assert np.iscomplexobj(eigenvectors)

    # Check left eigenvector property v @ T ≈ lambda v for each returned pair
    for val, vec in zip(eigenvalues, eigenvectors):
        lhs = vec @ T
        rhs = val * vec
        np.testing.assert_allclose(lhs, rhs, atol=1e-10, rtol=1e-10)
