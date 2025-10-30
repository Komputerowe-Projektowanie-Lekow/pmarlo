"""Unit tests for Kinetic Importance Score."""

from __future__ import annotations

import numpy as np
import pytest

from pmarlo.conformations.kinetic_importance import KineticImportanceScore


def test_kis_init():
    """Test KIS initialization."""
    T = np.array([[0.9, 0.1], [0.2, 0.8]])
    pi = np.array([0.67, 0.33])

    kis = KineticImportanceScore(T, pi)

    assert kis.n_states == 2
    assert kis.T.shape == (2, 2)


def test_kis_compute():
    """Test KIS computation."""
    # 3-state system
    T = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])

    eigenvalues, eigenvectors = np.linalg.eig(T.T)
    idx = np.argmax(np.abs(eigenvalues))
    pi = np.real(eigenvectors[:, idx])
    pi = pi / np.sum(pi)

    kis = KineticImportanceScore(T, pi)

    result = kis.compute(k_slow=2)

    assert result.kis_scores.shape == (3,)
    assert result.k_slow == 2
    assert result.ranked_states.shape == (3,)
    # Check that scores are non-negative
    assert np.all(result.kis_scores >= 0)


def test_kis_select_k_slow():
    """Test automatic k_slow selection."""
    T = np.array([[0.9, 0.1], [0.2, 0.8]])

    eigenvalues, eigenvectors = np.linalg.eig(T.T)
    idx = np.argmax(np.abs(eigenvalues))
    pi = np.real(eigenvectors[:, idx])
    pi = pi / np.sum(pi)

    kis = KineticImportanceScore(T, pi)

    # With implied timescales
    its = np.array([10.0, 2.0, 0.5])
    k_slow = kis.select_k_slow(its, method="timescale_gap", gap_threshold=2.0)

    assert k_slow >= 2  # At least 2 slow modes


def test_kis_ranking():
    """Test that KIS produces valid ranking."""
    # 4-state system with asymmetric populations
    T = np.array(
        [
            [0.95, 0.03, 0.01, 0.01],
            [0.05, 0.90, 0.03, 0.02],
            [0.02, 0.05, 0.90, 0.03],
            [0.01, 0.02, 0.05, 0.92],
        ]
    )

    eigenvalues, eigenvectors = np.linalg.eig(T.T)
    idx = np.argmax(np.abs(eigenvalues))
    pi = np.real(eigenvectors[:, idx])
    pi = pi / np.sum(pi)

    kis = KineticImportanceScore(T, pi)

    result = kis.compute(k_slow=2)

    # Ranked states should be in descending order
    for i in range(len(result.ranked_states) - 1):
        state_i = result.ranked_states[i]
        state_j = result.ranked_states[i + 1]
        assert result.kis_scores[state_i] >= result.kis_scores[state_j]
