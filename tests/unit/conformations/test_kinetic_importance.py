"""Unit tests for Kinetic Importance Score."""

from __future__ import annotations

import numpy as np
import pytest

from pmarlo.conformations.kinetic_importance import KineticImportanceScore
from pmarlo.conformations.results import KISResult


def hyperparam_stability_idea(
    original_top_k: np.ndarray,
    ensemble_rankings: list[np.ndarray],
    ensemble_kis: list[np.ndarray],
) -> tuple[float, np.ndarray]:
    """Reference implementation of the hyperparameter stability idea."""
    if len(ensemble_rankings) == 0:
        n_states = len(ensemble_kis[0]) if ensemble_kis else len(original_top_k)
        return 0.0, np.zeros(n_states)

    k = len(original_top_k)
    overlaps = []

    original_set = set(map(int, original_top_k))

    for ranking in ensemble_rankings:
        ensemble_top_k = set(map(int, ranking[:k]))
        overlap = len(original_set & ensemble_top_k) / float(k)
        overlaps.append(overlap)

    stability = float(np.mean(overlaps))

    kis_mat = np.stack(ensemble_kis, axis=0)
    ensemble_std = kis_mat.std(axis=0)

    return stability, ensemble_std


def _make_kis_result(scores: np.ndarray, ranking: np.ndarray) -> KISResult:
    """Create a minimal KISResult for mocking compute() calls."""
    scores = np.asarray(scores, dtype=float)
    ranking = np.asarray(ranking, dtype=int)
    if scores.shape[0] != ranking.shape[0]:
        raise ValueError('KIS scores and rankings must have the same length')

    n_states = scores.shape[0]
    k_slow = 2
    eigenvectors = np.zeros((k_slow + 1, n_states), dtype=float)
    eigenvalues = np.zeros(k_slow + 1, dtype=float)
    return KISResult(
        kis_scores=scores,
        k_slow=k_slow,
        eigenvectors=eigenvectors,
        eigenvalues=eigenvalues,
        ranked_states=ranking,
    )


def _install_mock_compute_sequence(
    monkeypatch: pytest.MonkeyPatch, results: list[KISResult]
) -> None:
    """Install a compute() mock that returns the provided KIS results in order."""
    queue = [res for res in results]

    def fake_compute(self, k_slow: int | str = 'auto', its=None) -> KISResult:
        if not queue:
            raise AssertionError('Unexpected KineticImportanceScore.compute() call')
        return queue.pop(0)

    monkeypatch.setattr(KineticImportanceScore, 'compute', fake_compute)


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


def test_hyperparam_stability_perfect_overlap(monkeypatch):
    n_states = 10
    original_top10 = np.arange(n_states)
    reference_kis = np.linspace(0.1, 1.0, n_states)

    ensemble_rankings = [np.arange(n_states) for _ in range(3)]
    ensemble_kis = [reference_kis.copy() for _ in range(3)]

    expected_stability, expected_std = hyperparam_stability_idea(
        original_top10,
        ensemble_rankings,
        ensemble_kis,
    )

    result_sequence: list[KISResult] = [
        _make_kis_result(reference_kis, original_top10)
    ]
    for ranking, kis_values in zip(ensemble_rankings, ensemble_kis):
        result_sequence.append(_make_kis_result(kis_values, ranking))

    _install_mock_compute_sequence(monkeypatch, result_sequence)

    def fake_rebuild(self, dtrajs, lag):
        return np.eye(n_states), np.ones(n_states) / n_states

    monkeypatch.setattr(
        KineticImportanceScore,
        '_rebuild_msm',
        fake_rebuild,
    )

    kis = KineticImportanceScore(np.eye(n_states), np.ones(n_states) / n_states)
    stability, ensemble_std = kis.hyperparameter_ensemble_stability(
        dtrajs=[np.arange(5)],
        lag_times=[1, 2, 3],
        k_slow=2,
    )

    assert stability == pytest.approx(expected_stability)
    assert ensemble_std.shape == (n_states,)
    assert np.allclose(ensemble_std, expected_std)


def test_hyperparam_stability_partial_overlap_and_std(monkeypatch):
    n_states = 20
    original_top10 = np.arange(10)
    reference_kis = np.linspace(1.0, 0.0, n_states)

    ranking_1 = np.arange(n_states)
    ranking_2 = np.array(
        [0, 1, 2, 3, 4, 10, 11, 12, 13, 14, 5, 6, 7, 8, 9, 15, 16, 17, 18, 19]
    )

    kis_1 = np.zeros(n_states)
    kis_1[0] = 1.0

    kis_2 = np.zeros(n_states)
    kis_2[1] = 1.0

    ensemble_rankings = [ranking_1, ranking_2]
    ensemble_kis = [kis_1, kis_2]

    expected_stability, expected_std = hyperparam_stability_idea(
        original_top10,
        ensemble_rankings,
        ensemble_kis,
    )

    result_sequence: list[KISResult] = [
        _make_kis_result(reference_kis, ranking_1)
    ]
    for ranking, kis_values in zip(ensemble_rankings, ensemble_kis):
        result_sequence.append(_make_kis_result(kis_values, ranking))

    _install_mock_compute_sequence(monkeypatch, result_sequence)

    def fake_rebuild(self, dtrajs, lag):
        return np.eye(n_states), np.ones(n_states) / n_states

    monkeypatch.setattr(
        KineticImportanceScore,
        '_rebuild_msm',
        fake_rebuild,
    )

    kis = KineticImportanceScore(np.eye(n_states), np.ones(n_states) / n_states)
    stability, ensemble_std = kis.hyperparameter_ensemble_stability(
        dtrajs=[np.arange(6)],
        lag_times=[1, 2],
        k_slow=2,
    )

    assert stability == pytest.approx(0.75)
    assert ensemble_std.shape == (n_states,)
    assert ensemble_std[0] == pytest.approx(0.5)
    assert ensemble_std[1] == pytest.approx(0.5)
    assert np.allclose(ensemble_std[2:], 0.0)


def test_hyperparam_stability_no_ensemble_members(monkeypatch):
    n_states = 10
    original_top10 = np.arange(10)
    reference_kis = np.zeros(n_states)

    expected_stability, expected_std = hyperparam_stability_idea(
        original_top10,
        [],
        [reference_kis],
    )

    result_sequence: list[KISResult] = [
        _make_kis_result(reference_kis, np.arange(n_states))
    ]

    _install_mock_compute_sequence(monkeypatch, result_sequence)

    def failing_rebuild(self, dtrajs, lag):
        raise RuntimeError('MSM build failed')

    monkeypatch.setattr(
        KineticImportanceScore,
        '_rebuild_msm',
        failing_rebuild,
    )

    kis = KineticImportanceScore(np.eye(n_states), np.ones(n_states) / n_states)
    stability, ensemble_std = kis.hyperparameter_ensemble_stability(
        dtrajs=[np.arange(4)],
        lag_times=[1, 2],
        k_slow=2,
    )

    assert stability == pytest.approx(expected_stability)
    assert ensemble_std.shape == (n_states,)
    assert np.allclose(ensemble_std, expected_std)
