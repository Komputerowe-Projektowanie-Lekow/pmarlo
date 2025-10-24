"""Unit tests for TPT analysis."""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("deeptime")

from pmarlo.conformations.tpt_analysis import TPTAnalysis


def simulate_two_state_msm() -> tuple[np.ndarray, np.ndarray]:
    """Create a simple two-state MSM for testing."""
    # Simple two-state system with barrier
    T = np.array([[0.95, 0.05], [0.03, 0.97]])

    # Stationary distribution
    eigenvalues, eigenvectors = np.linalg.eig(T.T)
    idx = np.argmax(np.abs(eigenvalues))
    pi = np.real(eigenvectors[:, idx])
    pi = pi / np.sum(pi)

    return T, pi


def test_tpt_analysis_init():
    """Test TPT analysis initialization."""
    T, pi = simulate_two_state_msm()

    tpt = TPTAnalysis(T, pi)

    assert tpt.T.shape == (2, 2)
    assert tpt.pi.shape == (2,)
    assert tpt.n_states == 2


def test_compute_committor():
    """Test committor computation."""
    T, pi = simulate_two_state_msm()

    tpt = TPTAnalysis(T, pi)

    source = np.array([0])
    sink = np.array([1])

    q_forward = tpt.compute_committor(source, sink, forward=True)

    # Source should have committor = 0
    assert q_forward[0] == pytest.approx(0.0, abs=1e-6)
    # Sink should have committor = 1
    assert q_forward[1] == pytest.approx(1.0, abs=1e-6)


def test_compute_reactive_flux():
    """Test reactive flux computation using deeptime's MSM.reactive_flux()."""
    T, pi = simulate_two_state_msm()

    tpt = TPTAnalysis(T, pi)

    source = np.array([0])
    sink = np.array([1])

    flux_matrix, net_flux, total_flux = tpt.compute_reactive_flux_direct(source, sink)

    assert flux_matrix.shape == (2, 2)
    assert net_flux.shape == (2, 2)
    assert total_flux > 0


def test_analyze_complete():
    """Test complete TPT analysis."""
    T, pi = simulate_two_state_msm()

    tpt = TPTAnalysis(T, pi)

    source = np.array([0])
    sink = np.array([1])

    result = tpt.analyze(source, sink, n_paths=2)

    assert result.source_states.shape == (1,)
    assert result.sink_states.shape == (1,)
    assert result.forward_committor.shape == (2,)
    assert result.rate > 0
    assert result.mfpt > 0
    assert result.total_flux > 0
    assert result.pathway_iterations >= len(result.pathways)
    assert result.pathway_max_iterations >= result.pathway_iterations


def test_find_bottleneck_states():
    """Test bottleneck state identification."""
    T, pi = simulate_two_state_msm()

    tpt = TPTAnalysis(T, pi)

    # Create simple flux matrix
    flux_matrix = np.array([[0.0, 0.5], [0.2, 0.0]])

    bottlenecks = tpt.find_bottleneck_states(flux_matrix, top_n=2)

    assert len(bottlenecks) == 2


def test_tpt_invalid_inputs():
    """Test TPT with invalid inputs."""
    T, pi = simulate_two_state_msm()

    # Overlapping source and sink
    tpt = TPTAnalysis(T, pi)

    with pytest.raises(ValueError, match="must not overlap"):
        tpt.analyze(np.array([0]), np.array([0]))


def test_tpt_large_system():
    """Test TPT on larger system."""
    # 5-state system
    n_states = 5

    # Create random transition matrix
    np.random.seed(42)
    T = np.random.rand(n_states, n_states)
    T = T / T.sum(axis=1, keepdims=True)

    # Compute stationary distribution
    eigenvalues, eigenvectors = np.linalg.eig(T.T)
    idx = np.argmax(np.abs(eigenvalues))
    pi = np.real(eigenvectors[:, idx])
    pi = pi / np.sum(pi)

    tpt = TPTAnalysis(T, pi)

    source = np.array([0, 1])
    sink = np.array([3, 4])

    result = tpt.analyze(source, sink, n_paths=3)

    assert result.forward_committor.shape == (n_states,)
    assert result.rate >= 0
    assert result.total_flux >= 0
    assert result.pathway_iterations >= len(result.pathways)
    assert result.pathway_max_iterations >= result.pathway_iterations

