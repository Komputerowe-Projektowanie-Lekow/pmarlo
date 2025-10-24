"""Unit tests for state detection."""

from __future__ import annotations

import numpy as np
import pytest

from pmarlo.conformations.state_detection import StateDetector


def test_detect_from_populations():
    """Test detection from populations."""
    detector = StateDetector()

    pi = np.array([0.4, 0.3, 0.2, 0.1])

    source, sink = detector.detect_from_populations(pi, top_n=2)

    assert len(source) == 1
    assert len(sink) == 1
    assert source[0] == 0  # Highest population
    assert sink[0] == 1  # Second highest


def test_from_state_indices():
    """Test manual specification by indices."""
    detector = StateDetector()

    source_indices = [0, 1, 2]
    sink_indices = [5, 6, 7]

    source, sink = detector.from_state_indices(source_indices, sink_indices)

    assert np.array_equal(source, np.array([0, 1, 2]))
    assert np.array_equal(sink, np.array([5, 6, 7]))


def test_from_macrostate_labels():
    """Test specification from macrostate labels."""
    detector = StateDetector()

    # 6 microstates in 2 macrostates
    labels = np.array([0, 0, 0, 1, 1, 1])

    source, sink = detector.from_macrostate_labels(labels, source_id=0, sink_id=1)

    assert np.array_equal(source, np.array([0, 1, 2]))
    assert np.array_equal(sink, np.array([3, 4, 5]))


def test_from_cv_ranges():
    """Test CV-based detection."""
    detector = StateDetector()

    # Mock CV data (e.g., dihedral angles)
    cv_data = np.array([-150, -140, -100, 20, 40, 50, 100, 110, 150])
    dtrajs = [np.arange(len(cv_data))]  # Each frame is a state

    source_range = (-180, -90)
    sink_range = (0, 90)

    source, sink = detector.from_cv_ranges(
        cv_data, "phi", source_range, sink_range, dtrajs
    )

    # States -150, -140, -100 in source range
    assert len(source) == 3
    # States 20, 40, 50 in sink range
    assert len(sink) == 3


def test_auto_detect_fallback():
    """Test auto-detection with minimal data."""
    detector = StateDetector()

    # Simple 3-state system
    T = np.array([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.05, 0.05, 0.9]])

    eigenvalues, eigenvectors = np.linalg.eig(T.T)
    idx = np.argmax(np.abs(eigenvalues))
    pi = np.real(eigenvectors[:, idx])
    pi = pi / np.sum(pi)

    # No FES or ITS, should fall back to population
    source, sink = detector.auto_detect(T, pi, fes=None, its=None, n_states=2)

    assert len(source) >= 1
    assert len(sink) >= 1


def test_detect_from_fes_local_minima():
    """Test FES-based detection with local minima."""
    detector = StateDetector()

    # Create mock FES with two clear minima
    fes_array = np.ones((10, 10)) * 10.0
    fes_array[2, 2] = 0.5  # First minimum
    fes_array[7, 7] = 1.0  # Second minimum

    # Mock FES object
    class MockFES:
        F = fes_array

    fes = MockFES()

    source, sink = detector.detect_from_fes(fes, n_basins=2, method="local_minima")

    assert len(source) >= 1
    assert len(sink) >= 1

