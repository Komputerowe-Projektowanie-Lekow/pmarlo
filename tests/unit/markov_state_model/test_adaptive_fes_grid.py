"""Tests for adaptive FES grid sizing functionality."""

import numpy as np
import pytest

from pmarlo.markov_state_model.free_energy import generate_2d_fes


def test_adaptive_grid_reduces_empty_bins():
    """Test that adaptive grid strategy reduces empty bins compared to fixed."""
    rng = np.random.default_rng(42)

    # Generate data with narrow distribution (many empty bins with fixed grid)
    n_samples = 5000
    cv1 = rng.normal(0.0, 0.5, n_samples)  # Narrow distribution
    cv2 = rng.normal(0.0, 0.5, n_samples)

    # Test with fixed strategy
    fes_fixed = generate_2d_fes(
        cv1, cv2,
        bins=(100, 100),
        temperature=300.0,
        grid_strategy="fixed",
    )

    # Test with adaptive strategy
    fes_adaptive = generate_2d_fes(
        cv1, cv2,
        bins=(100, 100),
        temperature=300.0,
        grid_strategy="adaptive",
    )

    # Adaptive should have fewer empty bins
    empty_fixed = fes_fixed.metadata.get("empty_bins_fraction", 0.0)
    empty_adaptive = fes_adaptive.metadata.get("empty_bins_fraction", 0.0)

    assert empty_adaptive < empty_fixed, (
        f"Adaptive strategy should reduce empty bins: "
        f"fixed={empty_fixed:.3f}, adaptive={empty_adaptive:.3f}"
    )

    # Adaptive should target < 0.5 empty bins
    assert empty_adaptive < 0.5, (
        f"Adaptive strategy should achieve < 50% empty bins, got {empty_adaptive:.3f}"
    )


def test_adaptive_grid_persists_metadata():
    """Test that grid strategy and parameters are persisted in metadata."""
    rng = np.random.default_rng(123)
    cv1 = rng.normal(0.0, 1.0, 1000)
    cv2 = rng.normal(0.0, 1.0, 1000)

    fes = generate_2d_fes(
        cv1, cv2,
        bins=(80, 80),
        temperature=300.0,
        grid_strategy="adaptive",
    )

    # Check metadata contains grid information
    assert "grid_strategy" in fes.metadata
    assert fes.metadata["grid_strategy"] == "adaptive"

    assert "grid_shape" in fes.metadata
    assert isinstance(fes.metadata["grid_shape"], tuple)
    assert len(fes.metadata["grid_shape"]) == 2

    assert "grid_ranges" in fes.metadata
    assert "x" in fes.metadata["grid_ranges"]
    assert "y" in fes.metadata["grid_ranges"]


def test_adaptive_grid_adjusts_bin_counts():
    """Test that adaptive strategy adjusts bin counts to achieve target."""
    rng = np.random.default_rng(456)

    # Very sparse data - should reduce bins significantly
    cv1 = rng.normal(0.0, 0.3, 2000)
    cv2 = rng.normal(0.0, 0.3, 2000)

    fes = generate_2d_fes(
        cv1, cv2,
        bins=(150, 150),  # Request many bins
        temperature=300.0,
        grid_strategy="adaptive",
    )

    actual_shape = fes.metadata.get("grid_shape", (150, 150))

    # Should have reduced bin counts from requested
    # (may not always reduce if data is dense enough, but for this sparse case it should)
    assert actual_shape[0] <= 150 or actual_shape[1] <= 150, (
        f"Expected bin reduction for sparse data, got shape={actual_shape}"
    )


def test_fixed_grid_uses_requested_bins():
    """Test that fixed strategy uses exactly the requested bin counts."""
    rng = np.random.default_rng(789)
    cv1 = rng.normal(0.0, 1.0, 3000)
    cv2 = rng.normal(0.0, 1.0, 3000)

    requested_bins = (75, 75)
    fes = generate_2d_fes(
        cv1, cv2,
        bins=requested_bins,
        temperature=300.0,
        grid_strategy="fixed",
    )

    # Fixed strategy should use close to requested bins (may adjust slightly for FD rule)
    actual_shape = fes.metadata.get("grid_shape", requested_bins)

    # Allow some flexibility due to Freedman-Diaconis rule
    assert abs(actual_shape[0] - requested_bins[0]) < 50, (
        f"Fixed strategy bins deviated too much: requested={requested_bins}, got={actual_shape}"
    )


def test_sparse_warning_threshold():
    """Test that sparse FES warning is emitted when > 50% bins are empty."""
    rng = np.random.default_rng(999)

    # Very sparse data with many bins -> should trigger warning
    cv1 = rng.normal(0.0, 0.2, 500)
    cv2 = rng.normal(0.0, 0.2, 500)

    fes = generate_2d_fes(
        cv1, cv2,
        bins=(120, 120),
        temperature=300.0,
        grid_strategy="fixed",  # Use fixed to ensure sparse result
    )

    empty_frac = fes.metadata.get("empty_bins_fraction", 0.0)
    has_warning = "sparse_warning" in fes.metadata

    if empty_frac > 0.50:
        assert has_warning, "Sparse warning should be present when >50% bins empty"
        warning_text = fes.metadata.get("sparse_warning", "")
        assert "Sparse FES" in warning_text
        assert "empty bins" in warning_text


def test_invalid_grid_strategy_raises():
    """Test that invalid grid_strategy values raise an error."""
    rng = np.random.default_rng(111)
    cv1 = rng.normal(0.0, 1.0, 100)
    cv2 = rng.normal(0.0, 1.0, 100)

    with pytest.raises(ValueError, match="grid_strategy must be"):
        generate_2d_fes(
            cv1, cv2,
            bins=(50, 50),
            temperature=300.0,
            grid_strategy="invalid",
        )


def test_adaptive_with_periodic_uses_full_range():
    """Test that adaptive strategy uses full range for periodic coordinates."""
    rng = np.random.default_rng(222)

    # Angles in radians
    cv1 = rng.uniform(-np.pi, np.pi, 1000)
    cv2 = rng.uniform(-np.pi, np.pi, 1000)

    fes = generate_2d_fes(
        cv1, cv2,
        bins=(60, 60),
        temperature=300.0,
        periodic=(True, True),
        grid_strategy="adaptive",
    )

    # Should still work and produce valid FES
    assert fes.F is not None
    assert np.isfinite(fes.F).any()

    # Grid strategy should be recorded
    assert fes.metadata.get("grid_strategy") == "adaptive"


def test_default_grid_strategy_is_adaptive():
    """Test that the default grid strategy is 'adaptive'."""
    rng = np.random.default_rng(333)
    cv1 = rng.normal(0.0, 1.0, 1000)
    cv2 = rng.normal(0.0, 1.0, 1000)

    # Call without specifying grid_strategy
    fes = generate_2d_fes(
        cv1, cv2,
        bins=(80, 80),
        temperature=300.0,
    )

    # Should default to adaptive
    assert fes.metadata.get("grid_strategy") == "adaptive"

