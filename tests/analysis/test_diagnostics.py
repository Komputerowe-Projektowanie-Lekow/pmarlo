import numpy as np
import pytest

from pmarlo.analysis.diagnostics import (
    InsufficientSamplesError,
    _canonical_correlations,
    _covariance,
    compute_diagnostics,
)


def test_compute_diagnostics_triviality_and_mass_warning():
    X = np.random.default_rng(0).normal(size=(200, 2))
    dataset = {"splits": {"train": {"X": X, "inputs": X.copy()}}}
    result = compute_diagnostics(dataset, diag_mass=0.99)

    canon = result.get("canonical_correlation", {})
    assert (
        "train" in canon and canon["train"]
    ), "expected canonical correlations for train split"
    warnings = result.get("warnings", [])
    assert any("reparametrize" in w for w in warnings)
    assert any("MSM diagonal mass" in w for w in warnings)


def test_compute_diagnostics_handles_missing_inputs():
    rng = np.random.default_rng(2)
    dataset = {"splits": {"train": {"X": rng.normal(size=(40, 2))}}}
    result = compute_diagnostics(dataset, diag_mass=0.1)

    assert result.get("canonical_correlation", {}) == {}
    assert "train" in result.get("autocorrelation", {})


def test_canonical_correlation_raises_on_insufficient_samples():
    # Only one paired sample -> should raise InsufficientSamplesError now.
    X = np.random.default_rng(123).normal(size=(1, 3))
    dataset = {"splits": {"tiny": {"X": X, "inputs": X.copy()}}}
    with pytest.raises(InsufficientSamplesError):
        compute_diagnostics(dataset)


def test_covariance_computes_correctly():
    """Test that covariance computation produces correct results."""
    rng = np.random.default_rng(7)
    X = rng.normal(size=(40, 3))
    centered = X - np.mean(X, axis=0, keepdims=True)

    result = _covariance(centered, centered.shape[0])
    expected = np.cov(centered, rowvar=False, ddof=1)

    assert result.shape == (3, 3)
    assert np.allclose(result, expected)
    # Verify covariance matrix is symmetric
    assert np.allclose(result, result.T)


def test_canonical_correlations_computes_correctly():
    """Test that canonical correlation analysis produces valid outputs."""
    rng = np.random.default_rng(42)
    X = rng.normal(size=(50, 3))
    Y = rng.normal(size=(50, 2))

    correlations = _canonical_correlations(X, Y)

    # Should return min(n_features_X, n_features_Y) correlations
    assert len(correlations) == 2
    # All correlations should be between 0 and 1
    assert all(0 <= corr <= 1 for corr in correlations)
    # Correlations should be finite
    assert all(np.isfinite(corr) for corr in correlations)
