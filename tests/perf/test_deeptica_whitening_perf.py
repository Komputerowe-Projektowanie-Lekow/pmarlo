from __future__ import annotations

"""Performance benchmarks for DeepTICA output whitening utilities."""

import os

import numpy as np
import pytest

from pmarlo.ml.deeptica.whitening import apply_output_transform

pytestmark = [pytest.mark.perf, pytest.mark.benchmark, pytest.mark.features]

pytest.importorskip("pytest_benchmark", reason="pytest-benchmark not installed")

if not os.getenv("PMARLO_RUN_PERF"):
    pytest.skip(
        "perf tests disabled; set PMARLO_RUN_PERF=1 to run", allow_module_level=True
    )


def _make_synthetic_projection(
    n_frames: int, n_features: int, seed: int = 1905
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate correlated projections with analytically known whitening data."""
    rng = np.random.default_rng(seed)
    mean = rng.normal(loc=0.0, scale=3.0, size=n_features)

    # Create a positive-definite covariance and its Cholesky factor
    factors = rng.normal(size=(n_features, n_features))
    covariance = factors @ factors.T + n_features * np.eye(n_features)
    cholesky = np.linalg.cholesky(covariance)

    # Sample correlated data then shift by the mean
    base = rng.standard_normal(size=(n_frames, n_features))
    correlated = base @ cholesky.T + mean

    # The whitening transform is the inverse of the Cholesky factor (upper form)
    transform = np.linalg.inv(cholesky.T)
    return (
        correlated.astype(np.float64),
        mean.astype(np.float64),
        transform.astype(np.float64),
    )


@pytest.mark.parametrize(
    ("n_frames", "n_features"),
    [
        (4_096, 8),
        (16_384, 24),
        (65_536, 48),
    ],
    ids=["small", "medium", "large"],
)
def test_apply_output_transform_whitens_projections(
    benchmark: pytest.BenchmarkFixture, n_frames: int, n_features: int
) -> None:
    """Benchmark output whitening and validate mean/covariance normalization."""

    projections, mean, transform = _make_synthetic_projection(n_frames, n_features)

    def _run() -> np.ndarray:
        return apply_output_transform(
            projections, mean=mean, W=transform, already_applied=False
        )

    whitened = benchmark(_run)

    assert whitened.shape == projections.shape

    # Numerical stability: use float64 with loose tolerances for large samples
    component_means = np.mean(whitened, axis=0)
    assert np.allclose(component_means, 0.0, atol=5e-3)

    covariance = np.cov(whitened, rowvar=False, ddof=0)
    identity = np.eye(n_features, dtype=np.float64)
    assert np.allclose(covariance, identity, atol=7e-3)


def test_apply_output_transform_noop_when_already_applied(
    benchmark: pytest.BenchmarkFixture,
) -> None:
    """Benchmark and validate that the \"already applied\" flag avoids extra work."""

    projections, mean, transform = _make_synthetic_projection(1024, 12)
    whitened = apply_output_transform(
        projections, mean=mean, W=transform, already_applied=False
    )

    def _run() -> np.ndarray:
        return apply_output_transform(
            whitened, mean=mean, W=transform, already_applied=True
        )

    result = benchmark(_run)

    # When the transform is marked as already applied, the data should be returned
    # unchanged without additional computation.
    assert result is whitened
    assert np.shares_memory(result, whitened)
    assert np.allclose(result, whitened)
