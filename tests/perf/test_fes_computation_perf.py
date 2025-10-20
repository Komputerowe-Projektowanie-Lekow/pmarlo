from __future__ import annotations

"""Performance benchmarks for Free Energy Surface (FES) computation.

These benchmarks measure FES computation performance which is critical for:
- Histogram computation with reweighting
- Gaussian smoothing
- Free energy conversion from probabilities
- Large dataset handling

Run with: pytest -m benchmark tests/perf/test_fes_computation_perf.py
"""

import os

import numpy as np
import pytest

pytestmark = [pytest.mark.perf, pytest.mark.benchmark, pytest.mark.msm]

# Optional dependencies
pytest.importorskip("pytest_benchmark", reason="pytest-benchmark not installed")

if not os.getenv("PMARLO_RUN_PERF"):
    pytest.skip(
        "perf tests disabled; set PMARLO_RUN_PERF=1 to run", allow_module_level=True
    )


def _generate_cv_data(
    n_frames: int, periodic: bool = False, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic CV data for FES benchmarking."""
    rng = np.random.default_rng(seed)

    if periodic:
        # Phi/psi angles: -180 to 180
        cv1 = rng.uniform(-180, 180, n_frames).astype(np.float64)
        cv2 = rng.uniform(-180, 180, n_frames).astype(np.float64)
    else:
        # Generic CVs
        cv1 = rng.standard_normal(n_frames).astype(np.float64)
        cv2 = rng.standard_normal(n_frames).astype(np.float64)

    return cv1, cv2


def _generate_weights(n_frames: int, seed: int = 42) -> np.ndarray:
    """Generate synthetic frame weights (from MSM stationary distribution)."""
    rng = np.random.default_rng(seed)
    # Simulate non-uniform weights
    weights = rng.exponential(scale=1.0, size=n_frames).astype(np.float64)
    weights /= weights.sum()
    return weights


@pytest.fixture
def small_cv_data():
    """Small CV dataset (1K frames)."""
    return _generate_cv_data(1000)


@pytest.fixture
def medium_cv_data():
    """Medium CV dataset (10K frames)."""
    return _generate_cv_data(10_000)


@pytest.fixture
def large_cv_data():
    """Large CV dataset (100K frames)."""
    return _generate_cv_data(100_000)


@pytest.fixture
def periodic_cv_data():
    """Periodic CV dataset (phi/psi angles)."""
    return _generate_cv_data(10_000, periodic=True)


def test_2d_histogram_unweighted(benchmark, medium_cv_data):
    """Benchmark 2D histogram computation without weights (baseline)."""
    cv1, cv2 = medium_cv_data

    def _compute_histogram():
        return np.histogram2d(cv1, cv2, bins=50)

    result = benchmark(_compute_histogram)
    H, xedges, yedges = result
    assert H.shape == (50, 50)


def test_2d_histogram_weighted(benchmark, medium_cv_data):
    """Benchmark 2D histogram computation with MSM weights (realistic case)."""
    cv1, cv2 = medium_cv_data
    weights = _generate_weights(len(cv1))

    def _compute_histogram():
        return np.histogram2d(cv1, cv2, bins=50, weights=weights)

    result = benchmark(_compute_histogram)
    H, xedges, yedges = result
    assert H.shape == (50, 50)


def test_gaussian_smoothing_small(benchmark):
    """Benchmark Gaussian smoothing on small FES grid (50x50)."""
    from scipy.ndimage import gaussian_filter

    # Simulate FES grid
    H = np.random.rand(50, 50)

    def _smooth():
        return gaussian_filter(H, sigma=0.6)

    smoothed = benchmark(_smooth)
    assert smoothed.shape == H.shape


def test_gaussian_smoothing_large(benchmark):
    """Benchmark Gaussian smoothing on large FES grid (200x200)."""
    from scipy.ndimage import gaussian_filter

    H = np.random.rand(200, 200)

    def _smooth():
        return gaussian_filter(H, sigma=0.6)

    smoothed = benchmark(_smooth)
    assert smoothed.shape == H.shape


def test_probability_to_free_energy(benchmark):
    """Benchmark probability to free energy conversion."""
    # Simulate probability histogram
    H = np.random.rand(50, 50)
    H /= H.sum()  # Normalize
    H[H < 1e-10] = 1e-10  # Avoid log(0)

    temperature = 300.0
    kT = 8.314e-3 * temperature  # kJ/mol

    def _convert():
        with np.errstate(divide="ignore", invalid="ignore"):
            F = -kT * np.log(H)
            F -= F.min()
        return F

    F = benchmark(_convert)
    assert F.shape == H.shape


def test_full_fes_pipeline_small(benchmark, small_cv_data):
    """Benchmark complete FES computation pipeline on small dataset."""
    from scipy.ndimage import gaussian_filter

    cv1, cv2 = small_cv_data
    weights = _generate_weights(len(cv1))
    bins = 50
    temperature = 300.0
    kT = 8.314e-3 * temperature

    def _compute_fes():
        # 1. Compute weighted histogram
        H, xedges, yedges = np.histogram2d(cv1, cv2, bins=bins, weights=weights)

        # 2. Smooth
        H_smooth = gaussian_filter(H, sigma=0.6)

        # 3. Convert to free energy
        H_smooth[H_smooth < 1e-10] = 1e-10
        with np.errstate(divide="ignore", invalid="ignore"):
            F = -kT * np.log(H_smooth)
            F -= F.min()

        return F, xedges, yedges

    result = benchmark(_compute_fes)
    F, xedges, yedges = result
    assert F.shape == (bins, bins)


def test_full_fes_pipeline_medium(benchmark, medium_cv_data):
    """Benchmark complete FES computation pipeline on medium dataset (realistic)."""
    from scipy.ndimage import gaussian_filter

    cv1, cv2 = medium_cv_data
    weights = _generate_weights(len(cv1))
    bins = 50
    temperature = 300.0
    kT = 8.314e-3 * temperature

    def _compute_fes():
        H, xedges, yedges = np.histogram2d(cv1, cv2, bins=bins, weights=weights)
        H_smooth = gaussian_filter(H, sigma=0.6)
        H_smooth[H_smooth < 1e-10] = 1e-10
        with np.errstate(divide="ignore", invalid="ignore"):
            F = -kT * np.log(H_smooth)
            F -= F.min()
        return F, xedges, yedges

    result = benchmark(_compute_fes)
    F, xedges, yedges = result
    assert F.shape == (bins, bins)


def test_full_fes_pipeline_large(benchmark, large_cv_data):
    """Benchmark complete FES computation pipeline on large dataset."""
    from scipy.ndimage import gaussian_filter

    cv1, cv2 = large_cv_data
    weights = _generate_weights(len(cv1))
    bins = 50
    temperature = 300.0
    kT = 8.314e-3 * temperature

    def _compute_fes():
        H, xedges, yedges = np.histogram2d(cv1, cv2, bins=bins, weights=weights)
        H_smooth = gaussian_filter(H, sigma=0.6)
        H_smooth[H_smooth < 1e-10] = 1e-10
        with np.errstate(divide="ignore", invalid="ignore"):
            F = -kT * np.log(H_smooth)
            F -= F.min()
        return F, xedges, yedges

    result = benchmark(_compute_fes)
    F, xedges, yedges = result
    assert F.shape == (bins, bins)


def test_periodic_boundary_handling(benchmark, periodic_cv_data):
    """Benchmark FES computation with periodic boundary conditions (phi/psi)."""
    from scipy.ndimage import gaussian_filter

    cv1, cv2 = periodic_cv_data
    weights = _generate_weights(len(cv1))
    bins = 50
    temperature = 300.0
    kT = 8.314e-3 * temperature

    # Use explicit ranges for periodic data
    ranges = [[-180, 180], [-180, 180]]

    def _compute_fes():
        H, xedges, yedges = np.histogram2d(
            cv1, cv2, bins=bins, range=ranges, weights=weights
        )

        # For periodic data, wrap boundaries before smoothing
        H_padded = np.pad(H, 1, mode="wrap")
        H_smooth = gaussian_filter(H_padded, sigma=0.6)[1:-1, 1:-1]

        H_smooth[H_smooth < 1e-10] = 1e-10
        with np.errstate(divide="ignore", invalid="ignore"):
            F = -kT * np.log(H_smooth)
            F -= F.min()

        return F, xedges, yedges

    result = benchmark(_compute_fes)
    F, xedges, yedges = result
    assert F.shape == (bins, bins)


def test_high_resolution_fes(benchmark, medium_cv_data):
    """Benchmark FES computation with high resolution (200x200 bins)."""
    from scipy.ndimage import gaussian_filter

    cv1, cv2 = medium_cv_data
    weights = _generate_weights(len(cv1))
    bins = 200  # High resolution
    temperature = 300.0
    kT = 8.314e-3 * temperature

    def _compute_fes():
        H, xedges, yedges = np.histogram2d(cv1, cv2, bins=bins, weights=weights)
        H_smooth = gaussian_filter(H, sigma=0.6)
        H_smooth[H_smooth < 1e-10] = 1e-10
        with np.errstate(divide="ignore", invalid="ignore"):
            F = -kT * np.log(H_smooth)
            F -= F.min()
        return F, xedges, yedges

    result = benchmark(_compute_fes)
    F, xedges, yedges = result
    assert F.shape == (bins, bins)


def test_multiple_smoothing_passes(benchmark):
    """Benchmark multiple Gaussian smoothing passes (iterative refinement)."""
    from scipy.ndimage import gaussian_filter

    H = np.random.rand(100, 100)

    def _smooth_multiple():
        result = H.copy()
        for _ in range(5):
            result = gaussian_filter(result, sigma=0.6)
        return result

    smoothed = benchmark(_smooth_multiple)
    assert smoothed.shape == H.shape


def test_contour_level_computation(benchmark):
    """Benchmark contour level computation for FES visualization."""
    # Simulate FES
    F = np.random.rand(50, 50) * 50.0  # kJ/mol range

    def _compute_levels():
        levels = np.linspace(F.min(), F.min() + 20, 10)
        return levels

    levels = benchmark(_compute_levels)
    assert len(levels) == 10

