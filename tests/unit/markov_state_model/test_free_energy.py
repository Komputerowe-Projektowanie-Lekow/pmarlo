import numpy as np
import pytest

from pmarlo.markov_state_model.free_energy import (
    FESResult,
    PMFResult,
    free_energy_from_density,
    generate_1d_pmf,
    generate_2d_fes,
)


def test_generate_1d_pmf_reference():
    data = np.linspace(-1.0, 1.0, 1000)
    res = generate_1d_pmf(data, bins=10, temperature=300.0, smoothing_sigma=None)
    assert isinstance(res, PMFResult)
    H, edges = np.histogram(data, bins=10, range=(data.min(), data.max()), density=True)
    F_ref = free_energy_from_density(H, temperature=300.0)
    F_res = np.nan_to_num(res.F, nan=np.inf, posinf=np.inf)
    close = np.isclose(F_res, F_ref, atol=2e-2)
    both_inf = (F_res == np.inf) & (F_ref == np.inf)
    assert np.all(close | both_inf)
    assert np.allclose(res.counts, H)


def test_generate_2d_fes_reference():
    rng = np.random.default_rng(0)
    x = rng.normal(size=500)
    y = rng.normal(size=500)
    res = generate_2d_fes(
        x, y, bins=(20, 20), temperature=300.0, smooth=False, min_count=0
    )
    assert isinstance(res, FESResult)

    # Test basic properties of the FES result
    assert res.F.shape == (len(res.xedges) - 1, len(res.yedges) - 1)
    assert np.all(np.isfinite(res.F[res.F != np.inf]))  # Finite values are finite
    assert (
        len(res.xedges) >= 20
    )  # At least the requested bins (due to adaptive binning)
    assert len(res.yedges) >= 20
    assert res.metadata["temperature"] == 300.0
    assert "counts" in res.metadata

    # Test that the minimum free energy is zero (after normalization)
    finite_F = res.F[np.isfinite(res.F)]
    if len(finite_F) > 0:
        assert np.allclose(np.min(finite_F), 0.0, atol=1e-10)


def test_generate_2d_fes_shape_mismatch():
    x = np.array([0.0, 1.0])
    y = np.array([0.0])
    with pytest.raises(ValueError):
        generate_2d_fes(x, y)


def test_generate_1d_pmf_invalid_temperature():
    data = np.array([0.0, 1.0])
    with pytest.raises(ValueError):
        generate_1d_pmf(data, temperature=-1.0)


def test_free_energy_from_density_applies_mask_when_not_inpaint():
    density = np.array([[0.4, 0.4], [0.2, 0.0]], dtype=float)
    mask = np.array([[False, False], [False, True]])
    F = free_energy_from_density(density, temperature=300.0, mask=mask, inpaint=False)
    assert np.isnan(F[1, 1])
    assert np.isfinite(F[0, 0])
    finite = F[np.isfinite(F)]
    assert np.allclose(np.min(finite), 0.0)


def test_free_energy_from_density_ignores_mask_when_inpainted():
    density = np.array([[0.4, 0.3], [0.2, 0.1]], dtype=float)
    mask = np.array([[False, False], [False, True]])
    F = free_energy_from_density(density, temperature=300.0, mask=mask, inpaint=True)
    assert not np.isnan(F[1, 1])
    finite = F[np.isfinite(F)]
    assert np.allclose(np.min(finite), 0.0)


def test_free_energy_from_density_requires_positive_temperature():
    with pytest.raises(ValueError):
        free_energy_from_density(np.array([1.0]), temperature=0.0)
