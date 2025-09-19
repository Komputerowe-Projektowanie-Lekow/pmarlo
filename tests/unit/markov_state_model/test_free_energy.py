import numpy as np
import pytest

from pmarlo.markov_state_model.free_energy import (
    FESResult,
    PMFResult,
    generate_1d_pmf,
    generate_2d_fes,
)


def _kT_kJ_per_mol(temperature_kelvin: float) -> float:
    from scipy import constants

    return float(constants.k * temperature_kelvin * constants.Avogadro / 1000.0)


def test_generate_1d_pmf_reference():
    data = np.linspace(-1.0, 1.0, 1000)
    res = generate_1d_pmf(data, bins=10, temperature=300.0, smoothing_sigma=None)
    assert isinstance(res, PMFResult)
    H, edges = np.histogram(data, bins=10, range=(data.min(), data.max()), density=True)
    kT = _kT_kJ_per_mol(300.0)
    tiny = np.finfo(float).tiny
    H_clipped = np.clip(H, tiny, None)
    F_ref = np.where(H > 0, -kT * np.log(H_clipped), np.inf)
    F_ref -= np.nanmin(F_ref)
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
