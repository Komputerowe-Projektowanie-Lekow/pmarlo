import numpy as np
import pytest

from pmarlo.fes import FESResult, PMFResult, generate_1d_pmf, generate_2d_fes


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
    H, xedges, yedges = np.histogram2d(
        x,
        y,
        bins=(20, 20),
        range=((x.min(), x.max()), (y.min(), y.max())),
        density=True,
    )
    kT = _kT_kJ_per_mol(300.0)
    tiny = np.finfo(float).tiny
    H_clipped = np.clip(H, tiny, None)
    F_ref = np.where(H > 0, -kT * np.log(H_clipped), np.inf)
    F_ref -= np.nanmin(F_ref)
    F_res = np.nan_to_num(res.F, nan=np.inf, posinf=np.inf)
    close = np.isclose(F_res, F_ref, atol=2e-2)
    both_inf = (F_res == np.inf) & (F_ref == np.inf)
    assert np.all(close | both_inf)
    assert np.allclose(res.metadata["counts"], H)


def test_generate_2d_fes_shape_mismatch():
    x = np.array([0.0, 1.0])
    y = np.array([0.0])
    with pytest.raises(ValueError):
        generate_2d_fes(x, y)


def test_generate_1d_pmf_invalid_temperature():
    data = np.array([0.0, 1.0])
    with pytest.raises(ValueError):
        generate_1d_pmf(data, temperature=-1.0)
