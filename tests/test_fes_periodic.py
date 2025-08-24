import numpy as np

from pmarlo.fes import generate_2d_fes
from pmarlo.fes.surfaces import periodic_kde_2d


def test_periodic_kde_mass_conservation():
    rng = np.random.default_rng(0)
    x = rng.uniform(-np.pi, np.pi, size=100)
    y = rng.uniform(-np.pi, np.pi, size=100)
    dens = periodic_kde_2d(x, y, bw=(0.35, 0.35), gridsize=(42, 42))
    area = (2 * np.pi / 42) * (2 * np.pi / 42)
    assert np.isclose(dens.sum() * area, 1.0, atol=1e-3)


def test_kde_blending_reduces_holes():
    rng = np.random.default_rng(1)
    x = rng.uniform(-np.pi, np.pi, size=20)
    y = rng.uniform(-np.pi, np.pi, size=20)
    x_deg = np.degrees(x)
    y_deg = np.degrees(y)
    hist = generate_2d_fes(
        x_deg,
        y_deg,
        bins=(42, 42),
        temperature=300.0,
        periodic=(True, True),
        smooth=False,
        inpaint=False,
        min_count=5,
    )
    kde = generate_2d_fes(
        x_deg,
        y_deg,
        bins=(42, 42),
        temperature=300.0,
        periodic=(True, True),
        smooth=True,
        inpaint=True,
        min_count=5,
    )
    holes_hist = np.isnan(hist.F).sum()
    holes_kde = np.isnan(kde.F).sum()
    assert holes_kde < holes_hist


def test_ring_continuity_across_boundaries():
    theta = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    phi = 180.0 + 40.0 * np.cos(theta)
    psi = 40.0 * np.sin(theta)
    res1 = generate_2d_fes(
        phi,
        psi,
        bins=(36, 36),
        temperature=300.0,
        periodic=(True, True),
        ranges=((-180.0, 180.0), (-180.0, 180.0)),
        smooth=False,
        inpaint=False,
        min_count=1,
    )
    res2 = generate_2d_fes(
        phi + 360.0,
        psi,
        bins=(36, 36),
        temperature=300.0,
        periodic=(True, True),
        ranges=((-180.0, 180.0), (-180.0, 180.0)),
        smooth=False,
        inpaint=False,
        min_count=1,
    )
    assert np.allclose(res1.F, res2.F, equal_nan=True)
