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
        min_count=5,
    )
    kde = generate_2d_fes(
        x_deg,
        y_deg,
        bins=(42, 42),
        temperature=300.0,
        periodic=(True, True),
        smooth=True,
        min_count=5,
    )
    holes_hist = np.isnan(hist.F).sum()
    holes_kde = np.isnan(kde.F).sum()
    assert holes_kde < holes_hist
