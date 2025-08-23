from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter


@dataclass
class PMFResult:
    """Result of a one-dimensional potential of mean force calculation."""

    F: np.ndarray
    edges: np.ndarray
    counts: np.ndarray
    periodic: bool
    temperature: float


@dataclass
class FESResult:
    """Result of a two-dimensional free-energy surface calculation."""

    F: np.ndarray
    xedges: np.ndarray
    yedges: np.ndarray
    counts: np.ndarray
    periodic: Tuple[bool, bool]
    temperature: float


def _kT_kJ_per_mol(temperature_kelvin: float) -> float:
    from scipy import constants

    # Cast to float because scipy.constants may be typed as Any
    return float(constants.k * temperature_kelvin * constants.Avogadro / 1000.0)


def generate_1d_pmf(
    cv: np.ndarray,
    bins: int = 100,
    temperature: float = 300.0,
    periodic: bool = False,
    range_: Optional[Tuple[float, float]] = None,
    smoothing_sigma: Optional[float] = None,
) -> PMFResult:
    """Generate a one-dimensional potential of mean force (PMF).

    Parameters
    ----------
    cv
        Collective variable samples.
    bins
        Number of histogram bins; must be positive.
    temperature
        Simulation temperature in Kelvin; must be positive.
    periodic
        Whether the CV is periodic.
    range_
        Optional histogram range as ``(min, max)``.
    smoothing_sigma
        Standard deviation for Gaussian smoothing; must be non-negative.
    """

    cv = np.asarray(cv, dtype=float).reshape(-1)
    if cv.size == 0:
        raise ValueError("cv array must not be empty")
    if bins <= 0:
        raise ValueError("bins must be positive")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if smoothing_sigma is not None and smoothing_sigma < 0:
        raise ValueError("smoothing_sigma must be non-negative")

    if range_ is None:
        hist_range = (float(np.min(cv)), float(np.max(cv)))
    else:
        hist_range = (float(range_[0]), float(range_[1]))
    if not np.isfinite(hist_range).all() or hist_range[0] >= hist_range[1]:
        raise ValueError("range_ must be finite with min < max")

    H, edges = np.histogram(cv, bins=bins, range=hist_range, density=True)
    if smoothing_sigma and smoothing_sigma > 0:
        H = gaussian_filter(
            H, sigma=float(smoothing_sigma), mode="wrap" if periodic else "reflect"
        )
    kT = _kT_kJ_per_mol(temperature)
    tiny = np.finfo(float).tiny
    H_clipped = np.clip(H, tiny, None)
    F = np.where(H > 0, -kT * np.log(H_clipped), np.inf)
    if np.any(np.isfinite(F)):
        F -= np.nanmin(F)
    return PMFResult(F=F, edges=edges, counts=H, periodic=periodic, temperature=temperature)


def generate_2d_fes(
    cv1: np.ndarray,
    cv2: np.ndarray,
    bins: Tuple[int, int] = (100, 100),
    temperature: float = 300.0,
    periodic: Tuple[bool, bool] = (False, False),
    ranges: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
    smoothing_sigma: Optional[float] = 0.6,
) -> FESResult:
    """Generate a two-dimensional free-energy surface (FES)."""

    x = np.asarray(cv1, dtype=float).reshape(-1)
    y = np.asarray(cv2, dtype=float).reshape(-1)
    if x.size == 0 or y.size == 0:
        raise ValueError("cv1 and cv2 must not be empty")
    if x.shape != y.shape:
        raise ValueError("cv1 and cv2 must have the same shape")
    if len(bins) != 2 or any(b <= 0 for b in bins):
        raise ValueError("bins must be a tuple of two positive integers")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if len(periodic) != 2:
        raise ValueError("periodic must be a tuple of two booleans")
    if smoothing_sigma is not None and smoothing_sigma < 0:
        raise ValueError("smoothing_sigma must be non-negative")

    if ranges is None:
        xr = (float(np.min(x)), float(np.max(x)))
        yr = (float(np.min(y)), float(np.max(y)))
    else:
        if len(ranges) != 2 or any(len(r) != 2 for r in ranges):
            raise ValueError("ranges must be ((xmin, xmax), (ymin, ymax))")
        xr = (float(ranges[0][0]), float(ranges[0][1]))
        yr = (float(ranges[1][0]), float(ranges[1][1]))
    if not np.isfinite(xr + yr).all() or xr[0] >= xr[1] or yr[0] >= yr[1]:
        raise ValueError("ranges must be finite with min < max for both axes")

    H, xedges, yedges = np.histogram2d(x, y, bins=bins, range=(xr, yr), density=True)
    if smoothing_sigma and smoothing_sigma > 0:
        mode = tuple("wrap" if p else "reflect" for p in periodic)
        H = gaussian_filter(H, sigma=float(smoothing_sigma), mode=mode)
    kT = _kT_kJ_per_mol(temperature)
    tiny = np.finfo(float).tiny
    H_clipped = np.clip(H, tiny, None)
    F = np.where(H > 0, -kT * np.log(H_clipped), np.inf)
    if np.any(np.isfinite(F)):
        F -= np.nanmin(F)
    return FESResult(
        F=F,
        xedges=xedges,
        yedges=yedges,
        counts=H,
        periodic=periodic,
        temperature=temperature,
    )
