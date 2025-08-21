from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter


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
) -> Dict[str, Any]:
    cv = np.asarray(cv, dtype=float).reshape(-1)
    if range_ is None:
        range_ = (float(np.min(cv)), float(np.max(cv)))
    H, edges = np.histogram(cv, bins=bins, range=range_, density=True)
    if smoothing_sigma and smoothing_sigma > 0:
        H = gaussian_filter(
            H, sigma=float(smoothing_sigma), mode="wrap" if periodic else "reflect"
        )
    kT = _kT_kJ_per_mol(temperature)
    F = np.full_like(H, np.inf, dtype=float)
    mask = H > 1e-12
    F[mask] = -kT * np.log(H[mask])
    if np.any(np.isfinite(F)):
        F -= np.nanmin(F)
    return {
        "F": F,
        "edges": edges,
        "counts": H,
        "periodic": periodic,
        "temperature": temperature,
    }


def generate_2d_fes(
    cv1: np.ndarray,
    cv2: np.ndarray,
    bins: Tuple[int, int] = (100, 100),
    temperature: float = 300.0,
    periodic: Tuple[bool, bool] = (False, False),
    ranges: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
    smoothing_sigma: Optional[float] = 0.6,
) -> Dict[str, Any]:
    x = np.asarray(cv1, dtype=float).reshape(-1)
    y = np.asarray(cv2, dtype=float).reshape(-1)
    if ranges is None:
        ranges = (
            (float(np.min(x)), float(np.max(x))),
            (float(np.min(y)), float(np.max(y))),
        )
    H, xedges, yedges = np.histogram2d(x, y, bins=bins, range=ranges, density=True)
    if smoothing_sigma and smoothing_sigma > 0:
        mode = "wrap" if any(periodic) else "reflect"
        H = gaussian_filter(H, sigma=float(smoothing_sigma), mode=mode)
    kT = _kT_kJ_per_mol(temperature)
    F = np.full_like(H, np.inf, dtype=float)
    mask = H > 1e-12
    F[mask] = -kT * np.log(H[mask])
    if np.any(np.isfinite(F)):
        F -= np.nanmin(F)
    return {
        "F": F,
        "xedges": xedges,
        "yedges": yedges,
        "counts": H,
        "periodic": periodic,
        "temperature": temperature,
    }
