"""Utility functions for Markov State Model calculations."""

from __future__ import annotations

import numpy as np
from deeptime.markov.tools.analysis import (
    timescales_from_eigenvalues as _dt_timescales_from_eigenvalues,
)
from numpy.typing import NDArray

from pmarlo import constants as const

EPS = const.NUMERIC_MIN_POSITIVE


def safe_timescales(
    lag: float, eigvals: NDArray[np.float64], eps: float = EPS
) -> NDArray[np.float64]:
    """Compute implied timescales while handling numerically unstable eigenvalues.

    Parameters
    ----------
    lag:
        Lag time used in the MSM.
    eigvals:
        Eigenvalues of the transition matrix.
    eps:
        Small value to clip eigenvalues away from 0 and 1.

    Returns
    -------
    np.ndarray
        Array of implied timescales. Eigenvalues outside the open interval
        ``(0, 1)`` yield ``np.nan`` timescales.
    """
    eig: NDArray[np.float64] = np.asarray(eigvals, dtype=np.float64)
    if eig.size == 0:
        return np.empty_like(eig, dtype=np.float64)

    clipped: NDArray[np.float64] = np.clip(eig, eps, 1 - eps).astype(
        np.float64, copy=False
    )
    flat_clipped = clipped.reshape(-1).astype(np.complex128, copy=False)
    ts_flat = _dt_timescales_from_eigenvalues(flat_clipped, tau=float(lag))
    timescales = np.asarray(ts_flat, dtype=np.float64).reshape(clipped.shape)

    invalid: NDArray[np.bool_] = (~np.isfinite(eig)) | (eig <= 0) | (eig >= 1)
    timescales = timescales.astype(np.float64, copy=False)
    timescales[invalid] = np.nan
    return timescales


def format_lag_window_ps(window: tuple[float, float]) -> str:
    """Return a pretty string for a lag-time window in picoseconds."""

    start_ps, end_ps = window
    return f"{start_ps:.3f}–{end_ps:.3f} ps"
