"""Utility functions for Markov State Model calculations."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

EPS = 1e-12


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
    clipped: NDArray[np.float64] = np.clip(eig, eps, 1 - eps).astype(
        np.float64, copy=False
    )
    logs: NDArray[np.float64] = np.log(clipped).astype(np.float64, copy=False)
    lag64 = np.float64(lag)
    # Use out-parameter to avoid creating an Any-typed temporary
    timescales: NDArray[np.float64] = np.empty_like(logs, dtype=np.float64)
    np.divide(-lag64, logs, out=timescales)
    invalid: NDArray[np.bool_] = (eig <= 0) | (eig >= 1)
    timescales[invalid] = np.nan
    return timescales


def format_lag_window_ps(window: tuple[float, float]) -> str:
    """Return a pretty string for a lag-time window in picoseconds."""

    start_ps, end_ps = window
    return f"{start_ps:.3f}–{end_ps:.3f} ps"
