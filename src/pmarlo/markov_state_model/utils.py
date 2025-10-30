"""Utility functions for Markov State Model calculations."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from pmarlo import constants as const

EPS = const.NUMERIC_MIN_POSITIVE

ComplexOrReal = np.complexfloating[Any, Any] | np.floating[Any]


def safe_timescales(
    lag: float, eigvals: NDArray[ComplexOrReal], eps: float = EPS
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
    eig = np.asarray(eigvals)
    if eig.size == 0:
        return np.empty_like(eig, dtype=np.float64)

    eig_complex = eig.astype(np.complex128, copy=False)
    magnitudes = np.abs(eig_complex)
    clipped = np.clip(magnitudes, eps, 1 - eps)
    flat_clipped = clipped.reshape(-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ts_flat = -float(lag) / np.log(flat_clipped)
    timescales = np.asarray(ts_flat, dtype=np.float64).reshape(eig_complex.shape)

    invalid_magnitude: NDArray[np.bool_] = (
        ~np.isfinite(magnitudes) | (magnitudes <= 0) | (magnitudes >= 1)
    )
    real_mask = np.isclose(eig_complex.imag, 0.0)
    invalid_real = real_mask & (
        (eig_complex.real <= 0.0) | (eig_complex.real >= 1.0)
    )
    invalid: NDArray[np.bool_] = invalid_magnitude | invalid_real
    timescales = timescales.astype(np.float64, copy=False)
    timescales[invalid] = np.nan
    return timescales


def format_lag_window_ps(window: tuple[float, float]) -> str:
    """Return a pretty string for a lag-time window in picoseconds."""

    start_ps, end_ps = window
    return f"{start_ps:.3f}–{end_ps:.3f} ps"
