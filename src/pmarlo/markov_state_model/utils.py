"""Utility functions for Markov State Model calculations."""

from __future__ import annotations

import numpy as np

EPS = 1e-12


def safe_timescales(lag: float, eigvals: np.ndarray, eps: float = EPS) -> np.ndarray:
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
    eigvals = np.asarray(eigvals, dtype=float)
    clipped = np.clip(eigvals, eps, 1 - eps)
    timescales = -lag / np.log(clipped)
    invalid = (eigvals <= 0) | (eigvals >= 1)
    timescales[invalid] = np.nan
    return timescales
