from __future__ import annotations

"""Uniform time-lagged pair construction for Deep-TICA training."""

from typing import List

import numpy as np


def lagged_time_pairs(
    length: int,
    lag: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return index pairs ``(i, i + lag)`` for one trajectory.

    Parameters
    ----------
    length:
        Number of frames in the trajectory.
    lag:
        Positive integer frame lag.
    """

    lag = int(lag)
    if lag <= 0:
        raise ValueError("lag must be a positive integer")
    if length <= 1:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    if lag >= length:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    i = np.arange(0, length - lag, dtype=np.int64)
    j = i + lag
    return i, j


def make_training_pairs_from_trajectory(
    trajectory: np.ndarray,
    lag: int,
) -> tuple[List[np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Build uniform time-lagged training pairs for one trajectory.

    Parameters
    ----------
    trajectory:
        Feature matrix for the trajectory.
    lag:
        Positive integer frame lag.

    Returns
    -------
    X_list, (idx_t, idx_tlag)
        Single feature block and index pairs over that block.
    """

    X = np.asarray(trajectory, dtype=np.float64)
    idx_t, idx_tlag = lagged_time_pairs(int(X.shape[0]), int(lag))
    return [X], (idx_t, idx_tlag)
