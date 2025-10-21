from __future__ import annotations

"""
Scaled-time pair construction for Deep-TICA training.

This module provides utilities to construct time-lagged index pairs using
scaled time t' where delta t' = exp(beta * V(s_t)) * delta t with discrete
delta t = 1 per frame (units assumed consistent with the provided bias).
"""

from typing import Iterable, List

import numpy as np


def scaled_time_pairs(
    length: int,
    logw: np.ndarray | None,
    tau_scaled: float,
    jitter: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return indices (i, j) such that scaled_time[j] - scaled_time[i] ≈ tau_scaled.

    Parameters
    ----------
    length:
        Number of frames in the shard.
    logw:
        Per-frame log-weights, log w_t = beta * V(s_t). When omitted, uniform
        time increments are assumed and `tau_scaled` is rounded to the nearest
        integer frame lag.
    tau_scaled:
        Target lag in scaled time units.
    jitter:
        Unused placeholder for potential future widening of the acceptance
        window. Currently ignored to keep pair selection deterministic.
    """

    if length <= 1:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    if logw is None:
        lag = int(round(float(tau_scaled)))
        if lag <= 0:
            raise ValueError("tau_scaled must be positive when log weights are absent")
        if lag >= length:
            return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

        i = np.arange(0, length - lag, dtype=np.int64)
        j = i + lag
        return i, j

    if getattr(logw, "size", 0) == 0:
        raise ValueError("Scaled-time pairing requires per-frame log weights")

    lw = np.asarray(logw, dtype=np.float64).reshape(-1)
    if lw.shape[0] != length:
        raise ValueError("logw length must match the number of frames")
    if not np.all(np.isfinite(lw)):
        raise ValueError("logw must contain finite values")
    if np.any(lw > 700.0) or np.any(lw < -700.0):
        raise OverflowError("logw values would overflow exp during scaling")

    wt = np.exp(lw)
    st = np.cumsum(wt)
    targets = st + float(tau_scaled)
    j = np.searchsorted(st, targets, side="left")
    j = np.minimum(j, length - 1)
    i = np.arange(length, dtype=np.int64)
    mask = j > i
    i = i[mask]
    j = j[mask]
    return i, j


def make_training_pairs_from_shards(
    shard_records: Iterable[
        tuple[np.ndarray, np.ndarray | None, np.ndarray | None, float | None]
    ],
    tau_scaled: float,
) -> tuple[List[np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Build concatenated training pairs across shards.

    Parameters
    ----------
    shard_records:
        Iterable of tuples (X, dtraj, bias_potential, temperature_K). Only X
        and bias/temperature are used to build pairs. When either bias or
        temperature is missing, uniform time spacing is assumed.
    tau_scaled:
        Target scaled-time lag passed to :func:`scaled_time_pairs`.

    Returns
    -------
    X_list, (idx_t, idx_tlag)
        Feature blocks and global index pairs over the concatenated X.
    """

    X_list: List[np.ndarray] = []
    idx_t_parts: List[np.ndarray] = []
    idx_tlag_parts: List[np.ndarray] = []
    offset = 0

    for rec in shard_records:
        X, _d, bias, T = rec
        X = np.asarray(X, dtype=np.float64)
        n = int(X.shape[0])
        X_list.append(X)
        if bias is not None and T is not None:
            beta = 1.0 / (0.008314462618 * float(T))  # kJ/mol/K to 1/kJ/mol
            logw = beta * np.asarray(bias, dtype=np.float64)
            if logw.shape[0] != n:
                raise ValueError("Bias potential must match number of frames")
        else:
            logw = None

        i, j = scaled_time_pairs(n, logw, tau_scaled)
        if len(i) > 0:
            idx_t_parts.append(offset + i)
            idx_tlag_parts.append(offset + j)
        offset += n

    if not idx_t_parts:
        return X_list, (np.array([], dtype=np.int64), np.array([], dtype=np.int64))

    idx_t = np.concatenate(idx_t_parts).astype(np.int64, copy=False)
    idx_tlag = np.concatenate(idx_tlag_parts).astype(np.int64, copy=False)
    return X_list, (idx_t, idx_tlag)
