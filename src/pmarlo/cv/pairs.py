from __future__ import annotations

"""
Scaled-time pair construction for Deep-TICA training.

This module provides utilities to construct time-lagged index pairs using
scaled time t' where delta t' = exp(beta * V(s_t)) * delta t with discrete
delta t = 1 per frame (units assumed consistent with the provided bias).

If no bias information is provided, the functions fall back to uniform-time
pairs equivalent to VAC/VAMP on unbiased data.
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
        Per-frame log-weights, log w_t = beta * V(s_t). If None or empty,
        uses uniform time with integer lag round(tau_scaled).
    tau_scaled:
        Target lag in scaled time units.
    jitter:
        Unused placeholder for potential future widening of the acceptance
        window. Currently ignored to keep pair selection deterministic.
    """

    if length <= 1:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    if logw is None or getattr(logw, "size", 0) == 0:
        lag = max(1, int(round(float(tau_scaled))))
        if lag >= length:
            return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
        i = np.arange(0, length - lag, dtype=np.int64)
        j = i + lag
        return i, j

    lw = np.asarray(logw, dtype=np.float64).reshape(-1)
    if lw.shape[0] != length:
        raise ValueError("logw length must match the number of frames")
    # Avoid numerical overflow/underflow in exp
    wt = np.exp(np.clip(lw, -80.0, 80.0))
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
        and bias/temperature are used to build pairs.
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
        # Compute log-weights if bias and temperature are available
        logw = None
        if bias is not None and T is not None:
            b = np.asarray(bias, dtype=np.float64).reshape(-1)
            if b.shape[0] != n:
                raise ValueError("bias_potential length must match frames in X")
            beta = 1.0 / (8.31446261815324e-3 * float(T))  # 1/(k_B T) in mol/kJ
            logw = beta * b

        i, j = scaled_time_pairs(n, logw, tau_scaled)
        if i.size:
            idx_t_parts.append(offset + i)
            idx_tlag_parts.append(offset + j)
        offset += n

    if idx_t_parts:
        idx_t = np.concatenate(idx_t_parts).astype(np.int64, copy=False)
        idx_tlag = np.concatenate(idx_tlag_parts).astype(np.int64, copy=False)
    else:
        idx_t = np.array([], dtype=np.int64)
        idx_tlag = np.array([], dtype=np.int64)

    return X_list, (idx_t, idx_tlag)
