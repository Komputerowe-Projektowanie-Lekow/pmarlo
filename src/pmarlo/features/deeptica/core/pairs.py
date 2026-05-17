"""Core helpers for building lagged index pairs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np

from pmarlo.utils.array import concatenate_or_empty

__all__ = ["PairInfo", "build_pair_info"]


@dataclass(slots=True)
class PairInfo:
    idx_t: np.ndarray
    idx_tau: np.ndarray
    weights: np.ndarray
    diagnostics: dict[str, object]


def build_pair_info(
    trajectories: Sequence[np.ndarray],
    tau_schedule: Sequence[int],
    *,
    pairs: Tuple[np.ndarray, np.ndarray] | None = None,
    weights: np.ndarray | None = None,
) -> PairInfo:
    schedule = _normalize_schedule(tau_schedule)
    idx_t, idx_tau = _derive_pairs(trajectories, schedule, pairs)
    weights_arr = _normalize_weights(idx_t.size, weights)
    diagnostics = _compute_diagnostics(trajectories, schedule, idx_t, idx_tau)
    return PairInfo(
        idx_t=idx_t, idx_tau=idx_tau, weights=weights_arr, diagnostics=diagnostics
    )


def _normalize_schedule(schedule: Sequence[int]) -> tuple[int, ...]:
    normalized = tuple(int(t) for t in schedule if int(t) > 0)
    if not normalized:
        raise ValueError("Tau schedule must contain at least one positive lag")
    return normalized


def _derive_pairs(
    trajectories: Sequence[np.ndarray],
    schedule: tuple[int, ...],
    pairs: Tuple[np.ndarray, np.ndarray] | None,
) -> tuple[np.ndarray, np.ndarray]:
    if pairs is not None:
        idx_t = np.asarray(pairs[0], dtype=np.int64).reshape(-1)
        idx_tau = np.asarray(pairs[1], dtype=np.int64).reshape(-1)
        total_length = sum(int(np.asarray(block).shape[0]) for block in trajectories)
        _validate_pairs(idx_t, idx_tau, total_length)
        return idx_t, idx_tau

    if len(schedule) > 1:
        return _concatenate_pairs(trajectories, schedule)
    return _build_uniform_pairs(trajectories, int(schedule[0]))


def _normalize_weights(count: int, weights: np.ndarray | None) -> np.ndarray:
    if weights is None:
        return np.ones((count,), dtype=np.float32)
    arr = np.asarray(weights, dtype=np.float32).reshape(-1)
    if count == 0:
        return arr[:0]
    if arr.size == 1 and count > 1:
        return np.full((count,), float(arr[0]), dtype=np.float32)
    if arr.size != count:
        raise ValueError("weights must match the number of lagged pairs")
    return arr


def _compute_diagnostics(
    trajectories: Sequence[np.ndarray],
    schedule: tuple[int, ...],
    idx_t: np.ndarray,
    idx_tau: np.ndarray,
) -> dict[str, object]:
    lengths = [int(np.asarray(block).shape[0]) for block in trajectories]
    max_tau = int(max(schedule))
    short_trajectories = [i for i, length in enumerate(lengths) if length <= max_tau]

    if len(schedule) > 1:
        total_possible = sum(
            sum(max(0, length - tau) for length in lengths) for tau in schedule
        )
    else:
        total_possible = sum(max(0, length - max_tau) for length in lengths)

    usable_pairs = int(min(idx_t.size, idx_tau.size))
    coverage = float(usable_pairs / total_possible) if total_possible else 0.0
    offsets = np.cumsum([0, *lengths])
    pairs_by_trajectory = [
        int(np.count_nonzero((idx_t >= start) & (idx_t < end)))
        for start, end in zip(offsets[:-1], offsets[1:])
    ]
    return {
        "usable_pairs": usable_pairs,
        "pair_coverage": coverage,
        "total_possible_pairs": int(total_possible),
        "short_trajectories": short_trajectories,
        "pairs_by_trajectory": pairs_by_trajectory,
        "lag_used": max_tau,
        "expected_pairs": int(total_possible),
        "tau_schedule_used": list(schedule),
    }


def _concatenate_pairs(
    trajectories: Sequence[np.ndarray], schedule: tuple[int, ...]
) -> tuple[np.ndarray, np.ndarray]:
    idx_parts: list[np.ndarray] = []
    tau_parts: list[np.ndarray] = []
    for tau in schedule:
        i, j = _build_uniform_pairs(trajectories, int(tau))
        if i.size and j.size:
            idx_parts.append(i)
            tau_parts.append(j)
    idx = concatenate_or_empty(idx_parts, dtype=np.int64, shape=(0,), copy=False)
    tau = concatenate_or_empty(tau_parts, dtype=np.int64, shape=(0,), copy=False)
    return idx, tau


def _build_uniform_pairs(
    trajectories: Sequence[np.ndarray],
    lag: int,
) -> tuple[np.ndarray, np.ndarray]:
    lag = int(lag)
    if lag < 1:
        raise ValueError("Lag must be a positive integer")
    idx_parts: list[np.ndarray] = []
    tau_parts: list[np.ndarray] = []
    offset = 0
    for block in trajectories:
        n = int(np.asarray(block).shape[0])
        if n > lag:
            i = np.arange(0, n - lag, dtype=np.int64)
            j = i + lag
            idx_parts.append(offset + i)
            tau_parts.append(offset + j)
        offset += n
    idx = concatenate_or_empty(idx_parts, dtype=np.int64, shape=(0,), copy=False)
    tau = concatenate_or_empty(tau_parts, dtype=np.int64, shape=(0,), copy=False)
    return idx, tau


def _validate_pairs(idx_t: np.ndarray, idx_tau: np.ndarray, total_length: int) -> None:
    if idx_t.shape != idx_tau.shape:
        raise ValueError("Pair index arrays must have the same shape")
    if idx_t.ndim != 1 or idx_tau.ndim != 1:
        raise ValueError("Pair index arrays must be one-dimensional")
    if idx_t.size == 0:
        return
    if np.min(idx_t) < 0 or np.min(idx_tau) < 0:
        raise ValueError("Pair indices must be non-negative")
    if total_length <= 0:
        raise ValueError("Pair indices provided but no trajectories are available")
    max_valid = total_length - 1
    if np.max(idx_t) > max_valid or np.max(idx_tau) > max_valid:
        raise ValueError("Pair indices exceed available trajectory length")
    shift = idx_tau - idx_t
    if np.min(shift) <= 0:
        raise ValueError("Pair indices must represent positive time lags")
