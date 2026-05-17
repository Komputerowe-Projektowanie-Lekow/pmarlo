from __future__ import annotations

"""Diagnostics for Deep-TICA pair availability in one trajectory."""

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from pmarlo.features.pairs import lagged_time_pairs


@dataclass(frozen=True)
class PairDiagItem:
    """Pair-count diagnostic for one trajectory."""

    id: str
    frames: int
    pairs: int
    frames_leq_lag: bool


@dataclass(frozen=True)
class PairDiagReport:
    """Aggregate diagnostic report for Deep-TICA lagged pairs."""

    lag_used: int
    n_trajectories: int
    frames_total: int
    pairs_total: int
    trajectory: PairDiagItem
    message: str
    too_short_count: int


def diagnose_deeptica_pairs(dataset: Any, lag: int) -> PairDiagReport:
    """Compute lagged pair counts for a single feature matrix without training."""

    features = _extract_feature_matrix(dataset)
    if features is None:
        raise ValueError("No dataset provided or dataset lacks an 'X' array")

    lag = int(lag)
    idx_t, _ = lagged_time_pairs(int(features.shape[0]), lag)
    pairs_total = int(idx_t.size)
    n_frames = int(features.shape[0])
    too_short = n_frames <= lag

    item = PairDiagItem(
        id="trajectory",
        frames=n_frames,
        pairs=pairs_total,
        frames_leq_lag=too_short,
    )
    message = f"Uniform pairs: {pairs_total} total for {n_frames} frames at lag={lag}."
    if too_short:
        message += " Dataset too short for requested lag."

    return PairDiagReport(
        lag_used=lag,
        n_trajectories=1,
        frames_total=n_frames,
        pairs_total=pairs_total,
        trajectory=item,
        message=message,
        too_short_count=1 if too_short else 0,
    )


def _extract_feature_matrix(dataset: Any) -> Optional[np.ndarray]:
    if not isinstance(dataset, dict) or "X" not in dataset:
        return None
    try:
        matrix = np.asarray(dataset.get("X"), dtype=np.float64)
    except Exception:
        return None
    if matrix.ndim != 2:
        return None
    return matrix
