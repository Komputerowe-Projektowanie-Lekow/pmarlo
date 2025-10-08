from __future__ import annotations

"""
Build temperature-partitioned DEMUX datasets with per-shard pairs and weights.

This module enforces the Temperature Demux Contract at the dataset boundary:
- Only shards with kind == "demux" are considered.
- Exactly one target temperature is selected; mixing temperatures is forbidden.
- Pairs are constructed within shards only (no cross-shard pairs).
- Optional bias reweighting is incorporated via beta(T) with weights derived
  from the per-frame bias potential.
"""

from dataclasses import dataclass
from typing import Any, Callable, Iterable, List, Optional, Sequence

import numpy as np

from pmarlo import constants as const

from pmarlo.features.pairs import scaled_time_pairs
from pmarlo.utils.errors import TemperatureConsistencyError


@dataclass(frozen=True)
class DemuxDataset:
    temperature_K: float
    shards: list[Any]
    X_list: list[np.ndarray]
    pairs: np.ndarray  # shape (N, 2) with global indices over concatenated X
    weights: (
        np.ndarray
    )  # length N, per-pair weights (geometric mean of per-frame weights)
    dt_ps: Optional[float]


def _is_demux(shard: Any) -> bool:
    kind = getattr(shard, "kind", None)
    if kind is None and hasattr(shard, "source"):
        try:
            src = dict(getattr(shard, "source"))
            kind = src.get("kind")
        except Exception:
            kind = None
    return str(kind) == "demux"


def _temperature_of(shard: Any) -> Optional[float]:
    # Prefer explicit temperature_K attribute (strict schema)
    t = getattr(shard, "temperature_K", None)
    if t is not None:
        try:
            return float(t)
        except Exception:
            return None
    # Fallback to legacy top-level temperature
    t2 = getattr(shard, "temperature", None)
    try:
        return None if t2 is None else float(t2)
    except Exception:
        return None


def _dt_ps_of(shard: Any) -> Optional[float]:
    v = getattr(shard, "dt_ps", None)
    try:
        return None if v is None else float(v)
    except Exception:
        return None


def build_demux_dataset(
    shards: Sequence[Any],
    target_temperature_K: float,
    lag_steps: int,
    feature_fn: Callable[[Any], np.ndarray],
    bias_to_weights_fn: Optional[Callable[[Any, float], np.ndarray]] = None,
) -> DemuxDataset:
    """Build a single-temperature dataset from DEMUX shards.

    Parameters
    ----------
    shards
        Collection of shard metadata objects (strict schema or legacy).
    target_temperature_K
        Temperature to select. Only shards with exactly this temperature are used.
    lag_steps
        Lag in scaled-time units passed to scaled_time_pairs. If bias_to_weights_fn
        is None, this reduces to uniform integer lag pairing.
    feature_fn
        Callable producing per-shard feature matrix X of shape (n_frames, k).
    bias_to_weights_fn
        Optional callable producing per-frame positive weights w_t (e.g., exp(beta*V)).

    Returns
    -------
    DemuxDataset

    Raises
    ------
    TemperatureConsistencyError
        If mixing temperatures is detected or no shards match the target.
    """
    tsel = float(target_temperature_K)
    chosen = _select_target_shards(shards, tsel)
    dt_ps = _extract_dt_ps(chosen)
    X_list, pairs, weights = _assemble_pairs_and_weights(
        chosen,
        tsel,
        lag_steps,
        feature_fn,
        bias_to_weights_fn,
    )

    return DemuxDataset(
        temperature_K=tsel,
        shards=list(chosen),
        X_list=X_list,
        pairs=pairs,
        weights=weights,
        dt_ps=dt_ps,
    )


def _select_target_shards(shards: Sequence[Any], target_K: float) -> list[Any]:
    """Return DEMUX shards at the target temperature, enforcing presence."""

    chosen: list[Any] = []
    for shard in shards:
        if not _is_demux(shard):
            continue
        temperature = _temperature_of(shard)
        if temperature is None:
            raise TemperatureConsistencyError("DEMUX shard missing temperature_K")
        if abs(float(temperature) - target_K) <= const.NUMERIC_ABSOLUTE_TOLERANCE:
            chosen.append(shard)

    if not chosen:
        raise TemperatureConsistencyError(
            f"No DEMUX shards at target temperature {target_K} K"
        )
    return chosen


def _extract_dt_ps(shards: Sequence[Any]) -> float | None:
    """Ensure the per-shard dt_ps values agree."""

    dt_vals = [_dt_ps_of(shard) for shard in shards]
    dt_set = {round(float(val), 12) for val in dt_vals if val is not None}
    if len(dt_set) > 1:
        raise TemperatureConsistencyError(
            f"Mismatched dt_ps across shards: {sorted(dt_set)}"
        )
    if dt_set:
        return float(next(iter(dt_set)))
    return None


def _assemble_pairs_and_weights(
    shards: Sequence[Any],
    temperature_K: float,
    lag_steps: int,
    feature_fn: Callable[[Any], np.ndarray],
    bias_to_weights_fn: Optional[Callable[[Any, float], np.ndarray]],
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """Compute features, index pairs, and pair weights for each shard."""

    X_list: list[np.ndarray] = []
    idx_t_parts: list[np.ndarray] = []
    idx_tau_parts: list[np.ndarray] = []
    weight_parts: list[np.ndarray] = []
    offset = 0

    for shard in shards:
        X = np.asarray(feature_fn(shard), dtype=np.float64)
        if X.ndim != 2 or X.shape[0] <= 1:
            raise ValueError("feature_fn must return a 2D array with >=2 frames")
        X_list.append(X)

        frame_weights, log_weights = _compute_frame_weights(
            shard,
            temperature_K,
            X.shape[0],
            bias_to_weights_fn,
        )

        idx_t, idx_tau = scaled_time_pairs(
            int(X.shape[0]),
            log_weights,
            tau_scaled=float(lag_steps),
        )
        if idx_t.size:
            idx_t_parts.append(offset + idx_t)
            idx_tau_parts.append(offset + idx_tau)
            weight_parts.append(_pair_weights(frame_weights, idx_t, idx_tau))
        offset += int(X.shape[0])

    if idx_t_parts:
        idx_t_all = np.concatenate(idx_t_parts).astype(np.int64, copy=False)
        idx_tau_all = np.concatenate(idx_tau_parts).astype(np.int64, copy=False)
        pairs = np.column_stack([idx_t_all, idx_tau_all]).astype(np.int64, copy=False)
        weights = (
            np.concatenate(weight_parts).astype(np.float64, copy=False)
            if weight_parts
            else np.ones((idx_t_all.shape[0],), dtype=np.float64)
        )
    else:
        pairs = np.empty((0, 2), dtype=np.int64)
        weights = np.empty((0,), dtype=np.float64)

    return X_list, pairs, weights


def _compute_frame_weights(
    shard: Any,
    temperature_K: float,
    n_frames: int,
    bias_to_weights_fn: Optional[Callable[[Any, float], np.ndarray]],
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Return per-frame weights and their logarithms when biasing is requested."""

    if bias_to_weights_fn is None:
        return None, None

    weights = np.asarray(
        bias_to_weights_fn(shard, temperature_K),
        dtype=np.float64,
    ).reshape(-1)
    if weights.shape[0] != n_frames:
        raise ValueError("bias_to_weights_fn length must match frames in X")
    if np.any(weights <= 0) or not np.all(np.isfinite(weights)):
        raise ValueError("weights must be positive and finite")
    return weights, np.log(weights)


def _pair_weights(
    frame_weights: np.ndarray | None,
    idx_t: np.ndarray,
    idx_tau: np.ndarray,
) -> np.ndarray:
    """Return geometric-mean pair weights, or ones if weights are absent."""

    if frame_weights is None:
        return np.ones_like(idx_t, dtype=np.float64)
    return np.sqrt(frame_weights[idx_t] * frame_weights[idx_tau]).astype(
        np.float64, copy=False
    )


def validate_demux_coverage(shards: Iterable[Any]) -> dict:
    """Summarize available DEMUX temperatures and total frames per temperature.

    Returns mapping {"temperatures": sorted list, "frames": {T: frames}}.
    """
    temps: List[float] = []
    frames_by_T: dict[float, int] = {}
    for s in shards:
        if not _is_demux(s):
            continue
        t = _temperature_of(s)
        if t is None:
            continue
        temps.append(float(t))
        n = int(getattr(s, "n_frames", 0) or 0)
        frames_by_T[float(t)] = frames_by_T.get(float(t), 0) + max(0, n)
    out_temps = sorted(set(round(float(x), 6) for x in temps))
    # normalize keys as floats
    frames_norm = {float(k): int(v) for k, v in frames_by_T.items()}
    return {"temperatures": [float(x) for x in out_temps], "frames": frames_norm}
