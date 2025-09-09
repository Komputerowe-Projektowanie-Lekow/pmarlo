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

from pmarlo.utils.errors import TemperatureConsistencyError
from pmarlo.features.pairs import scaled_time_pairs


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
    # 1) filter to DEMUX and exact temperature
    chosen: List[Any] = []
    temps_seen: List[float] = []
    for s in shards:
        if not _is_demux(s):
            continue
        t = _temperature_of(s)
        if t is None:
            raise TemperatureConsistencyError("DEMUX shard missing temperature_K")
        temps_seen.append(float(t))
        if abs(float(t) - tsel) <= 1e-6:
            chosen.append(s)

    if not chosen:
        raise TemperatureConsistencyError(
            f"No DEMUX shards at target temperature {tsel} K"
        )

    # 2) ensure dt_ps identical (or all None)
    dt_vals = [_dt_ps_of(s) for s in chosen]
    dt_set = {round(float(x), 12) for x in dt_vals if x is not None}
    dt_ps = None
    if len(dt_set) > 1:
        raise TemperatureConsistencyError(
            f"Mismatched dt_ps across shards: {sorted(dt_set)}"
        )
    if len(dt_set) == 1:
        dt_ps = float(next(iter(dt_set)))

    # 3) build X per shard and construct within-shard pairs/weights
    X_list: List[np.ndarray] = []
    idx_t_parts: List[np.ndarray] = []
    idx_tau_parts: List[np.ndarray] = []
    w_parts: List[np.ndarray] = []
    offset = 0

    for s in chosen:
        X = np.asarray(feature_fn(s), dtype=np.float64)
        if X.ndim != 2 or X.shape[0] <= 1:
            raise ValueError("feature_fn must return a 2D array with >=2 frames")
        n = int(X.shape[0])
        X_list.append(X)
        # per-frame weights
        w_frame = None
        if bias_to_weights_fn is not None:
            w_frame = np.asarray(bias_to_weights_fn(s, tsel), dtype=np.float64).reshape(
                -1
            )
            if w_frame.shape[0] != n:
                raise ValueError("bias_to_weights_fn length must match frames in X")
            if np.any(w_frame <= 0) or not np.all(np.isfinite(w_frame)):
                raise ValueError("weights must be positive and finite")
            logw = np.log(w_frame)
        else:
            logw = None
        i, j = scaled_time_pairs(n, logw, tau_scaled=float(lag_steps))
        if i.size:
            idx_t_parts.append(offset + i)
            idx_tau_parts.append(offset + j)
            # pair weights: geometric mean of per-frame weights; default ones
            if w_frame is None:
                w_parts.append(np.ones_like(i, dtype=np.float64))
            else:
                w_parts.append(
                    np.sqrt(w_frame[i] * w_frame[j]).astype(np.float64, copy=False)
                )
        offset += n

    if idx_t_parts:
        idx_t = np.concatenate(idx_t_parts).astype(np.int64, copy=False)
        idx_tau = np.concatenate(idx_tau_parts).astype(np.int64, copy=False)
        pairs = np.column_stack([idx_t, idx_tau]).astype(np.int64, copy=False)
        weights = (
            np.concatenate(w_parts).astype(np.float64, copy=False)
            if w_parts
            else np.ones((idx_t.shape[0],), dtype=np.float64)
        )
    else:
        pairs = np.empty((0, 2), dtype=np.int64)
        weights = np.empty((0,), dtype=np.float64)

    return DemuxDataset(
        temperature_K=tsel,
        shards=list(chosen),
        X_list=X_list,
        pairs=pairs,
        weights=weights,
        dt_ps=dt_ps,
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
