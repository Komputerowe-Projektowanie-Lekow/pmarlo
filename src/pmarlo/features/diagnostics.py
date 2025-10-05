from __future__ import annotations

"""Diagnostics for Deep‑TICA pair availability (dry‑run, no training).

Provides a quick way to inspect whether a chosen lag will produce
time‑lagged pairs across the current dataset shards.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from pmarlo.features.deeptica.core import build_pair_info


@dataclass(frozen=True)
class PairDiagItem:
    """Per‑shard diagnostic for pair counts and metadata.

    Attributes
    ----------
    id
        Shard identifier string.
    frames
        Number of frames in the shard block.
    pairs
        Number of scaled‑time pairs at the requested lag (uniform if no bias).
    pairs_uniform
        Number of uniform‑time pairs at the requested lag (baseline reference).
    has_bias
        True if bias/temperature metadata were available for this shard.
    temperature
        Temperature in Kelvin if available; otherwise None.
    frames_leq_lag
        True if frames <= lag (cannot form any uniform pairs).
    """

    id: str
    frames: int
    pairs: int
    pairs_uniform: int
    has_bias: bool
    temperature: Optional[float]
    frames_leq_lag: bool


@dataclass(frozen=True)
class PairDiagReport:
    """Aggregate diagnostic report for Deep‑TICA pairs.

    Now includes additional context helpful for debugging training skips.
    """

    lag_used: int
    n_shards: int
    frames_total: int
    pairs_total: int
    per_shard: List[PairDiagItem]
    message: str
    # Extra context
    scaled_time_used: bool
    shards_with_bias: int
    too_short_count: int
    pairs_total_uniform: int
    duplicates: List[str]


def _collect_shards_info(dataset: Any) -> List[Dict[str, Any]]:
    if isinstance(dataset, dict) and "__shards__" in dataset:
        return list(dataset["__shards__"])  # type: ignore[return-value]
    return []


def diagnose_deeptica_pairs(dataset: Any, lag: int) -> PairDiagReport:
    """Compute per-shard pair counts for a given lag without training.

    The input dataset should match the structure produced by
    ``pmarlo.data.aggregate.load_shards_as_dataset`` or the dataset
    forwarded to the engine builder, i.e., contain ``"X"`` and optionally
    ``"__shards__"``.
    """

    shards = _collect_shards_info(dataset)
    features = _extract_feature_matrix(dataset)

    from pmarlo.features.pairs import scaled_time_pairs  # local import

    if shards:
        summary = _diagnose_with_shards(shards, lag, scaled_time_pairs)
    else:
        summary = _diagnose_without_shards(features, lag, scaled_time_pairs)

    return PairDiagReport(**summary)


def _extract_feature_matrix(dataset: Any) -> Optional[np.ndarray]:
    if not isinstance(dataset, dict) or "X" not in dataset:
        return None
    try:
        matrix = np.asarray(dataset.get("X"), dtype=np.float64)
    except Exception:
        return None
    return matrix


def _diagnose_with_shards(
    shards: List[Dict[str, Any]],
    lag: int,
    pair_fn: Any,
) -> Dict[str, Any]:
    lag = max(1, int(lag))
    dummy_blocks = [
        np.empty((max(0, int(shard.get("frames", 0))), 0), dtype=np.float32)
        for shard in shards
    ]
    pair_info = build_pair_info(dummy_blocks, [lag]) if dummy_blocks else None

    if pair_info is not None:
        diagnostics = pair_info.diagnostics
        uniform_counts = [int(count) for count in diagnostics.get("pairs_by_shard", [])]
        if len(uniform_counts) < len(shards):
            uniform_counts.extend([0] * (len(shards) - len(uniform_counts)))
        pairs_total_uniform = int(diagnostics.get("usable_pairs", 0))
        short_shards = {int(idx) for idx in diagnostics.get("short_shards", [])}
    else:
        uniform_counts = [0] * len(shards)
        pairs_total_uniform = 0
        short_shards = set()

    per: List[PairDiagItem] = []
    frames_total = 0
    pairs_total = 0
    shards_with_bias = 0
    too_short_count = 0

    for idx, shard in enumerate(shards):
        uniform = uniform_counts[idx] if idx < len(uniform_counts) else 0
        item, stats = _evaluate_shard(
            shard,
            lag,
            pair_fn,
            uniform_pairs=uniform,
            is_short=idx in short_shards,
        )
        per.append(item)
        frames_total += stats.frames
        pairs_total += stats.pairs
        shards_with_bias += stats.bias_flag
        too_short_count += stats.short_flag

    duplicates = _find_duplicates(per)
    message = _format_shard_message(
        shards,
        lag,
        pairs_total,
        pairs_total_uniform,
        shards_with_bias,
        too_short_count,
        duplicates,
    )
    return {
        "lag_used": lag,
        "n_shards": len(shards),
        "frames_total": frames_total,
        "pairs_total": pairs_total,
        "per_shard": per,
        "message": message,
        "scaled_time_used": shards_with_bias > 0,
        "shards_with_bias": shards_with_bias,
        "too_short_count": too_short_count,
        "pairs_total_uniform": pairs_total_uniform,
        "duplicates": duplicates,
    }


@dataclass
class _ShardStats:
    frames: int
    pairs: int
    pairs_uniform: int
    bias_flag: int
    short_flag: int


def _evaluate_shard(
    shard: Dict[str, Any],
    lag: int,
    pair_fn: Any,
    *,
    uniform_pairs: int,
    is_short: bool,
) -> Tuple[PairDiagItem, _ShardStats]:
    shard_id = str(shard.get("id", "unknown"))
    frames = max(0, int(shard.get("frames", 0)))
    bias = shard.get("bias_potential")
    temp_k = shard.get("temperature")
    has_bias = bias is not None and temp_k is not None

    logw = _resolve_log_weights(bias, temp_k) if has_bias else None
    if logw is not None:
        scaled_indices = pair_fn(frames, logw, float(lag), jitter=0.0)
        pairs_scaled = int(len(scaled_indices[0])) if scaled_indices else 0
    elif has_bias:
        pairs_scaled = uniform_pairs
    else:
        pairs_scaled = uniform_pairs

    frames_leq_lag = bool(is_short or frames <= lag)

    item = PairDiagItem(
        id=shard_id,
        frames=frames,
        pairs=pairs_scaled,
        pairs_uniform=uniform_pairs,
        has_bias=has_bias,
        temperature=temp_k if has_bias else None,
        frames_leq_lag=frames_leq_lag,
    )
    stats = _ShardStats(
        frames=frames,
        pairs=pairs_scaled,
        pairs_uniform=uniform_pairs,
        bias_flag=1 if has_bias else 0,
        short_flag=1 if frames_leq_lag else 0,
    )
    return item, stats


def _resolve_log_weights(bias: Any, temp_k: Any) -> Optional[np.ndarray]:
    try:
        beta = 1.0 / (0.008314462618 / float(temp_k))
        return beta * np.asarray(bias, dtype=np.float64)
    except Exception:
        return None


def _find_duplicates(items: List[PairDiagItem]) -> List[str]:
    seen: set[str] = set()
    duplicates: List[str] = []
    for item in items:
        if item.id in seen:
            duplicates.append(item.id)
        else:
            seen.add(item.id)
    return duplicates


def _format_shard_message(
    shards: List[Dict[str, Any]],
    lag: int,
    pairs_total: int,
    pairs_uniform: int,
    shards_with_bias: int,
    too_short_count: int,
    duplicates: List[str],
) -> str:
    shard_count = len(shards)
    if shards_with_bias > 0:
        message = (
            f"Scaled-time pairs: {pairs_total} total across {shard_count} shards "
            f"({shards_with_bias} with bias). Uniform baseline: {pairs_uniform} pairs."
        )
    else:
        message = f"Uniform pairs: {pairs_total} total across {shard_count} shards."
    if too_short_count > 0:
        message += f" {too_short_count} shards too short for lag={lag}."
    if duplicates:
        message += f" {len(duplicates)} duplicate shard IDs found."
    return message


def _diagnose_without_shards(
    features: Optional[np.ndarray],
    lag: int,
    pair_fn: Any,
) -> Dict[str, Any]:
    if features is None:
        raise ValueError(
            "No dataset provided or dataset lacks both '__shards__' metadata and 'X' array"
        )
    n_frames = int(features.shape[0])
    lag = max(1, int(lag))
    indices = pair_fn(n_frames, None, float(lag))
    pairs_scaled = int(len(indices[0])) if indices else 0
    item = PairDiagItem(
        id="concatenated",
        frames=n_frames,
        pairs=pairs_scaled,
        pairs_uniform=pairs_scaled,
        has_bias=False,
        temperature=None,
        frames_leq_lag=n_frames <= lag,
    )
    message = (
        f"No shard metadata found. Using concatenated X ({n_frames} frames): "
        f"{pairs_scaled} pairs at lag={lag}."
    )
    if n_frames <= lag:
        message += " Dataset too short for requested lag."
    return {
        "lag_used": lag,
        "n_shards": 1,
        "frames_total": n_frames,
        "pairs_total": pairs_scaled,
        "per_shard": [item],
        "message": message,
        "scaled_time_used": False,
        "shards_with_bias": 0,
        "too_short_count": 1 if n_frames <= lag else 0,
        "pairs_total_uniform": pairs_scaled,
        "duplicates": [],
    }
