from __future__ import annotations

"""Diagnostics for Deep‑TICA pair availability (dry‑run, no training).

Provides a quick way to inspect whether a chosen lag will produce
time‑lagged pairs across the current dataset shards.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np


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
    """Compute per‑shard pair counts for a given lag without training.

    The input ``dataset`` should match the structure produced by
    :func:`pmarlo.data.aggregate.load_shards_as_dataset` or the dataset
    forwarded to the engine builder, i.e., contain ``"X"`` and optionally
    ``"__shards__"``.
    """
    shards = _collect_shards_info(dataset)
    X_all = None
    try:
        X_all = (
            np.asarray(dataset.get("X"), dtype=np.float64)
            if isinstance(dataset, dict)
            else None
        )
    except Exception:
        X_all = None

    per: List[PairDiagItem] = []
    frames_total = 0
    pairs_total = 0
    pairs_total_uniform = 0
    shards_with_bias = 0

    from pmarlo.features.pairs import (  # local import; light dependency
        scaled_time_pairs,
    )

    if shards:
        # Use shard metadata for precise diagnostics
        duplicates: List[str] = []
        too_short_count = 0
        for s in shards:
            s_id = str(s.get("id", "unknown"))
            frames = int(s.get("frames", 0))
            frames_total += frames
            bias = s.get("bias_potential")
            temp_k = s.get("temperature")
            has_bias = bias is not None and temp_k is not None
            if has_bias:
                shards_with_bias += 1
                # Compute scaled-time pairs
                logw = None
                if bias is not None and temp_k is not None:
                    # Convert bias to log-weights
                    try:
                        beta = 1.0 / (
                            0.008314462618 / float(temp_k)
                        )  # kJ/mol/K to 1/kJ/mol
                        logw = beta * np.asarray(bias, dtype=np.float64)
                    except Exception:
                        logw = None
                i_scaled, j_scaled = scaled_time_pairs(
                    frames, logw, float(lag), jitter=0.0
                )
                pairs_scaled = int(len(i_scaled))
                pairs_total += pairs_scaled
            else:
                # Uniform pairs (no bias)
                pairs_scaled = max(0, frames - lag) if frames > lag else 0
                pairs_total += pairs_scaled

            # Always compute uniform baseline
            i_uniform, j_uniform = scaled_time_pairs(frames, None, float(lag))
            pairs_uniform = int(len(i_uniform))
            pairs_total_uniform += pairs_uniform

            frames_leq_lag = frames <= lag
            if frames_leq_lag:
                too_short_count += 1

            per.append(
                PairDiagItem(
                    id=s_id,
                    frames=frames,
                    pairs=pairs_scaled,
                    pairs_uniform=pairs_uniform,
                    has_bias=has_bias,
                    temperature=temp_k,
                    frames_leq_lag=frames_leq_lag,
                )
            )

        # Deduplicate shard IDs for summary
        seen = set()
        duplicates = []
        for item in per:
            if item.id in seen:
                duplicates.append(item.id)
            else:
                seen.add(item.id)

        # Construct diagnostic message
        scaled_time_used = shards_with_bias > 0
        if scaled_time_used:
            message = (
                f"Scaled-time pairs: {pairs_total} total across {len(shards)} shards "
                f"({shards_with_bias} with bias). "
                f"Uniform baseline: {pairs_total_uniform} pairs."
            )
        else:
            message = f"Uniform pairs: {pairs_total} total across {len(shards)} shards."

        if too_short_count > 0:
            message += f" {too_short_count} shards too short for lag={lag}."
        if duplicates:
            message += f" {len(duplicates)} duplicate shard IDs found."

    else:
        # Fallback: use concatenated X if no shard metadata
        if X_all is not None:
            n_frames = int(X_all.shape[0])
            frames_total = n_frames
            i_scaled, j_scaled = scaled_time_pairs(n_frames, None, float(lag))
            pairs_scaled = int(len(i_scaled))
            pairs_total = pairs_scaled
            pairs_total_uniform = pairs_scaled  # same as scaled when no bias
            scaled_time_used = False
            too_short_count = 1 if n_frames <= lag else 0
            duplicates = []

            per.append(
                PairDiagItem(
                    id="concatenated",
                    frames=n_frames,
                    pairs=pairs_scaled,
                    pairs_uniform=pairs_scaled,
                    has_bias=False,
                    temperature=None,
                    frames_leq_lag=n_frames <= lag,
                )
            )

            message = (
                f"No shard metadata found. Using concatenated X ({n_frames} frames): "
                f"{pairs_scaled} pairs at lag={lag}."
            )
            if too_short_count > 0:
                message += " Dataset too short for requested lag."
            shards_with_bias = 0
        else:
            raise ValueError(
                "No dataset provided or dataset lacks both '__shards__' metadata and 'X' array"
            )

    return PairDiagReport(
        lag_used=lag,
        n_shards=len(shards) if shards else 1,
        frames_total=frames_total,
        pairs_total=pairs_total,
        per_shard=per,
        message=message,
        scaled_time_used=scaled_time_used,
        shards_with_bias=shards_with_bias,
        too_short_count=too_short_count,
        pairs_total_uniform=pairs_total_uniform,
        duplicates=duplicates,
    )
