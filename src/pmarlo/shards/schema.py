from __future__ import annotations

"""Core shard data structures and invariants for PMARLO joint workflow."""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from pmarlo import constants as const

__all__ = [
    "FeatureSpec",
    "ShardMeta",
    "Shard",
    "validate_invariants",
]


@dataclass(frozen=True)
class FeatureSpec:
    """Description of features stored in a shard array."""

    name: str
    scaler: str
    columns: Tuple[str, ...]

    def __post_init__(self) -> None:  # pragma: no cover - dataclass hook
        object.__setattr__(self, "columns", tuple(str(c) for c in self.columns))
        if not self.columns:
            raise ValueError("FeatureSpec.columns must not be empty")


@dataclass(frozen=True)
class ShardMeta:
    """Strict metadata associated with a shard."""

    schema_version: str
    shard_id: str
    temperature_K: float
    beta: float
    replica_id: int
    segment_id: int
    exchange_window_id: int
    n_frames: int
    dt_ps: float
    feature_spec: FeatureSpec
    provenance: Dict[str, Any]

    def __post_init__(self) -> None:  # pragma: no cover - dataclass hook
        _require(bool(self.schema_version), "schema_version is required")
        _require(self.temperature_K > 0.0, "temperature_K must be positive")
        _require(self.beta > 0.0, "beta must be positive")
        _require(self.n_frames > 0, "n_frames must be positive")
        _require(self.dt_ps > 0.0, "dt_ps must be positive")
        _require(
            isinstance(self.provenance, dict),
            "provenance must be a dictionary",
        )


@dataclass
class Shard:
    """In-memory representation of shard data and metadata."""

    meta: ShardMeta
    X: np.ndarray  # [frames, features]
    t_index: np.ndarray  # [frames]
    dt_ps: float
    energy: Optional[np.ndarray] = None
    bias: Optional[np.ndarray] = None
    w_frame: Optional[np.ndarray] = None


def validate_invariants(shard: Shard) -> None:
    """Validate strict invariants for shard data/metadata alignment."""

    meta = shard.meta
    X = np.asarray(shard.X)
    t_index = np.asarray(shard.t_index)

    _require(X.ndim == 2, "X must be a 2-D array")
    _require(t_index.ndim == 1, "t_index must be 1-D")
    _require(X.shape[0] == meta.n_frames, "X rows must equal meta.n_frames")
    _require(
        t_index.size == meta.n_frames,
        "t_index length must equal meta.n_frames",
    )

    _require(
        np.array_equal(t_index, np.arange(meta.n_frames, dtype=t_index.dtype)),
        "t_index must be contiguous starting at 0",
    )

    _require(abs(float(shard.dt_ps) - float(meta.dt_ps)) < const.NUMERIC_MIN_POSITIVE, "dt_ps mismatch")

    n = meta.n_frames
    for name, arr in {
        "energy": shard.energy,
        "bias": shard.bias,
        "w_frame": shard.w_frame,
    }.items():
        if arr is None:
            continue
        arr_np = np.asarray(arr)
        _require(
            arr_np.ndim == 1 and arr_np.size == n,
            f"{name} must be 1-D with length n_frames",
        )

    _require(
        X.shape[1] == len(meta.feature_spec.columns),
        "feature_spec.columns must match number of feature dimensions",
    )

    from .id import canonical_shard_id  # late import to avoid cycle

    expected_id = canonical_shard_id(meta)
    _require(
        meta.shard_id == expected_id,
        f"shard_id '{meta.shard_id}' does not match canonical '{expected_id}'",
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)
