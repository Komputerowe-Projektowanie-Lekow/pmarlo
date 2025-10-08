from __future__ import annotations

"""
Versioned, strict shard JSON schema for PMARLO datasets.

This module defines minimal dataclasses for shard metadata that are
temperature-aware and explicitly distinguish DEMUX vs REPLICA provenance.

Key rules:
- schema_version "2.0" is the current version.
- kind: "demux" requires temperature_K (float) and forbids replica_index.
- kind: "replica" requires replica_index (int >= 0) and forbids temperature_K.

These models intentionally do not depend on filename patterns for critical
fields. When migrating from legacy shards (v1), use shard_io to perform a
compatibility parse from stored metadata with strict checks.
"""

from dataclasses import dataclass

from pmarlo import constants as const
from typing import Any, Dict, Literal, Optional

SCHEMA_VERSION = const.SHARD_SCHEMA_VERSION


@dataclass(frozen=True)
class BaseShard:
    """Common shard metadata across kinds.

    This is the single source of truth passed downstream. Filenames are
    considered opaque; consumers use these fields only.
    """

    # Schema
    schema_version: str

    # Identity & provenance
    id: str  # normalized ID (e.g., canonical or legacy-fallback)
    kind: Literal["demux", "replica"]
    run_id: str

    # Data description
    n_frames: int
    dt_ps: Optional[float]
    cv_names: tuple[str, ...]
    periodic: tuple[bool, ...]

    # Source data pointers
    topology_path: Optional[str]
    traj_path: Optional[str]
    exchange_log_path: Optional[str]

    # Optional bias/analysis meta
    bias_info: Dict[str, Any] | None
    created_at: str

    # Extra for legacy carry-over
    legacy: Dict[str, Any] | None


@dataclass(frozen=True)
class DemuxShard(BaseShard):
    temperature_K: float
    replica_index: None = None


@dataclass(frozen=True)
class ReplicaShard(BaseShard):
    replica_index: int
    temperature_K: None = None


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


def validate_fields(
    kind: str, temperature_K: Optional[float], replica_index: Optional[int]
) -> None:
    """Validate mutual exclusion/requirement between temperature and replica index."""
    if kind == "demux":
        _require(temperature_K is not None, "demux shard requires temperature_K")
        _require(replica_index is None, "demux shard forbids replica_index")
    elif kind == "replica":
        _require(
            replica_index is not None and int(replica_index) >= 0,
            "replica shard requires non-negative replica_index",
        )
        _require(temperature_K is None, "replica shard forbids temperature_K")
    else:
        raise ValueError(f"invalid shard kind: {kind}")
