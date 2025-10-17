from __future__ import annotations

"""Versioned, strict shard JSON schema for PMARLO datasets."""

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional

from pmarlo import constants as const
from pmarlo.utils.validation import require

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
    id: str  # normalized ID derived from canonical rules
    kind: Literal["demux", "replica"]
    run_id: str
    json_path: str

    # Data description
    n_frames: int
    dt_ps: float | None
    cv_names: tuple[str, ...]
    periodic: tuple[bool, ...]

    # Source data pointers
    topology_path: Optional[str]
    traj_path: Optional[str]
    exchange_log_path: Optional[str]

    # Optional bias/analysis meta
    bias_info: Dict[str, Any] | None
    created_at: str

    # Raw provenance payload for downstream consumers
    raw: Dict[str, Any] | None


@dataclass(frozen=True)
class DemuxShard(BaseShard):
    temperature_K: float
    replica_index: None = None


@dataclass(frozen=True)
class ReplicaShard(BaseShard):
    replica_index: int
    temperature_K: None = None


def validate_fields(
    kind: str, temperature_K: Optional[float], replica_index: Optional[int]
) -> None:
    """Validate mutual exclusion/requirement between temperature and replica index."""
    if kind == "demux":
        require(temperature_K is not None, "demux shard requires temperature_K")
        require(replica_index is None, "demux shard forbids replica_index")
    elif kind == "replica":
        require(
            replica_index is not None and int(replica_index) >= 0,
            "replica shard requires non-negative replica_index",
        )
        require(temperature_K is None, "replica shard forbids temperature_K")
    else:
        raise ValueError(f"invalid shard kind: {kind}")
