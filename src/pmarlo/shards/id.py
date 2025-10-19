from __future__ import annotations

"""Canonical shard identifiers."""

from .schema import ShardMeta

__all__ = ["canonical_shard_id"]


def canonical_shard_id(meta: ShardMeta) -> str:
    """Return canonical identifier enforcing DEMUX uniqueness."""

    replica = int(meta.replica_id)
    segment = int(meta.segment_id)
    t_kelvin = int(round(meta.temperature_K))

    # Include kind in ID to prevent collisions between demux and replica shards
    kind = str(meta.provenance.get("kind", "demux")).lower()
    if kind not in {"demux", "replica"}:
        raise ValueError(f"Invalid shard kind: {kind}")

    if kind == "replica":
        return f"replica_T{t_kelvin}K_seg{segment:04d}_rep{replica:03d}"
    return f"T{t_kelvin}K_seg{segment:04d}_rep{replica:03d}"
