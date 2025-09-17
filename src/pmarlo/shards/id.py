from __future__ import annotations

"""Canonical shard identifiers."""

from .schema import ShardMeta

__all__ = ["canonical_shard_id"]


def canonical_shard_id(meta: ShardMeta) -> str:
    """Return canonical identifier enforcing DEMUX uniqueness."""

    replica = int(meta.replica_id)
    segment = int(meta.segment_id)
    t_kelvin = int(round(meta.temperature_K))
    return f"T{t_kelvin}K_seg{segment:04d}_rep{replica:03d}"
