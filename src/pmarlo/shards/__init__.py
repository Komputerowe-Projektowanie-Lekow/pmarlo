from __future__ import annotations

"""Public interface for PMARLO shard utilities."""

from .schema import FeatureSpec, Shard, ShardMeta, validate_invariants
from .format import read_shard, read_shard_npz_json, write_shard, write_shard_npz_json
from .meta import load_shard_meta
from .id import canonical_shard_id
from .discover import discover_shard_jsons, list_temperatures, iter_metas
from .pair_builder import PairBuilder
from .assemble import group_by_temperature, load_shards, select_shards
from .emit import emit_shards_from_trajectories, ExtractShard

__all__ = [
    "FeatureSpec",
    "Shard",
    "ShardMeta",
    "validate_invariants",
    "read_shard",
    "write_shard",
    "read_shard_npz_json",
    "write_shard_npz_json",
    "load_shard_meta",
    "canonical_shard_id",
    "discover_shard_jsons",
    "list_temperatures",
    "iter_metas",
    "PairBuilder",
    "group_by_temperature",
    "load_shards",
    "select_shards",
    "emit_shards_from_trajectories",
    "ExtractShard",
]
