from __future__ import annotations

"""Public interface for PMARLO shard utilities."""

from importlib import import_module
from typing import Any, Dict, Tuple

from .assemble import group_by_temperature, load_shards, select_shards
from .discover import discover_shard_jsons, iter_metas, list_temperatures
from .format import read_shard, read_shard_npz_json, write_shard, write_shard_npz_json
from .id import canonical_shard_id
from .meta import load_shard_meta
from .pair_builder import PairBuilder
from .schema import FeatureSpec, Shard, ShardMeta, validate_invariants

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

_OPTIONAL_EXPORTS: Dict[str, Tuple[str, str]] = {
    "emit_shards_from_trajectories": (
        "pmarlo.shards.emit",
        "emit_shards_from_trajectories",
    ),
    "ExtractShard": ("pmarlo.shards.emit", "ExtractShard"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _OPTIONAL_EXPORTS[name]
    except KeyError:  # pragma: no cover - defensive guard
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
