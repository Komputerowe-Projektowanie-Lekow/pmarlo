from __future__ import annotations

"""Deterministic shard discovery utilities."""

from pathlib import Path
from typing import Iterator, List

from .meta import load_shard_meta

__all__ = ["discover_shard_jsons", "list_temperatures", "iter_metas"]


def discover_shard_jsons(root: Path) -> List[Path]:
    """Return sorted list of shard JSON paths under ``root``."""

    root = Path(root)
    jsons = sorted(p for p in root.rglob("*.json"))
    return jsons


def iter_metas(root: Path) -> Iterator:
    """Yield ``ShardMeta`` objects for all shards under ``root``."""

    for json_path in discover_shard_jsons(root):
        yield load_shard_meta(json_path)


def list_temperatures(root: Path) -> List[float]:
    """Return sorted unique temperatures discovered under ``root``."""

    temps = {meta.temperature_K for meta in iter_metas(root)}
    return sorted(temps)
