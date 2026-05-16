from __future__ import annotations

"""Utilities for maintaining shard indexes backed by strict metadata."""

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, cast

from pmarlo import constants as const
from pmarlo.shards.discover import discover_shard_jsons
from pmarlo.shards.id import canonical_shard_id
from pmarlo.shards.meta import load_shard_meta
from pmarlo.utils.path_utils import ensure_directory

INDEX_VERSION = const.SHARD_INDEX_VERSION

__all__ = [
    "rescan_shards",
    "prune_missing_shards",
    "ShardRegistry",
]


def _collect_json_paths(paths: Iterable[Path]) -> List[Path]:
    collected: List[Path] = []
    for entry in paths:
        path = Path(entry)
        if path.is_dir():
            collected.extend(discover_shard_jsons(path))
        elif path.suffix.lower() == ".json":
            collected.append(path)
    return collected


def rescan_shards(
    roots: Sequence[Path],
    out_index_json: Path,
) -> Path:
    json_paths = sorted({p.resolve() for p in _collect_json_paths(roots)})
    entries = []
    for jp in json_paths:
        meta = load_shard_meta(jp)
        entries.append(
            {
                "path": str(jp),
                "canonical_id": canonical_shard_id(meta),
                "temperature_K": meta.temperature_K,
                "replica_id": meta.replica_id,
                "segment_id": meta.segment_id,
            }
        )

    index_data = {
        "version": INDEX_VERSION,
        "roots": sorted(str(Path(r).resolve()) for r in roots),
        "entries": entries,
        "count": len(entries),
    }

    out_path = Path(out_index_json)
    ensure_directory(out_path.parent)
    out_path.write_text(json.dumps(index_data, indent=2, sort_keys=True))
    return out_path


def prune_missing_shards(index_json_path: Path) -> Path:
    path = Path(index_json_path)
    try:
        data = json.loads(path.read_text())
    except Exception:
        return path

    entries = data.get("entries")
    if not isinstance(entries, list):
        return path

    kept = []
    for entry in entries:
        json_path = Path(entry.get("path", ""))
        if json_path.exists():
            kept.append(entry)
    data["entries"] = kept
    data["count"] = len(kept)
    path.write_text(json.dumps(data, indent=2, sort_keys=True))
    return path


class ShardRegistry:
    """JSON-backed registry for shard indexes."""

    def __init__(self, index_path: Path) -> None:
        self.index_path = Path(index_path)

    def load(self) -> Dict[str, Any]:
        try:
            data = json.loads(self.index_path.read_text())
            if not isinstance(data, dict):
                raise ValueError
            return data
        except Exception:
            return {"version": INDEX_VERSION, "roots": [], "entries": [], "count": 0}

    def save(self, data: Dict[str, object]) -> None:
        ensure_directory(self.index_path.parent)
        self.index_path.write_text(json.dumps(data, indent=2, sort_keys=True))

    def rescan(self, roots: Sequence[Path]) -> Path:
        return rescan_shards(roots, self.index_path)

    def prune(self) -> Path:
        return prune_missing_shards(self.index_path)

    def add(self, shard_json: Path) -> None:
        data = self.load()
        meta = load_shard_meta(shard_json)
        entry = {
            "path": str(Path(shard_json).resolve()),
            "canonical_id": canonical_shard_id(meta),
            "temperature_K": meta.temperature_K,
            "replica_id": meta.replica_id,
            "segment_id": meta.segment_id,
        }
        entries_raw = cast(List[Dict[str, Any]], data.get("entries", []))
        entries: List[Dict[str, Any]] = [
            e
            for e in entries_raw
            if isinstance(e, dict) and e.get("path") != entry["path"]
        ]
        entries.append(entry)
        entries.sort(key=lambda e: e["canonical_id"])
        data["entries"] = entries
        data["count"] = len(entries)
        self.save(data)

    def remove(self, shard_json: Path) -> None:
        data = self.load()
        path_str = str(Path(shard_json).resolve())
        entries_raw = cast(List[Dict[str, Any]], data.get("entries", []))
        entries: List[Dict[str, Any]] = [
            e for e in entries_raw if isinstance(e, dict) and e.get("path") != path_str
        ]
        data["entries"] = entries
        data["count"] = len(entries)
        self.save(data)

    def validate_paths(self) -> Dict[str, List[str]]:
        data = self.load()
        entries_raw = cast(List[Dict[str, Any]], data.get("entries", []))
        entries: List[Dict[str, Any]] = (
            entries_raw if isinstance(entries_raw, list) else []
        )
        missing: List[str] = []
        kept: List[str] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            json_path = Path(entry.get("path", ""))
            if json_path.exists():
                kept.append(str(json_path))
            else:
                missing.append(str(json_path))
        if missing:
            valid_entries: List[Dict[str, Any]] = [
                e for e in entries if isinstance(e, dict) and e.get("path") in kept
            ]
            data["entries"] = valid_entries
            data["count"] = len(valid_entries)
            self.save(data)
        return {"missing": missing, "kept": kept}
