from __future__ import annotations

"""
Robust shard discovery and indexing utilities.

This module provides tolerant parsing of shard JSON filenames and helpers to
rescan/prune a lightweight index after workspace cleanups.
"""

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from pmarlo.io.shard_id import ShardId


@dataclass(frozen=True)
class ShardIndexEntry:
    path: str
    uid: str
    run_id: str
    kind: str  # "demux" | "replica"
    temperature_K: Optional[int]


def _nearest_run_id(p: Path) -> str:
    cur = p.resolve().parent
    for _ in range(6):
        if cur.name.startswith("run-"):
            return cur.name
        if cur.parent == cur:
            break
        cur = cur.parent
    return ""


def parse_shard_json_filename(json_path: Path) -> Dict[str, Any]:
    """Tolerant parser for shard JSON path.

    Tries metadata first; falls back to filename and parent directories.
    Returns mapping with keys: run_id, kind, temperature_K, uid.
    """
    p = Path(json_path)
    run_id = _nearest_run_id(p)
    kind = "replica"
    temperature_K: Optional[int] = None
    uid = f"fallback|{p.stem}|{str(p.resolve())}"

    # Attempt to read metadata to refine fields
    try:
        from pmarlo.data.shard import read_shard  # lazy import

        meta, _, _ = read_shard(p)
        run_id = run_id or str(getattr(meta, "source", {}).get("run_uid", ""))
        src = dict(getattr(meta, "source", {}))
        src_path_str = (
            src.get("traj")
            or src.get("path")
            or src.get("file")
            or src.get("source_path")
            or ""
        )
        if src_path_str:
            s = str(src_path_str)
            if "demux" in s.lower():
                kind = "demux"
            elif "replica" in s.lower():
                kind = "replica"
        # If path didn’t hint, use temperature heuristic
        if kind == "replica":
            try:
                t = float(getattr(meta, "temperature", float("nan")))
                if t == int(t):
                    # Treat integer-like temperature as demux by default
                    kind = "demux"
                    temperature_K = int(t)
            except Exception:
                pass
        else:
            try:
                t = float(getattr(meta, "temperature", float("nan")))
                if t == int(t):
                    temperature_K = int(t)
            except Exception:
                pass
        # Prefer canonical uid when possible
        try:
            from pmarlo.io.shard_id import parse_shard_id

            # Use source path when present; otherwise pass the JSON path
            src_for_parse = Path(src_path_str) if src_path_str else p
            sid = parse_shard_id(src_for_parse, require_exists=False)
            uid = sid.canonical()
        except Exception:
            pass
    except Exception:
        # Fallback to filename heuristics if JSON cannot be read
        name = p.name
        m = re.search(r"demux_T(\d+)K", name)
        if m:
            kind = "demux"
            temperature_K = int(m.group(1))
        elif "demux" in name.lower():
            kind = "demux"
        # run_id may still be empty; keep fallback uid
        pass

    return {
        "run_id": run_id,
        "kind": kind,
        "temperature_K": temperature_K,
        "uid": uid,
    }


def build_shard_id_from_json_fallback(
    json_path: Path, dataset_hash: str = ""
) -> ShardId:
    """Construct a ShardId from shard JSON using tolerant parsing.

    This is a best-effort fallback for cases where canonical trajectory path is
    unavailable (e.g., after cleanup).
    """
    info = parse_shard_json_filename(json_path)
    run_id = info.get("run_id") or _nearest_run_id(json_path) or "run-unknown"
    kind = info.get("kind") or "replica"
    tK = info.get("temperature_K") if kind == "demux" else None
    ridx = None if kind == "demux" else 0
    # local_index by sorted order within run dir if available
    local_index = 0
    run_dir_name = run_id
    try:
        cur = Path(json_path).resolve().parent
        run_dir: Optional[Path] = None
        for _ in range(6):
            if cur.name == run_dir_name:
                run_dir = cur
                break
            if cur.parent == cur:
                break
            cur = cur.parent
        if run_dir is not None:
            siblings = sorted(run_dir.rglob("*.json"))
            local_index = max(0, siblings.index(Path(json_path).resolve()))
    except Exception:
        local_index = 0
    return ShardId(
        run_id=str(run_id),
        source_kind="demux" if kind == "demux" else "replica",
        temperature_K=(int(tK) if (tK is not None) else None),
        replica_index=(int(ridx) if (ridx is not None) else None),
        local_index=int(local_index),
        source_path=Path(json_path).resolve(),
        dataset_hash=str(dataset_hash),
    )


def rescan_shards(root_dirs: Sequence[Path], out_index_json: Path) -> Path:
    """Rescan root directories for shard JSONs and write an index JSON.

    Scans only deterministic shard manifests named like ``shard_*.json``. This
    avoids pulling unrelated JSON files into the index and prevents stale or
    legacy entries from persisting after cleanup.

    The index format:
    {"version": 1, "roots": [...], "entries": [{...}], "count": N}
    """
    entries: List[ShardIndexEntry] = []
    for root in root_dirs:
        root = Path(root)
        if not root.exists():
            continue
        # Accept simple shard filenames (preferred)
        for jp in sorted(root.rglob("shard_*.json")):
            try:
                info = parse_shard_json_filename(jp)
                entries.append(
                    ShardIndexEntry(
                        path=str(jp.resolve()),
                        uid=str(info.get("uid", "")),
                        run_id=str(info.get("run_id", "")),
                        kind=str(info.get("kind", "replica")),
                        temperature_K=(
                            int(info["temperature_K"])
                            if info.get("temperature_K") is not None
                            else None
                        ),
                    )
                )
            except Exception:
                # Skip problematic entries silently
                continue
    payload = {
        "version": 1,
        "roots": [str(Path(r)) for r in root_dirs],
        "entries": [asdict(e) for e in entries],
        "count": int(len(entries)),
    }
    out_index_json = Path(out_index_json)
    out_index_json.parent.mkdir(parents=True, exist_ok=True)
    out_index_json.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":"))
    )
    return out_index_json


def prune_missing_shards(index_json_path: Path) -> Path:
    """Remove missing shard paths from an existing index JSON in place."""
    p = Path(index_json_path)
    try:
        data = json.loads(p.read_text())
    except Exception:
        return p
    entries = data.get("entries", [])
    if not isinstance(entries, list):
        return p
    kept: List[Dict[str, Any]] = []
    for e in entries:
        try:
            path = Path(e.get("path", ""))
            if path.exists():
                kept.append(e)
        except Exception:
            continue
    data["entries"] = kept
    data["count"] = int(len(kept))
    p.write_text(json.dumps(data, sort_keys=True, separators=(",", ":")))
    return p


class ShardRegistry:
    """Lightweight registry persisted as JSON for shard integrity and cleanup.

    The registry stores entries with canonical-like IDs and basic provenance and
    provides helpers to keep the index consistent after disk cleanup.
    """

    def __init__(self, index_path: Path) -> None:
        self.index_path = Path(index_path)

    def load(self) -> dict:
        try:
            return json.loads(self.index_path.read_text())
        except Exception:
            return {"version": 1, "roots": [], "entries": [], "count": 0}

    def save(self, data: dict) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.index_path.write_text(
            json.dumps(data, sort_keys=True, separators=(",", ":"))
        )

    def rescan(self, roots: Sequence[Path]) -> Path:
        return rescan_shards(roots, self.index_path)

    def prune(self) -> Path:
        return prune_missing_shards(self.index_path)

    def add(self, shard_json: Path) -> None:
        data = self.load()
        try:
            info = parse_shard_json_filename(Path(shard_json))
            entry = ShardIndexEntry(
                path=str(Path(shard_json).resolve()),
                uid=str(info.get("uid", "")),
                run_id=str(info.get("run_id", "")),
                kind=str(info.get("kind", "replica")),
                temperature_K=(
                    int(info["temperature_K"])
                    if info.get("temperature_K") is not None
                    else None
                ),
            )
            entries = list(data.get("entries", []))
            # Deduplicate by path
            entries = [e for e in entries if e.get("path") != entry.path]
            entries.append(asdict(entry))
            data["entries"] = entries
            data["count"] = int(len(entries))
            self.save(data)
        except Exception:
            # ignore invalid/unsupported entries
            return

    def remove(self, shard_json: Path) -> None:
        data = self.load()
        entries = [
            e
            for e in list(data.get("entries", []))
            if e.get("path") != str(Path(shard_json).resolve())
        ]
        data["entries"] = entries
        data["count"] = int(len(entries))
        self.save(data)

    def validate_paths(self) -> dict:
        data = self.load()
        entries = list(data.get("entries", []))
        missing = []
        kept = []
        for e in entries:
            try:
                p = Path(e.get("path", ""))
                if p.exists():
                    kept.append(e)
                else:
                    missing.append(e.get("path", ""))
            except Exception:
                continue
        if missing:
            data["entries"] = kept
            data["count"] = int(len(kept))
            self.save(data)
        return {"missing": missing, "kept": [e.get("path", "") for e in kept]}


def migrate_legacy_shards(root: Path) -> dict:
    """One-time tolerant migration helper: map legacy shard JSONs to canonical IDs.

    Does not modify shard JSON files; returns a mapping of json_path -> canonical_id
    for diagnostic or index population purposes.
    """
    root = Path(root)
    mapping: Dict[str, str] = {}
    for jp in sorted(root.rglob("shard_*.json")):
        try:
            info = parse_shard_json_filename(jp)
            mapping[str(jp.resolve())] = str(info.get("uid", ""))
        except Exception:
            continue
    return mapping
