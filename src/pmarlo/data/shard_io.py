from __future__ import annotations

"""
Shard discovery and strict, versioned parsing helpers.

This module centralizes shard JSON handling to eliminate filename-based
ambiguities. All downstream code should use the parsed objects here and treat
filenames as opaque.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

from pmarlo.io.shard_id import parse_shard_id

from .shard_schema import (
    SCHEMA_VERSION,
    BaseShard,
    DemuxShard,
    ReplicaShard,
    validate_fields,
)


def _infer_kind_from_source_path(p: Path) -> Tuple[str, dict]:
    """Infer shard kind/run_id/temp/replica from a trajectory source path using a strict parser.

    This does not rely on conventions beyond the canonical patterns handled by
    parse_shard_id and returns a dict with fields to merge.
    """
    sid = parse_shard_id(p, require_exists=False)
    kind = sid.source_kind
    run_id = sid.run_id
    base: dict = {"kind": kind, "run_id": run_id}
    if kind == "demux" and sid.temperature_K is not None:
        base["temperature_K"] = float(sid.temperature_K)
    if kind == "replica" and sid.replica_index is not None:
        base["replica_index"] = int(sid.replica_index)
    return kind, base


def _coerce_tuple_str(x) -> tuple[str, ...]:
    return tuple(str(s) for s in (x or ()))


def _coerce_tuple_bool(x) -> tuple[bool, ...]:
    return tuple(bool(b) for b in (x or ()))


def load_shard_meta(json_path: Path) -> BaseShard:
    """Load shard metadata from JSON and validate against schema.

    Supports legacy v1 shards by performing a strict compatibility parse using
    metadata provided in the JSON; never infers temperature/kind from ambiguous
    names alone.
    """
    json_path = Path(json_path)
    raw = json.loads(json_path.read_text())

    # Legacy v1: enrich using strict inference from declared source path only
    schema_version = str(raw.get("schema_version", "pmarlo.shard.v1"))

    # Prefer meta.source to obtain paths/kind; only accept explicit keys
    source_meta: Dict = (
        dict(raw.get("source", {})) if isinstance(raw.get("source", {}), dict) else {}
    )
    traj_path_str = (
        source_meta.get("traj")
        or source_meta.get("path")
        or source_meta.get("file")
        or source_meta.get("source_path")
    )
    traj_path = (
        Path(traj_path_str)
        if isinstance(traj_path_str, str) and traj_path_str
        else None
    )

    # Determine kind/run/role
    kind = source_meta.get("kind") if isinstance(source_meta.get("kind"), str) else None
    run_id = (
        source_meta.get("run_id")
        if isinstance(source_meta.get("run_id"), str)
        else None
    )
    temperature_K = (
        source_meta.get("temperature_K")
        if source_meta.get("temperature_K") is not None
        else None
    )
    replica_index = (
        source_meta.get("replica_index")
        if source_meta.get("replica_index") is not None
        else None
    )

    # If not explicitly provided, strictly parse from the trajectory path
    if (kind is None or run_id is None) and traj_path is not None:
        parsed_kind, fields = _infer_kind_from_source_path(traj_path)
        kind = kind or fields.get("kind")
        run_id = run_id or fields.get("run_id")
        temperature_K = (
            temperature_K if temperature_K is not None else fields.get("temperature_K")
        )
        replica_index = (
            replica_index if replica_index is not None else fields.get("replica_index")
        )

    # Validate mutual requirements
    if kind is None or run_id is None:
        raise ValueError(
            f"Shard missing required provenance (kind/run_id) and could not be resolved from metadata: {json_path}"
        )

    # DEMUX-first: if the shard is declared as replica, enforce that temperature_K is not set here
    # Note: legacy shards may have a top-level 'temperature' used for binning; we do not mix with kind rules here.
    validate_fields(kind, temperature_K, replica_index)

    # Build normalized ID (filename-agnostic) – use canonical if possible
    try:
        if traj_path is not None:
            sid = parse_shard_id(traj_path, require_exists=False)
            canonical_id = sid.canonical()
        else:
            raise RuntimeError
    except Exception:
        # Fallback legacy identifier
        canonical_id = f"{run_id}|{raw.get('shard_id') or json_path.stem}|{traj_path_str or str(json_path)}"

    # Compose common fields
    cv_names = _coerce_tuple_str(raw.get("cv_names", ()))
    periodic = _coerce_tuple_bool(raw.get("periodic", ()))
    created_at = str(raw.get("created_at", "1970-01-01T00:00:00Z"))
    n_frames = int(raw.get("n_frames", 0))
    dt_ps = None
    if "dt_ps" in raw:
        try:
            dt_ps = float(raw.get("dt_ps"))
        except Exception:
            dt_ps = None

    common_kwargs = dict(
        schema_version=(
            SCHEMA_VERSION
            if schema_version.startswith("pmarlo.shard.v")
            else str(schema_version)
        ),
        id=str(canonical_id),
        kind=str(kind),
        run_id=str(run_id),
        n_frames=n_frames,
        dt_ps=dt_ps,
        cv_names=cv_names,
        periodic=periodic,
        topology_path=source_meta.get("topology") or source_meta.get("topology_path"),
        traj_path=(str(traj_path) if traj_path is not None else None),
        exchange_log_path=source_meta.get("exchange_log")
        or source_meta.get("exchange_log_path"),
        bias_info=(
            source_meta.get("bias_info")
            if isinstance(source_meta.get("bias_info"), dict)
            else None
        ),
        created_at=created_at,
        legacy=raw,
    )

    if kind == "demux":
        return DemuxShard(temperature_K=float(temperature_K), **common_kwargs)  # type: ignore[arg-type]
    else:
        return ReplicaShard(replica_index=int(replica_index), **common_kwargs)  # type: ignore[arg-type]


def discover_shards(root: Path | str) -> List[BaseShard]:
    """Recursively discover shard JSON files under root and parse them strictly.

    Ignores non-JSON files; tolerates unreadable/invalid JSON by skipping.
    """
    root = Path(root)
    shards: List[BaseShard] = []
    for p in root.rglob("*.json"):
        # heuristic: shard JSONs typically have sibling NPZ; prefer those
        if not (p.with_suffix(".npz").exists() or p.name.startswith("shard_")):
            # skip unrelated JSON files
            continue
        try:
            shards.append(load_shard_meta(p))
        except Exception:
            # strictly ignore unsupported or invalid files
            continue
    return shards
