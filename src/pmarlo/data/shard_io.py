from __future__ import annotations

"""Shard discovery and strict, versioned parsing helpers."""

from collections.abc import Iterable as IterableABC
from pathlib import Path
from typing import Any, Dict, List, Mapping

from pmarlo.shards.format import read_shard_npz_json
from pmarlo.utils.validation import require

from .shard_schema import (
    SCHEMA_VERSION,
    BaseShard,
    DemuxShard,
    ReplicaShard,
)


def _coerce_tuple_str(x: object) -> tuple[str, ...]:
    if x is None:
        return tuple()
    if isinstance(x, Mapping):
        iterable: IterableABC[Any] = x.keys()
    elif isinstance(x, IterableABC):
        iterable = x
    else:
        raise TypeError("value is not iterable")
    return tuple(str(s) for s in iterable)


def _coerce_tuple_bool(x: object) -> tuple[bool, ...]:
    if x is None:
        return tuple()
    if isinstance(x, Mapping):
        iterable: IterableABC[Any] = x.values()
    elif isinstance(x, IterableABC):
        iterable = x
    else:
        raise TypeError("value is not iterable")
    return tuple(bool(b) for b in iterable)


def load_shard_meta(json_path: Path) -> BaseShard:
    """Load shard metadata from canonical JSON."""

    json_path = Path(json_path)
    shard = read_shard_npz_json(json_path.with_suffix(".npz"), json_path)
    meta = shard.meta
    provenance: Dict[str, Any] = dict(meta.provenance)

    schema_version = str(meta.schema_version)
    require(
        schema_version == SCHEMA_VERSION,
        f"Shard schema_version {schema_version} does not match {SCHEMA_VERSION}",
    )

    kind = provenance.get("kind")
    run_id = provenance.get("run_id")
    require(isinstance(kind, str), f"Shard {json_path} missing provenance.kind")
    require(isinstance(run_id, str), f"Shard {json_path} missing provenance.run_id")
    kind_str = str(kind).strip().lower()
    run_id_str = str(run_id)

    periodic_raw = provenance.get("periodic")
    require(
        isinstance(periodic_raw, (list, tuple)),
        f"Shard {json_path} must declare periodic flags",
    )
    cv_names = _coerce_tuple_str(meta.feature_spec.columns)
    periodic = _coerce_tuple_bool(periodic_raw)
    require(
        len(periodic) == len(cv_names),
        f"Shard {json_path} periodic flags length mismatch",
    )

    created_at = provenance.get("created_at")
    require(
        isinstance(created_at, str) and created_at,
        f"Shard {json_path} missing created_at",
    )

    topology_path = provenance.get("topology") or provenance.get("topology_path")
    topology_path = str(topology_path) if topology_path is not None else None

    traj_path = provenance.get("traj") or provenance.get("path")
    traj_path = str(traj_path) if traj_path is not None else None

    exchange_log = provenance.get("exchange_log") or provenance.get("exchange_log_path")
    exchange_log = str(exchange_log) if exchange_log is not None else None

    bias_payload = provenance.get("bias_info")
    bias_info: Dict[str, Any] | None
    if isinstance(bias_payload, dict):
        bias_info = dict(bias_payload)
    else:
        bias_info = None

    dt_ps_value = (
        float(meta.dt_ps) if getattr(meta, "dt_ps", None) is not None else None
    )

    base_kwargs: Dict[str, Any] = dict(
        schema_version=schema_version,
        id=str(meta.shard_id),
        kind=kind_str,
        run_id=run_id_str,
        json_path=str(json_path),
        n_frames=int(meta.n_frames),
        dt_ps=dt_ps_value,
        cv_names=cv_names,
        periodic=periodic,
        topology_path=topology_path,
        traj_path=traj_path,
        exchange_log_path=exchange_log,
        bias_info=bias_info,
        created_at=str(created_at),
        raw=provenance,
    )

    if kind_str == "demux":
        return DemuxShard(temperature_K=float(meta.temperature_K), **base_kwargs)

    replica_index = provenance.get("replica_index", meta.replica_id)
    require(
        isinstance(replica_index, (int, float)),
        f"Shard {json_path} missing replica_index",
    )
    return ReplicaShard(replica_index=int(replica_index), **base_kwargs)


def discover_shards(root: Path | str) -> List[BaseShard]:
    """Recursively discover shard JSON files under root and parse them strictly."""

    root = Path(root)
    shards: List[BaseShard] = []
    for p in root.rglob("*.json"):
        if not p.with_suffix(".npz").exists():
            continue
        shards.append(load_shard_meta(p))
    return shards
