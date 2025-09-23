from __future__ import annotations

"""
Deterministic shard format for feature/CV slices.

This module defines a minimal schema split into:
- NPZ file with arrays: X (float64, n×k), dtraj (int32 1-D or empty),
  and optional bias_potential (float64 1-D or empty)
- JSON file with :class:`ShardMeta` describing provenance and an arrays hash

Invariants:
- CV arrays are 1-D, equal length; order of CV names is preserved
- JSON bytes are canonical for identical inputs (sorted keys, canonical separators)
- Integrity is verified by SHA-256 over dtype, shape, and raw bytes

Example:
    json_path = write_shard(
        Path("shards"),
        "shard_000",
        {"phi": phi, "psi": psi},
        None,
        {"phi": True, "psi": False},
        42,
        300.0,
        {"created_at": "1970-01-01T00:00:00Z"},
    )
    meta, X, dtraj = read_shard(json_path)
"""

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np

from pmarlo.io.shard_id import ShardId


@dataclass(frozen=True)
class ShardMeta:
    """Metadata describing a deterministic feature shard.

    JSON serialization is expected to be canonical via :func:`_canonical_json`.
    """

    shard_id: str
    seed: int
    temperature: float
    n_frames: int
    cv_names: Tuple[str, ...]
    periodic: Tuple[bool, ...]
    created_at: str
    source: Dict[str, Any]
    arrays_hash: str
    schema_version: str = "2.0"


def _canonical_json(obj: Dict[str, Any]) -> str:
    """Produce deterministic JSON string for dictionaries.

    Keys are sorted, separators are canonical, and NaNs are not allowed.
    """

    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def generate_canonical_shard_id_from_meta(meta: ShardMeta, json_path: Path) -> str:
    """
    Generate canonical shard ID from shard metadata.

    This function attempts to construct a canonical shard identifier from the
    metadata stored in a shard JSON file. It uses the source path information
    to determine the canonical form.

    Parameters
    ----------
    meta : ShardMeta
        Shard metadata object
    json_path : Path
        Path to the shard JSON file

    Returns
    -------
    str
        Canonical shard identifier, or legacy format if parsing fails

    Examples
    --------
    >>> meta = ShardMeta(...)
    >>> canonical_id = generate_canonical_shard_id_from_meta(meta, Path("shard.json"))
    >>> print(canonical_id)
    'run-20250906-170155:demux:T300:0'
    """
    try:
        # Extract source path from metadata
        source = meta.source or {}
        source_path_str = (
            source.get("traj")
            or source.get("path")
            or source.get("file")
            or source.get("source_path")
        )

        if source_path_str:
            source_path = Path(source_path_str)
            # Try to parse canonical ID from source path
            try:
                shard_id = ShardId(
                    run_id=source.get("run_id") or source.get("run_uid") or "",
                    source_kind=(
                        "demux" if "demux" in str(source_path).lower() else "replica"
                    ),
                    temperature_K=(
                        int(meta.temperature) if meta.temperature != 300.0 else None
                    ),
                    replica_index=None,  # Will be determined by parsing
                    local_index=0,  # Placeholder, will be computed
                    source_path=source_path,
                    dataset_hash="",  # Not available from meta
                )

                # Use the parsing function to get proper local_index
                from pmarlo.io.shard_id import parse_shard_id

                parsed_id = parse_shard_id(source_path)
                canonical_id: str = parsed_id.canonical()
                return canonical_id

            except Exception:
                pass

        # Fallback to legacy format
        run_uid = (
            source.get("run_uid") or source.get("run_id") or source.get("run_dir") or ""
        )
        return f"{run_uid}|{meta.shard_id}|{str(json_path.resolve())}"

    except Exception:
        # Ultimate fallback
        return f"fallback|{meta.shard_id}|{str(json_path.resolve())}"


def _sha256_bytes(*arrays: np.ndarray) -> str:
    """Compute SHA-256 over dtype, shape and raw bytes of given arrays.

    Arrays are converted to C-contiguous layout for stable hashing.
    """

    h = sha256()
    for arr in arrays:
        a = np.ascontiguousarray(arr)
        h.update(str(a.dtype.str).encode("utf-8"))
        h.update(str(a.shape).encode("utf-8"))
        h.update(a.tobytes())
    return h.hexdigest()


def _ensure_float64_column_stack(columns: Iterable[np.ndarray]) -> np.ndarray:
    xs = [np.asarray(c, dtype=np.float64).reshape(-1) for c in columns]
    if not xs:
        raise ValueError("No CV arrays provided")
    n = xs[0].shape[0]
    for x in xs:
        if x.shape[0] != n:
            raise ValueError("All CV arrays must have the same length")
    return np.column_stack(xs).astype(np.float64, copy=False)


def write_shard(
    out_dir: Path,
    shard_id: str,
    cvs: Dict[str, np.ndarray],
    dtraj: np.ndarray | None,
    periodic: Dict[str, bool],
    seed: int,
    temperature: float,
    source: Dict[str, Any],
    *,
    bias_potential: np.ndarray | None = None,
) -> Path:
    """Write a deterministic shard as NPZ (arrays) + JSON (metadata).

    Returns the path to the JSON file.
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Preserve insertion order of CV names to allow mismatch detection later
    cv_names = tuple(cvs.keys())
    X = _ensure_float64_column_stack([cvs[name] for name in cv_names])

    # dtraj is optional; when absent, store an empty int32 array in NPZ
    if dtraj is None:
        dtraj_arr = np.array([], dtype=np.int32)
    else:
        dtraj_arr = np.asarray(dtraj, dtype=np.int32).reshape(-1)

    # bias_potential is optional per-frame array (float64)
    if bias_potential is None:
        bias_arr = np.array([], dtype=np.float64)
    else:
        bias_arr = np.asarray(bias_potential, dtype=np.float64).reshape(-1)

    npz_path = out_dir / f"{shard_id}.npz"
    # Use compressed NPZ; integrity is based on array bytes, not file bytes
    np.savez_compressed(npz_path, X=X, dtraj=dtraj_arr, bias_potential=bias_arr)

    arrays_hash = (
        _sha256_bytes(X) if dtraj_arr.size == 0 else _sha256_bytes(X, dtraj_arr)
    )
    n_frames = int(X.shape[0])
    periodic_tuple = tuple(bool(periodic.get(name, False)) for name in cv_names)

    # created_at must be deterministic for identical inputs unless provided.
    created_at = str(source.get("created_at", "1970-01-01T00:00:00Z"))

    # Enrich source with explicit temperature_K and has_bias flags for provenance
    enriched_source = dict(source)
    try:
        # Try to parse canonical provenance from a provided trajectory path
        traj_path_str = (
            enriched_source.get("traj")
            or enriched_source.get("path")
            or enriched_source.get("file")
            or enriched_source.get("source_path")
        )
        if isinstance(traj_path_str, str) and traj_path_str:
            try:
                from pmarlo.io.shard_id import parse_shard_id as _parse_sid

                sid = _parse_sid(Path(traj_path_str), require_exists=False)
                # DEMUX-first: store explicit kind/run_id and conditional role fields
                enriched_source.update(
                    {
                        "kind": sid.source_kind,
                        "run_id": sid.run_id,
                    }
                )
                if sid.source_kind == "demux" and sid.temperature_K is not None:
                    enriched_source["temperature_K"] = float(sid.temperature_K)
                if sid.source_kind == "replica" and sid.replica_index is not None:
                    enriched_source["replica_index"] = int(sid.replica_index)
            except Exception:
                # Best-effort only; leave as-is on failure
                pass

        enriched_source.update(
            {
                "temperature_K": float(temperature),
                "has_bias": bool(bias_arr.size > 0),
            }
        )
    except Exception:
        # Guard against non-mapping inputs; fallback to a minimal dict
        enriched_source = {
            "temperature_K": float(temperature),
            "has_bias": bool(bias_arr.size > 0),
        }

    meta = ShardMeta(
        shard_id=str(shard_id),
        seed=int(seed),
        temperature=float(temperature),
        n_frames=n_frames,
        cv_names=cv_names,
        periodic=periodic_tuple,
        created_at=created_at,
        source=enriched_source,
        arrays_hash=arrays_hash,
    )

    # Serialize metadata canonically
    json_obj: Dict[str, Any] = {
        "shard_id": meta.shard_id,
        "seed": meta.seed,
        "temperature": meta.temperature,
        "n_frames": meta.n_frames,
        "cv_names": list(meta.cv_names),
        "periodic": list(meta.periodic),
        "created_at": meta.created_at,
        "source": meta.source,
        "arrays_hash": meta.arrays_hash,
        "schema_version": meta.schema_version,
    }
    json_str = _canonical_json(json_obj)
    json_path = out_dir / f"{shard_id}.json"
    json_path.write_text(json_str)
    return json_path


def read_shard(json_path: Path) -> tuple[ShardMeta, np.ndarray, np.ndarray | None]:
    """Read shard JSON + NPZ and verify integrity against the stored hash.

    Raises ValueError on hash mismatch.
    """

    json_path = Path(json_path)
    data = json.loads(json_path.read_text())

    shard_id = str(data["shard_id"])  # prefer the JSON field for sanity
    npz_path = json_path.with_name(f"{shard_id}.npz")
    with np.load(npz_path) as f:
        X = np.asarray(f["X"], dtype=np.float64)
        dtraj_arr = np.asarray(f["dtraj"], dtype=np.int32)
        # Optional field for forward-compatibility with extended schema
        bias_arr = (
            np.asarray(f["bias_potential"], dtype=np.float64)
            if "bias_potential" in getattr(f, "files", [])
            else np.array([], dtype=np.float64)
        )

    arrays_hash = (
        _sha256_bytes(X) if dtraj_arr.size == 0 else _sha256_bytes(X, dtraj_arr)
    )
    if arrays_hash != data.get("arrays_hash"):
        raise ValueError(
            "Shard arrays hash mismatch: expected "
            f"{data.get('arrays_hash')} but computed {arrays_hash}"
        )

    # Preserve existing source, but ensure explicit temperature_K / has_bias keys
    source_in = dict(data.get("source", {}))
    if "temperature_K" not in source_in:
        source_in["temperature_K"] = float(data.get("temperature", np.nan))
    source_in["has_bias"] = bool(bias_arr.size > 0)

    meta = ShardMeta(
        shard_id=shard_id,
        seed=int(data["seed"]),
        temperature=float(data["temperature"]),
        n_frames=int(data["n_frames"]),
        cv_names=tuple(str(x) for x in data["cv_names"]),
        periodic=tuple(bool(x) for x in data["periodic"]),
        created_at=str(data["created_at"]),
        source=source_in,
        arrays_hash=str(data["arrays_hash"]),
        schema_version=str(data.get("schema_version", "pmarlo.shard.v1")),
    )

    dtraj_out: np.ndarray | None = None if dtraj_arr.size == 0 else dtraj_arr
    return meta, X, dtraj_out
