from __future__ import annotations

"""NPZ/JSON serialization helpers for PMARLO shards."""

import hashlib
import json
from pathlib import Path
from typing import Tuple

import numpy as np

from pmarlo.utils.json_io import load_json_file
from pmarlo.utils.path_utils import ensure_directory

from .schema import FeatureSpec, Shard, ShardMeta, validate_invariants

__all__ = [
    "write_shard_npz_json",
    "read_shard_npz_json",
    "write_shard",
    "read_shard",
    "hash_shard_arrays",
]


def write_shard_npz_json(
    shard: Shard, npz_path: Path, json_path: Path
) -> Tuple[Path, Path]:
    """Persist shard arrays/metadata to disk in the canonical layout."""

    npz_path = Path(npz_path)
    json_path = Path(json_path)
    ensure_directory(npz_path.parent)
    ensure_directory(json_path.parent)

    validate_invariants(shard)

    np.savez_compressed(
        npz_path,
        X=_coerce_array(shard.X, np.float32, ndim=2),
        t_index=_coerce_array(shard.t_index, np.int64, ndim=1),
        dt_ps=np.array(shard.dt_ps, dtype=np.float32),
        energy=_optional_array(shard.energy, np.float32),
        bias=_optional_array(shard.bias, np.float32),
        w_frame=_optional_array(shard.w_frame, np.float32),
    )

    meta = shard.meta
    payload = {
        "schema_version": meta.schema_version,
        "shard_id": meta.shard_id,
        "temperature_K": meta.temperature_K,
        "beta": meta.beta,
        "replica_id": int(meta.replica_id),
        "segment_id": int(meta.segment_id),
        "exchange_window_id": int(meta.exchange_window_id),
        "n_frames": int(meta.n_frames),
        "dt_ps": float(meta.dt_ps),
        "feature_spec": {
            "name": meta.feature_spec.name,
            "scaler": meta.feature_spec.scaler,
            "columns": list(meta.feature_spec.columns),
        },
        "provenance": meta.provenance,
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=False))
    return npz_path, json_path


def read_shard_npz_json(npz_path: Path, json_path: Path) -> Shard:
    """Load shard arrays/metadata from canonical NPZ+JSON files."""

    json_payload = load_json_file(json_path)
    feature_spec = FeatureSpec(
        name=str(json_payload["feature_spec"]["name"]),
        scaler=str(json_payload["feature_spec"]["scaler"]),
        columns=tuple(json_payload["feature_spec"]["columns"]),
    )
    meta = ShardMeta(
        schema_version=str(json_payload["schema_version"]),
        shard_id=str(json_payload["shard_id"]),
        temperature_K=float(json_payload["temperature_K"]),
        beta=float(json_payload["beta"]),
        replica_id=int(json_payload["replica_id"]),
        segment_id=int(json_payload["segment_id"]),
        exchange_window_id=int(json_payload["exchange_window_id"]),
        n_frames=int(json_payload["n_frames"]),
        dt_ps=float(json_payload["dt_ps"]),
        feature_spec=feature_spec,
        provenance=dict(json_payload["provenance"]),
    )

    npz = np.load(Path(npz_path))
    shard = Shard(
        meta=meta,
        X=_coerce_array(npz["X"], np.float32, ndim=2),
        t_index=_coerce_array(npz["t_index"], np.int64, ndim=1),
        dt_ps=float(np.array(npz["dt_ps"]).item()),
        energy=_optional_loaded(npz, "energy"),
        bias=_optional_loaded(npz, "bias"),
        w_frame=_optional_loaded(npz, "w_frame"),
    )
    validate_invariants(shard)
    return shard


def write_shard(shard: Shard, out_dir: Path) -> Path:
    """Write shard to ``out_dir`` using ``meta.shard_id`` as stem."""

    out_dir = Path(out_dir)
    stem = shard.meta.shard_id
    npz_path = out_dir / f"{stem}.npz"
    json_path = out_dir / f"{stem}.json"
    _, json_written = write_shard_npz_json(shard, npz_path, json_path)
    return json_written


def read_shard(json_path: Path) -> Shard:
    """Read shard using JSON path and sibling NPZ."""

    json_path = Path(json_path)
    npz_path = json_path.with_suffix(".npz")
    return read_shard_npz_json(npz_path, json_path)


def hash_shard_arrays(*arrays: np.ndarray) -> str:
    """Compute deterministic SHA-256 hash over array dtypes, shapes, and data."""

    hasher = hashlib.sha256()
    for arr in arrays:
        data = np.asarray(arr)
        hasher.update(str(data.dtype).encode("utf-8"))
        hasher.update(np.asarray(data.shape, dtype=np.int64).tobytes())
        hasher.update(np.ascontiguousarray(data).tobytes())
    return hasher.hexdigest()


def _coerce_array(arr, dtype, *, ndim: int) -> np.ndarray:
    out = np.asarray(arr, dtype=dtype)
    if out.ndim != ndim:
        raise ValueError(f"Expected array with {ndim} dimensions, got {out.ndim}")
    return out


def _optional_array(arr, dtype) -> np.ndarray:
    if arr is None:
        return np.array([], dtype=dtype)
    return np.asarray(arr, dtype=dtype).reshape(-1)


def _optional_loaded(npz, key: str):
    if key not in npz.files:
        return None
    data = np.asarray(npz[key])
    if data.size == 0:
        return None
    return data.reshape(-1)
