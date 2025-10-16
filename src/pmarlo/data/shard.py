from __future__ import annotations

"""
Shim around :mod:`pmarlo.shards.format` exposing the historical
``pmarlo.data.shard`` API while using the single canonical shard schema.

The legacy module used a bespoke JSON layout with array hashes and multiple
variants. To enforce a single shard format the helpers below translate the old
function signatures (used throughout tests and utilities) to the canonical
:class:`pmarlo.shards.schema.Shard` representation.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np

from pmarlo import constants as const
from pmarlo.shards.format import read_shard_npz_json, write_shard_npz_json
from pmarlo.shards.schema import FeatureSpec, Shard, ShardMeta

__all__ = ["write_shard", "read_shard", "_sha256_bytes"]


@dataclass(frozen=True)
class LegacyShardMeta:
    """Lightweight adapter exposing historical ``pmarlo.data.shard`` fields."""

    schema_version: str
    shard_id: str
    temperature: float
    temperature_K: float
    beta: float
    n_frames: int
    dt_ps: float
    cv_names: tuple[str, ...]
    periodic: tuple[bool, ...]
    created_at: str
    source: Dict[str, object]
    provenance: Dict[str, object]
    kind: str | None = None
    replica_id: int = 0
    segment_id: int = 0
    exchange_window_id: int = 0

    def __getattr__(self, item: str):
        # Backwards compatibility: alias temperature_K -> temperature, etc.
        if item == "temperature_K":
            return self.temperature_K
        if item == "beta":
            return self.beta
        raise AttributeError(item)


def _stack_columns(cvs: Dict[str, np.ndarray]) -> np.ndarray:
    if not cvs:
        raise ValueError("cvs dictionary must contain at least one column")
    arrays = []
    length = None
    for key in cvs:
        arr = np.asarray(cvs[key], dtype=np.float32).reshape(-1)
        if length is None:
            length = arr.shape[0]
        elif arr.shape[0] != length:
            raise ValueError("all CV arrays must have the same length")
        arrays.append(arr)
    return np.column_stack(arrays)


def write_shard(
    out_dir: Path,
    shard_id: str,
    cvs: Dict[str, np.ndarray],
    dtraj: np.ndarray | None,
    periodic: Dict[str, bool],
    seed: int,
    temperature: float,
    source: Dict[str, object] | None = None,
) -> Path:
    """
    Write a shard using the canonical NPZ+JSON schema.

    Parameters mirror the historical helper; ``dtraj`` is accepted for
    compatibility but stored only when provided.
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X = _stack_columns(cvs)
    n_frames = X.shape[0]

    feature_spec = FeatureSpec(
        name=str((source or {}).get("feature_spec_name", "pmarlo_features")),
        scaler=str((source or {}).get("feature_scaler", "identity")),
        columns=tuple(str(k) for k in cvs.keys()),
    )

    meta = ShardMeta(
        schema_version="2.0",
        shard_id=str(shard_id),
        temperature_K=float(temperature),
        beta=float(1.0 / (const.BOLTZMANN_CONSTANT_KJ_PER_MOL * float(temperature))),
        replica_id=int((source or {}).get("replica_id", 0)),
        segment_id=int((source or {}).get("segment_id", 0)),
        exchange_window_id=int((source or {}).get("exchange_window_id", 0)),
        n_frames=int(n_frames),
        dt_ps=float((source or {}).get("dt_ps", 1.0)),
        feature_spec=feature_spec,
        provenance=dict(source or {}),
    )

    shard = Shard(
        meta=meta,
        X=X.astype(np.float32, copy=False),
        t_index=np.arange(n_frames, dtype=np.int64),
        dt_ps=meta.dt_ps,
        energy=None,
        bias=None,
        w_frame=None,
    )
    npz_path, json_path = write_shard_npz_json(
        shard,
        out_dir / f"{meta.shard_id}.npz",
        out_dir / f"{meta.shard_id}.json",
    )

    # If trajectory indices were supplied, persist them as an optional array.
    if dtraj is not None:
        dtraj_arr = np.asarray(dtraj, dtype=np.int32).reshape(-1)
        with np.load(npz_path) as data:
            payload = {name: data[name] for name in data.files}
        payload["dtraj"] = dtraj_arr
        np.savez_compressed(npz_path, **payload)

    return json_path


def read_shard(json_path: Path) -> tuple[ShardMeta, np.ndarray, np.ndarray | None]:
    """
    Read a shard in canonical format and return ``(meta, X, dtraj)``.

    The returned CV matrix uses ``float64`` to preserve legacy behaviour in
    downstream consumers.
    """

    json_path = Path(json_path)
    shard = read_shard_npz_json(json_path.with_suffix(".npz"), json_path)

    dtraj: np.ndarray | None = None
    with np.load(json_path.with_suffix(".npz")) as data:
        if "dtraj" in data.files:
            arr = np.asarray(data["dtraj"], dtype=np.int32)
            if arr.size > 0:
                dtraj = arr

    columns = tuple(str(c) for c in shard.meta.feature_spec.columns)
    periodic = tuple(
        bool(v)
        for v in shard.meta.provenance.get(
            "periodic", [False for _ in range(len(columns))]
        )
    )
    periodic = periodic if len(periodic) == len(columns) else tuple(False for _ in columns)

    created_at = str(shard.meta.provenance.get("created_at", "1970-01-01T00:00:00Z"))
    source = dict(shard.meta.provenance.get("source", {}))

    legacy_meta = LegacyShardMeta(
        schema_version=shard.meta.schema_version,
        shard_id=shard.meta.shard_id,
        temperature=shard.meta.temperature_K,
        temperature_K=shard.meta.temperature_K,
        beta=shard.meta.beta,
        n_frames=shard.meta.n_frames,
        dt_ps=shard.meta.dt_ps,
        cv_names=columns,
        periodic=periodic,
        created_at=created_at,
        source=source,
        provenance=shard.meta.provenance,
        kind=source.get("kind") if isinstance(source.get("kind"), str) else None,
        replica_id=shard.meta.replica_id,
        segment_id=shard.meta.segment_id,
        exchange_window_id=shard.meta.exchange_window_id,
    )

    return legacy_meta, shard.X.astype(np.float64, copy=False), dtraj


def _sha256_bytes(*arrays: np.ndarray) -> str:
    """
    Helper maintained for backwards-compatible tests that verify deterministic
    hashing. Uses the same semantics as the classic implementation.
    """

    from hashlib import sha256

    hasher = sha256()
    for arr in arrays:
        contiguous = np.ascontiguousarray(arr)
        hasher.update(str(contiguous.dtype.str).encode("utf-8"))
        hasher.update(str(contiguous.shape).encode("utf-8"))
        hasher.update(contiguous.tobytes())
    return hasher.hexdigest()
