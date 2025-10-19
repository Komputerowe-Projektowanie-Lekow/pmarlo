from __future__ import annotations

"""Canonical shard helpers exposing the historical pmarlo.data.shard API."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, cast

import numbers

import numpy as np

from pmarlo import constants as const
from pmarlo.shards.format import read_shard_npz_json, write_shard_npz_json
from pmarlo.shards.id import canonical_shard_id
from pmarlo.shards.schema import FeatureSpec, Shard, ShardMeta
from pmarlo.utils.path_utils import ensure_directory
from pmarlo.utils.validation import require

__all__ = ["write_shard", "read_shard", "_sha256_bytes"]


@dataclass(frozen=True)
class ShardDetails:
    """Lightweight adapter exposing shard metadata for downstream consumers."""

    meta: ShardMeta
    source: Dict[str, object]

    @property
    def shard_id(self) -> str:
        return str(self.meta.shard_id)

    @property
    def temperature_K(self) -> float:
        return float(self.meta.temperature_K)

    @property
    def cv_names(self) -> tuple[str, ...]:
        columns = cast(
            Sequence[object], getattr(self.meta.feature_spec, "columns", tuple())
        )
        return tuple(str(name) for name in columns)

    @property
    def periodic(self) -> tuple[bool, ...]:
        periodic_raw = self.source.get("periodic")
        if not isinstance(periodic_raw, (list, tuple)):
            raise ValueError("periodic flags missing from shard provenance")
        if len(periodic_raw) != len(self.cv_names):
            raise ValueError("periodic flags missing from shard provenance")
        return tuple(bool(v) for v in periodic_raw)


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
    *,
    bias_potential: np.ndarray | None = None,
) -> Path:
    """Write a shard using the canonical NPZ+JSON schema."""

    out_dir = Path(out_dir)
    ensure_directory(out_dir)

    if source is None:
        raise ValueError("source metadata is required")

    source_dict: Dict[str, object] = dict(source)
    for key in ("created_at", "kind", "run_id", "replica_id", "segment_id"):
        require(key in source_dict, f"source missing required key '{key}'")

    kind_raw = source_dict["kind"]
    if not isinstance(kind_raw, str):
        raise ValueError("kind must be a string")
    kind = kind_raw.strip().lower()
    if kind not in {"demux", "replica"}:
        raise ValueError("kind must be 'demux' or 'replica'")

    run_id_raw = source_dict["run_id"]
    if not isinstance(run_id_raw, str):
        raise ValueError("run_id must be a string")

    replica_raw = source_dict["replica_id"]
    if not isinstance(replica_raw, numbers.Integral):
        raise ValueError("replica_id must be an integer")
    replica_id = int(replica_raw)

    segment_raw = source_dict["segment_id"]
    if not isinstance(segment_raw, numbers.Integral):
        raise ValueError("segment_id must be an integer")
    segment_id = int(segment_raw)

    exchange_raw = source_dict.get("exchange_window_id", 0)
    if isinstance(exchange_raw, numbers.Integral):
        exchange_window_id = int(exchange_raw)
    else:
        raise ValueError("exchange_window_id must be an integer if provided")

    column_order = tuple(cvs.keys())
    periodic_map: Dict[str, bool] = dict(periodic or {})
    ordered_periodic = [bool(periodic_map.get(name, False)) for name in column_order]

    t_kelvin = float(temperature)
    X = _stack_columns(cvs)
    n_frames = X.shape[0]

    source_dict.update(
        {
            "kind": kind,
            "run_id": run_id_raw,
            "replica_id": replica_id,
            "segment_id": segment_id,
            "exchange_window_id": exchange_window_id,
            "seed": int(seed),
            "n_frames": n_frames,
            "periodic": ordered_periodic,
            "temperature_K": t_kelvin,
        }
    )

    dt_ps_value = source_dict.get("dt_ps", 1.0)
    if isinstance(dt_ps_value, numbers.Real):
        dt_ps = float(dt_ps_value)
    elif isinstance(dt_ps_value, str):
        try:
            dt_ps = float(dt_ps_value)
        except ValueError as exc:  # pragma: no cover - defensive
            raise ValueError("dt_ps must be numeric") from exc
    else:
        raise ValueError("dt_ps must be numeric")
    feature_spec = FeatureSpec(
        name=str(source_dict.get("feature_spec_name", "pmarlo_features")),
        scaler=str(source_dict.get("feature_scaler", "identity")),
        columns=tuple(str(k) for k in column_order),
    )

    bias_array: np.ndarray | None = None
    if bias_potential is not None:
        bias_array = np.asarray(bias_potential, dtype=np.float32).reshape(-1)
        if bias_array.shape[0] != n_frames:
            raise ValueError("bias_potential length must match number of frames")
        source_dict["has_bias"] = True
    else:
        source_dict["has_bias"] = False

    provenance = dict(source_dict)
    if "source" not in provenance:
        provenance["source"] = dict(source)

    meta = ShardMeta(
        schema_version=const.SHARD_SCHEMA_VERSION,
        shard_id="placeholder",
        temperature_K=t_kelvin,
        beta=float(1.0 / (const.BOLTZMANN_CONSTANT_KJ_PER_MOL * t_kelvin)),
        replica_id=replica_id,
        segment_id=segment_id,
        exchange_window_id=exchange_window_id,
        n_frames=int(n_frames),
        dt_ps=dt_ps,
        feature_spec=feature_spec,
        provenance=provenance,
    )
    canonical_id = canonical_shard_id(meta)
    if shard_id != canonical_id:
        raise ValueError(
            f"shard_id '{shard_id}' does not match canonical '{canonical_id}'"
        )

    meta = ShardMeta(
        schema_version=meta.schema_version,
        shard_id=canonical_id,
        temperature_K=meta.temperature_K,
        beta=meta.beta,
        replica_id=meta.replica_id,
        segment_id=meta.segment_id,
        exchange_window_id=meta.exchange_window_id,
        n_frames=meta.n_frames,
        dt_ps=meta.dt_ps,
        feature_spec=meta.feature_spec,
        provenance=meta.provenance,
    )

    shard_obj = Shard(
        meta=meta,
        X=X.astype(np.float32, copy=False),
        t_index=np.arange(n_frames, dtype=np.int64),
        dt_ps=meta.dt_ps,
        energy=None,
        bias=bias_array,
        w_frame=None,
    )
    npz_path, json_path = write_shard_npz_json(
        shard_obj,
        out_dir / f"{meta.shard_id}.npz",
        out_dir / f"{meta.shard_id}.json",
    )

    if dtraj is not None:
        dtraj_arr = np.asarray(dtraj, dtype=np.int32).reshape(-1)
        with np.load(npz_path) as data:
            payload = {name: data[name] for name in data.files}
        payload["dtraj"] = dtraj_arr
        np.savez_compressed(npz_path, **payload)

    return Path(json_path)


def read_shard(json_path: Path) -> tuple[ShardDetails, np.ndarray, np.ndarray | None]:
    """Read a shard in canonical format and return (meta, X, dtraj)."""

    json_path = Path(json_path)
    shard = read_shard_npz_json(json_path.with_suffix(".npz"), json_path)

    dtraj: np.ndarray | None = None
    with np.load(json_path.with_suffix(".npz")) as data:
        if "dtraj" in data.files:
            arr = np.asarray(data["dtraj"], dtype=np.int32)
            if arr.size > 0:
                dtraj = arr

    provenance_payload = getattr(shard.meta, "provenance", None)
    source = (
        dict(cast(Mapping[str, object], provenance_payload))
        if isinstance(provenance_payload, Mapping)
        else {}
    )
    return (
        ShardDetails(meta=shard.meta, source=source),
        shard.X.astype(np.float64, copy=False),
        dtraj,
    )


def _sha256_bytes(*arrays: np.ndarray) -> str:
    """Helper retained for deterministic hashing tests."""

    from hashlib import sha256

    hasher = sha256()
    for arr in arrays:
        contiguous = np.ascontiguousarray(arr)
        hasher.update(str(contiguous.dtype.str).encode("utf-8"))
        hasher.update(str(contiguous.shape).encode("utf-8"))
        hasher.update(contiguous.tobytes())
    return hasher.hexdigest()
