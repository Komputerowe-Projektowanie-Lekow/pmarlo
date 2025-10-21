from __future__ import annotations

"""Canonical shard helpers exposing the historical pmarlo.data.shard API."""

import numbers
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, cast

import numpy as np

from pmarlo import constants as const
from pmarlo.shards.format import read_shard_npz_json, write_shard_npz_json
from pmarlo.shards.id import canonical_shard_id
from pmarlo.shards.schema import FeatureSpec, Shard, ShardMeta
from pmarlo.utils.path_utils import ensure_directory
from pmarlo.utils.validation import require

__all__ = ["write_shard", "read_shard", "_sha256_bytes"]


def _normalise_float_dtype(dtype: np.dtype[Any] | type | str) -> np.dtype[Any]:
    """Return a floating point ``numpy.dtype`` instance for ``dtype``."""

    resolved = np.dtype(dtype)
    if resolved.kind != "f":
        raise TypeError("dtype must be a floating-point dtype")
    return resolved


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


@dataclass(frozen=True)
class _SourceCore:
    """Normalised required provenance fields used when writing shards."""

    kind: str
    run_id: str
    replica_id: int
    segment_id: int
    exchange_window_id: int


def _ensure_source_dict(source: Mapping[str, object] | None) -> Dict[str, object]:
    if source is None:
        raise ValueError("source metadata is required")
    return dict(source)


def _coerce_int(value: object, *, name: str) -> int:
    if not isinstance(value, numbers.Integral):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _coerce_str(value: object, *, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    return value


def _normalise_source_metadata(source_dict: Dict[str, object]) -> _SourceCore:
    for key in ("created_at", "kind", "run_id", "replica_id", "segment_id"):
        require(key in source_dict, f"source missing required key '{key}'")

    kind_raw = _coerce_str(source_dict["kind"], name="kind").strip().lower()
    if kind_raw not in {"demux", "replica"}:
        raise ValueError("kind must be 'demux' or 'replica'")

    run_id = _coerce_str(source_dict["run_id"], name="run_id")
    replica_id = _coerce_int(source_dict["replica_id"], name="replica_id")
    segment_id = _coerce_int(source_dict["segment_id"], name="segment_id")
    exchange_window_id = _coerce_int(
        source_dict.get("exchange_window_id", 0), name="exchange_window_id"
    )

    source_dict["kind"] = kind_raw
    source_dict["run_id"] = run_id
    source_dict["replica_id"] = replica_id
    source_dict["segment_id"] = segment_id
    source_dict["exchange_window_id"] = exchange_window_id

    return _SourceCore(
        kind=kind_raw,
        run_id=run_id,
        replica_id=replica_id,
        segment_id=segment_id,
        exchange_window_id=exchange_window_id,
    )


def _resolve_periodic_flags(
    column_order: Sequence[str], periodic: Mapping[str, bool]
) -> list[bool]:
    periodic_map: Dict[str, bool] = dict(periodic or {})
    return [bool(periodic_map.get(name, False)) for name in column_order]


def _coerce_dt_ps(value: object) -> float:
    if isinstance(value, numbers.Real):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError as exc:  # pragma: no cover - defensive
            raise ValueError("dt_ps must be numeric") from exc
    raise ValueError("dt_ps must be numeric")


def _prepare_bias_array(
    bias_potential: np.ndarray | None,
    n_frames: int,
    *,
    target: Dict[str, object],
    dtype: np.dtype[Any],
) -> np.ndarray | None:
    if bias_potential is None:
        target["has_bias"] = False
        return None

    bias_array = np.asarray(bias_potential, dtype=dtype).reshape(-1)
    if bias_array.shape[0] != n_frames:
        raise ValueError("bias_potential length must match number of frames")
    target["has_bias"] = True
    return bias_array


def _stack_columns(cvs: Dict[str, np.ndarray], *, dtype: np.dtype[Any]) -> np.ndarray:
    if not cvs:
        raise ValueError("cvs dictionary must contain at least one column")
    arrays = []
    length = None
    for key in cvs:
        arr = np.asarray(cvs[key], dtype=dtype).reshape(-1)
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
    compute_arrays_hash: bool = False,
    dtype: np.dtype[Any] | type | str = np.float32,
) -> Path:
    """Write a shard using the canonical NPZ+JSON schema."""

    out_dir = Path(out_dir)
    ensure_directory(out_dir)

    dtype = _normalise_float_dtype(dtype)

    source_payload = dict(source or {})
    source_dict = _ensure_source_dict(source)
    core = _normalise_source_metadata(source_dict)

    column_order = tuple(cvs.keys())
    ordered_periodic = _resolve_periodic_flags(column_order, periodic or {})

    t_kelvin = float(temperature)
    X = _stack_columns(cvs, dtype=dtype)
    n_frames = X.shape[0]

    source_dict.update(
        {
            "seed": int(seed),
            "temperature_K": t_kelvin,
            "n_frames": n_frames,
            "columns": column_order,
            "periodic": ordered_periodic,
        }
    )

    dt_ps_value = source_dict.get("dt_ps", 1.0)
    dt_ps = _coerce_dt_ps(dt_ps_value)
    feature_spec = FeatureSpec(
        name=str(source_dict.get("feature_spec_name", "pmarlo_features")),
        scaler=str(source_dict.get("feature_scaler", "identity")),
        columns=tuple(str(k) for k in column_order),
    )

    bias_array = _prepare_bias_array(
        bias_potential, n_frames, target=source_dict, dtype=dtype
    )

    dtraj_arr: np.ndarray | None = None
    if dtraj is not None:
        dtraj_arr = np.asarray(dtraj, dtype=np.int32).reshape(-1)
        if dtraj_arr.shape[0] != n_frames:
            raise ValueError("dtraj length must match number of frames")

    if compute_arrays_hash:
        arrays_for_hash = [np.asarray(X, dtype=np.float32, order="C")]
        if dtraj_arr is not None:
            arrays_for_hash.append(dtraj_arr)
        source_dict["arrays_hash"] = _sha256_bytes(*arrays_for_hash)

    provenance = dict(source_dict)
    if "source" not in provenance:
        provenance["source"] = dict(source_payload)

    meta = ShardMeta(
        schema_version=const.SHARD_SCHEMA_VERSION,
        shard_id="placeholder",
        temperature_K=t_kelvin,
        beta=float(1.0 / (const.BOLTZMANN_CONSTANT_KJ_PER_MOL * t_kelvin)),
        replica_id=core.replica_id,
        segment_id=core.segment_id,
        exchange_window_id=core.exchange_window_id,
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
        X=X.astype(dtype, copy=False),
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

    if dtraj_arr is not None:
        with np.load(npz_path) as data:
            payload = {name: data[name] for name in data.files}
        payload["dtraj"] = dtraj_arr
        np.savez_compressed(npz_path, **payload)

    return Path(json_path)


def read_shard(
    json_path: Path,
    *,
    dtype: np.dtype[Any] | type | str = np.float32,
    validate_arrays_hash: bool = False,
) -> tuple[ShardDetails, np.ndarray, np.ndarray | None]:
    """Read a shard in canonical format and return ``(meta, X, dtraj)``."""

    dtype = _normalise_float_dtype(dtype)

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

    if validate_arrays_hash:
        expected_hash = source.get("arrays_hash")
        if not isinstance(expected_hash, str):
            raise KeyError("arrays_hash missing from shard provenance")

        arrays_for_hash = [np.asarray(shard.X, dtype=np.float32, order="C")]
        if dtraj is not None:
            arrays_for_hash.append(np.asarray(dtraj, dtype=np.int32))

        actual_hash = _sha256_bytes(*arrays_for_hash)
        if actual_hash != expected_hash:
            raise ValueError("Shard arrays hash mismatch")

    return (
        ShardDetails(meta=shard.meta, source=source),
        shard.X.astype(dtype, copy=False),
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
