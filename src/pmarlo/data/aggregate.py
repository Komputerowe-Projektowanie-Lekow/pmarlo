from __future__ import annotations

"""
Aggregate many shard files and build a global analysis envelope.

This module loads compatible shards (same cv_names and periodicity),
concatenates their CV matrices, assembles a dataset dict, and calls
``pmarlo.transform.build.build_result`` to produce MSM/FES/TRAM results.

Outputs a single JSON bundle via BuildResult.to_json() with a dataset hash
recorded into RunMetadata (when available) for end-to-end reproducibility.
"""

from dataclasses import dataclass, replace
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, List, Mapping, Optional, Sequence, cast

import numpy as np

from pmarlo.data.shard import ShardDetails, read_shard
from pmarlo.data.shard_schema import DemuxShard
from pmarlo.transform.plan import TransformPlan
from pmarlo.utils.errors import TemperatureConsistencyError
from pmarlo.utils.path_utils import ensure_directory

from .shard_io import load_shard_meta

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pmarlo.transform.build import AppliedOpts, BuildOpts, BuildResult


@lru_cache(maxsize=1)
def _transform_build_handles():
    from pmarlo.transform.build import AppliedOpts as _AppliedOpts
    from pmarlo.transform.build import BuildOpts as _BuildOpts
    from pmarlo.transform.build import BuildResult as _BuildResult
    from pmarlo.transform.build import build_result as _build_result

    return _AppliedOpts, _BuildOpts, _BuildResult, _build_result


_PROGRESS_ALIAS_KEYS = (
    "progress_callback",
    "callback",
    "on_event",
    "progress",
    "reporter",
)


def coerce_progress_callback(
    kwargs: dict[str, Any],
) -> Optional[Callable[[str, Mapping[str, Any]], None]]:
    cb: Optional[Callable[[str, Mapping[str, Any]], None]] = None
    for key in _PROGRESS_ALIAS_KEYS:
        value = kwargs.get(key)
        if value is not None and callable(value):
            cb = cast(Callable[[str, Mapping[str, Any]], None], value)
            break
    if cb is not None:
        kwargs.setdefault("progress_callback", cb)
    return cb


def _unique_shard_uid(details: ShardDetails) -> str:
    """Return the canonical shard identifier."""

    return str(details.shard_id)


@dataclass(slots=True)
class AggregatedShards:
    dataset: dict[str, Any]
    dtrajs: List[np.ndarray | None]
    shards_info: List[dict]
    cv_names: tuple[str, ...]
    X_all: np.ndarray


def _aggregate_shard_contents(shard_jsons: Sequence[Path]) -> AggregatedShards:
    """Load shards, enforce safety rails, and build the dataset payload."""

    import logging
    logger = logging.getLogger("pmarlo")

    paths = _normalise_shard_paths(shard_jsons)

    cv_names_ref: tuple[str, ...] | None = None
    periodic_ref: tuple[bool, ...] | None = None
    X_parts: List[np.ndarray] = []
    dtrajs: List[np.ndarray | None] = []
    shards_info: List[dict] = []
    kinds: list[str] = []
    temps: list[float] = []

    for idx, path in enumerate(paths):
        meta_info = load_shard_meta(path)
        kinds.append(meta_info.kind)
        if isinstance(meta_info, DemuxShard):
            temps.append(float(meta_info.temperature_K))

        details, X, dtraj = read_shard(path)

        cv_names_ref, periodic_ref = _validate_or_set_refs(
            details,
            cv_names_ref,
            periodic_ref,
        )

        X_np = np.asarray(X, dtype=np.float64)

        # Log shape immediately after loading the shard data
        logger.info(f"Loaded shard {idx} ({path.name}): raw shape={X_np.shape}")

        X_parts.append(X_np)
        dtrajs.append(None if dtraj is None else np.asarray(dtraj, dtype=np.int32))

        shard_info = _build_shard_info(details, path, X_np, dtraj)
        shards_info.append(shard_info)

    _validate_shard_safety(kinds, temps)

    cv_names = tuple(cv_names_ref or tuple())
    periodic = tuple(periodic_ref or tuple())
    X_all = np.vstack(X_parts).astype(np.float64, copy=False)

    # Log featurized/concatenated shape
    logger.info(f"Featurized all shards: concatenated shape={X_all.shape}")

    _fill_shard_offsets(shards_info)

    dataset = {
        "X": X_all,
        "cv_names": cv_names,
        "periodic": periodic,
        "dtrajs": [d for d in dtrajs if d is not None],
        "__shards__": shards_info,
    }
    return AggregatedShards(dataset, dtrajs, shards_info, cv_names, X_all)


def _normalise_shard_paths(shard_jsons: Sequence[Path]) -> list[Path]:
    if not shard_jsons:
        raise ValueError("No shard JSONs provided")
    return [Path(p) for p in shard_jsons]


def _compute_stride_metadata(
    meta: Any,
    frames_loaded: int,
) -> tuple[int, int | None, float | None, bool]:
    frames_declared = int(meta.n_frames)
    stride_ratio = None
    if frames_loaded > 0 and frames_declared > 0:
        stride_ratio = float(frames_declared) / float(frames_loaded)
    effective_stride = None
    if stride_ratio is not None and stride_ratio > 0:
        effective_stride = max(1, int(round(stride_ratio)))
    preview_truncated = bool(frames_loaded > 0 and frames_declared > frames_loaded)
    return frames_declared, effective_stride, stride_ratio, preview_truncated


def _extract_time_metadata(
    source_dict: Mapping[str, Any],
) -> tuple[dict[str, Any], Any, Any]:
    time_fields: dict[str, Any] = {}
    first_ts = None
    last_ts = None
    for key, value in source_dict.items():
        if not isinstance(key, str):
            continue
        key_lower = key.lower()
        if "time" not in key_lower:
            continue
        time_fields[key] = value
        if ("first" in key_lower or "start" in key_lower) and first_ts is None:
            first_ts = value
        if "last" in key_lower or "stop" in key_lower or "end" in key_lower:
            last_ts = value
    return time_fields, first_ts, last_ts


def _augment_with_source_metadata(
    info: dict[str, Any],
    source_dict: Mapping[str, Any],
    path: Path,
    *,
    effective_stride: int | None,
    stride_ratio: float | None,
) -> None:
    source_path = Path(
        source_dict.get("traj")
        or source_dict.get("path")
        or source_dict.get("file")
        or path
    ).resolve()
    info["source_path"] = str(source_path)
    info["run_uid"] = (
        source_dict.get("run_uid")
        or source_dict.get("run_id")
        or source_dict.get("run_dir")
    )
    if effective_stride and effective_stride > 1:
        info.setdefault("notes", {})["stride_ratio"] = (
            stride_ratio if stride_ratio is not None else effective_stride
        )
    time_fields, first_ts, last_ts = _extract_time_metadata(source_dict)
    if time_fields:
        info["time_metadata"] = time_fields
    if first_ts is not None:
        info["first_timestamp"] = first_ts
    if last_ts is not None:
        info["last_timestamp"] = last_ts


def _build_shard_info(
    details: ShardDetails, path: Path, X_np: np.ndarray, dtraj: Any
) -> dict:
    bias_arr = _maybe_read_bias(path.with_name(f"{details.shard_id}.npz"))
    uid = _unique_shard_uid(details)
    shard_dtraj = None if dtraj is None else np.asarray(dtraj, dtype=np.int32)

    source_dict = dict(details.source)

    frames_loaded = int(X_np.shape[0])
    (
        frames_declared,
        effective_stride,
        stride_ratio,
        preview_truncated,
    ) = _compute_stride_metadata(details.meta, frames_loaded)

    info: dict[str, Any] = {
        "id": str(uid),
        "start": 0,
        "stop": frames_loaded,
        "dtraj": shard_dtraj,
        "bias_potential": bias_arr,
        "temperature": float(details.temperature_K),
        "source": source_dict,
        "frames_loaded": frames_loaded,
        "frames_declared": frames_declared,
        "effective_frame_stride": effective_stride,
        "preview_truncated": preview_truncated,
    }

    _augment_with_source_metadata(
        info,
        source_dict,
        path,
        effective_stride=effective_stride,
        stride_ratio=stride_ratio,
    )

    return info


def _validate_shard_safety(kinds: Sequence[str], temps: Sequence[float]) -> None:
    if kinds:
        unique_kinds = sorted(set(kinds))
        if len(unique_kinds) > 1:
            raise TemperatureConsistencyError(
                f"Mixed shard kinds not allowed: {unique_kinds}. DEMUX-only is required."
            )
        if unique_kinds[0] != "demux":
            raise TemperatureConsistencyError(
                f"Replica shards are not accepted for learning; found kind={unique_kinds[0]}"
            )
    if temps:
        utemps = sorted(set(round(float(t), 6) for t in temps))
        if len(utemps) > 1:
            raise TemperatureConsistencyError(
                f"Multiple DEMUX temperatures detected: {utemps}. Provide a single-T dataset."
            )


def _fill_shard_offsets(shards_info: Sequence[dict]) -> None:
    offset = 0
    for shard in shards_info:
        length = int(shard["stop"])
        if length <= 0:
            raise TemperatureConsistencyError("Shard length must be positive")
        shard["start"] = offset
        shard["stop"] = offset + length
        offset += length


def load_shards_as_dataset(shard_jsons: Sequence[Path]) -> dict:
    """Load shard JSON files and return a dataset mapping used by the builder.

    The returned dict matches the structure expected by ``pmarlo.transform.build``:
    - keys: ``"X"``, ``"cv_names"``, ``"periodic"``, ``"dtrajs"``, ``"__shards__"``

    Parameters
    ----------
    shard_jsons
        Collection of paths to shard JSON files produced by ``emit``.

    Returns
    -------
    dict
        A dataset mapping containing concatenated CVs and per‑shard metadata.
    """
    aggregated = _aggregate_shard_contents(shard_jsons)
    return aggregated.dataset


def _validate_or_set_refs(
    details: ShardDetails,
    cv_names_ref: tuple[str, ...] | None,
    periodic_ref: tuple[bool, ...] | None,
) -> tuple[tuple[str, ...] | None, tuple[bool, ...] | None]:
    cv_names = details.cv_names
    periodic = details.periodic
    if cv_names_ref is None:
        return cv_names, periodic
    if cv_names != cv_names_ref:
        raise ValueError(f"Shard CV names mismatch: {cv_names} != {cv_names_ref}")
    if periodic != periodic_ref:
        raise ValueError(f"Shard periodic mismatch: {periodic} != {periodic_ref}")
    return cv_names_ref, periodic_ref


def _maybe_read_bias(npz_path: Path) -> np.ndarray | None:
    if not npz_path.exists():
        return None
    with np.load(npz_path) as f:
        if "bias_potential" not in getattr(f, "files", []):
            return None
        b = np.asarray(f["bias_potential"], dtype=np.float64).reshape(-1)
        if b.size == 0:
            return None
        return b


def _dataset_hash(
    dtrajs: List[np.ndarray | None], X: np.ndarray, cv_names: Sequence[str]
) -> str:
    """Compute deterministic dataset hash over CV names, X, and dtrajs list."""

    h = sha256()
    h.update(",".join([str(x) for x in cv_names]).encode("utf-8"))
    Xc = np.ascontiguousarray(X)
    h.update(str(Xc.dtype.str).encode("utf-8"))
    h.update(str(Xc.shape).encode("utf-8"))
    h.update(Xc.tobytes())
    for d in dtrajs:
        if d is None:
            h.update(b"NONE")
        else:
            dc = np.ascontiguousarray(d.astype(np.int32, copy=False))
            h.update(str(dc.dtype.str).encode("utf-8"))
            h.update(str(dc.shape).encode("utf-8"))
            h.update(dc.tobytes())
    return h.hexdigest()


def aggregate_and_build(
    shard_jsons: Sequence[Path],
    *,
    opts: "BuildOpts",
    plan: TransformPlan,
    applied: "AppliedOpts",
    out_bundle: Path,
    **kwargs,
) -> tuple["BuildResult", str]:
    """Load shards, aggregate a dataset, build with the transform pipeline, and archive.

    Returns (BuildResult, dataset_hash_hex).
    """

    aggregated = _aggregate_shard_contents(shard_jsons)

    dataset = aggregated.dataset
    dtrajs = aggregated.dtrajs
    shards_info = aggregated.shards_info
    cv_names = aggregated.cv_names
    X_all = aggregated.X_all

    # Optional unified progress callback forwarding (aliases accepted)
    cb = coerce_progress_callback(kwargs)
    _, _, _, build_result = _transform_build_handles()

    res = build_result(
        dataset, opts=opts, plan=plan, applied=applied, progress_callback=cb
    )
    # Attach shard usage into artifacts for downstream gating checks
    try:
        shard_ids = [str(s.get("id", "")) for s in shards_info]
        art = dict(res.artifacts or {})
        art.setdefault("shards_used", shard_ids)
        art.setdefault("shards_count", int(len(shard_ids)))
        res.artifacts = art  # type: ignore[assignment]
    except Exception:
        pass
    # Optional: merge extra artifacts before writing
    extra_artifacts = kwargs.get("extra_artifacts")
    if isinstance(extra_artifacts, dict) and extra_artifacts:
        try:
            art = dict(res.artifacts or {})
            art.update(extra_artifacts)
            res.artifacts = art  # type: ignore[assignment]
        except Exception:
            pass

    ds_hash = _dataset_hash(dtrajs, X_all, cv_names)
    try:
        new_md = replace(res.metadata, dataset_hash=ds_hash, digest=ds_hash)
        res.metadata = new_md  # type: ignore[assignment]
    except Exception:
        try:
            res.messages.append(f"dataset_hash:{ds_hash}")  # type: ignore[attr-defined]
        except Exception:
            pass

    out_bundle = Path(out_bundle)
    ensure_directory(out_bundle.parent)
    out_bundle.write_text(res.to_json())
    return res, ds_hash
