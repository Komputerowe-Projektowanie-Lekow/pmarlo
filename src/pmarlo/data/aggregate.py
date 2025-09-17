from __future__ import annotations

"""
Aggregate many shard files and build a global analysis envelope.

This module loads compatible shards (same cv_names and periodicity),
concatenates their CV matrices, assembles a dataset dict, and calls
``pmarlo.transform.build.build_result`` to produce MSM/FES/TRAM results.

Outputs a single JSON bundle via BuildResult.to_json() with a dataset hash
recorded into RunMetadata (when available) for end-to-end reproducibility.
"""

from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from typing import List, Sequence

import numpy as np

from pmarlo.data.shard import read_shard
from pmarlo.io.shard_id import parse_shard_id
from pmarlo.transform.build import AppliedOpts, BuildOpts, BuildResult, build_result
from pmarlo.transform.plan import TransformPlan
from pmarlo.transform.progress import coerce_progress_callback
from pmarlo.utils.errors import TemperatureConsistencyError

from .shard_io import load_shard_meta


def _unique_shard_uid(meta, p: Path) -> str:
    """Build a collision-resistant shard identity for aggregation.

    Uses canonical shard ID when possible, falls back to legacy format for compatibility.
    """
    try:
        # Try to parse canonical ID from the source path
        src = dict(getattr(meta, "source", {}))
        src_path_str = (
            src.get("traj")
            or src.get("path")
            or src.get("file")
            or src.get("source_path")
            or str(Path(p).resolve())
        )
        src_path = Path(src_path_str)

        # Attempt to parse canonical shard ID
        try:
            shard_id = parse_shard_id(src_path, require_exists=False)
            return shard_id.canonical()
        except Exception:
            # Fall back to legacy format if parsing fails
            pass

        # Legacy format for backward compatibility
        run_uid = src.get("run_uid") or src.get("run_id") or src.get("run_dir") or ""
        return f"{run_uid}|{getattr(meta,'shard_id','')}|{src_path_str}"
    except Exception:
        # Ultimate fallback
        return f"fallback|{getattr(meta,'shard_id','')}|{str(Path(p).resolve())}"


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
    if not shard_jsons:
        raise ValueError("No shard JSONs provided")

    cv_names_ref: tuple[str, ...] | None = None
    periodic_ref: tuple[bool, ...] | None = None
    X_parts: List[np.ndarray] = []
    dtrajs: List[np.ndarray | None] = []
    shards_info: List[dict] = []

    # Enforce DEMUX-only and single-temperature safety rails
    kinds: list[str] = []
    temps: list[float] = []

    for p in shard_jsons:
        p = Path(p)
        try:
            meta2 = load_shard_meta(p)
            kinds.append(meta2.kind)
            if meta2.kind == "demux":
                # Demux shards must carry temperature_K
                temps.append(float(getattr(meta2, "temperature_K")))
        except Exception:
            # Fallback: use legacy meta but we cannot relax rules below
            pass
        meta, X, dtraj = read_shard(p)
        cv_names_ref, periodic_ref = _validate_or_set_refs(
            meta, cv_names_ref, periodic_ref
        )
        X_np = np.asarray(X, dtype=np.float64)
        X_parts.append(X_np)
        dtrajs.append(None if dtraj is None else np.asarray(dtraj, dtype=np.int32))
        bias_arr = _maybe_read_bias(p.with_name(f"{meta.shard_id}.npz"))
        uid = _unique_shard_uid(meta, p)
        info = {
            "id": str(uid),  # prefer unique ID to avoid collisions
            "legacy_id": str(getattr(meta, "shard_id", p.stem)),
            "start": 0,  # placeholder; filled after we know offsets
            "stop": int(X_np.shape[0]),
            "dtraj": None if dtraj is None else np.asarray(dtraj, dtype=np.int32),
            "bias_potential": bias_arr,
            "temperature": float(meta.temperature),
            # include entire source for downstream validators (kind/run/paths)
            "source": dict(getattr(meta, "source", {})),
        }
        # expose path and run uid for debugging/uniqueness
        try:
            info["source_path"] = str(
                Path(
                    info.get("source", {}).get("traj")
                    or info.get("source", {}).get("path")
                    or info.get("source", {}).get("file")
                    or p
                ).resolve()
            )
            info["run_uid"] = (
                info.get("source", {}).get("run_uid")
                or info.get("source", {}).get("run_id")
                or info.get("source", {}).get("run_dir")
            )
        except Exception:
            pass
        shards_info.append(info)

    # Safety rails
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

    cv_names = tuple(cv_names_ref or tuple())
    periodic = tuple(periodic_ref or tuple())
    X_all = np.vstack(X_parts).astype(np.float64, copy=False)

    # Fill global start/stop offsets for shards_info to allow slice-based access.
    offset = 0
    for s in shards_info:
        length = int(s["stop"])  # currently holds local length
        if length <= 0:
            raise TemperatureConsistencyError("Shard length must be positive")
        s["start"] = offset
        s["stop"] = offset + length
        offset += length

    dataset = {
        "X": X_all,
        "cv_names": cv_names,
        "periodic": periodic,
        "dtrajs": [d for d in dtrajs if d is not None],
        "__shards__": shards_info,
    }
    return dataset


def _validate_or_set_refs(
    meta, cv_names_ref: tuple[str, ...] | None, periodic_ref: tuple[bool, ...] | None
) -> tuple[tuple[str, ...] | None, tuple[bool, ...] | None]:
    if cv_names_ref is None:
        return meta.cv_names, meta.periodic
    if meta.cv_names != cv_names_ref:
        raise ValueError(f"Shard CV names mismatch: {meta.cv_names} != {cv_names_ref}")
    if meta.periodic != periodic_ref:
        raise ValueError(f"Shard periodic mismatch: {meta.periodic} != {periodic_ref}")
    return cv_names_ref, periodic_ref


def _maybe_read_bias(npz_path: Path) -> np.ndarray | None:
    try:
        with np.load(npz_path) as f:
            if "bias_potential" in getattr(f, "files", []):
                b = np.asarray(f["bias_potential"], dtype=np.float64).reshape(-1)
                if b.size > 0:
                    return b
    except Exception:
        return None
    return None


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
    opts: BuildOpts,
    plan: TransformPlan,
    applied: AppliedOpts,
    out_bundle: Path,
    **kwargs,
) -> tuple[BuildResult, str]:
    """Load shards, aggregate a dataset, build with the transform pipeline, and archive.

    Returns (BuildResult, dataset_hash_hex).
    """

    if not shard_jsons:
        raise ValueError("No shard JSONs provided")

    cv_names_ref: tuple[str, ...] | None = None
    periodic_ref: tuple[bool, ...] | None = None
    X_parts: List[np.ndarray] = []
    dtrajs: List[np.ndarray | None] = []
    shards_info: List[dict] = []

    # Enforce DEMUX-only and single-temperature safety rails
    kinds: list[str] = []
    temps: list[float] = []

    for p in shard_jsons:
        p = Path(p)
        try:
            meta2 = load_shard_meta(p)
            kinds.append(meta2.kind)
            if meta2.kind == "demux":
                temps.append(float(getattr(meta2, "temperature_K")))
        except Exception:
            pass
        meta, X, dtraj = read_shard(p)
        cv_names_ref, periodic_ref = _validate_or_set_refs(
            meta, cv_names_ref, periodic_ref
        )
        X_np = np.asarray(X, dtype=np.float64)
        X_parts.append(X_np)
        dtrajs.append(None if dtraj is None else np.asarray(dtraj, dtype=np.int32))
        bias_arr = _maybe_read_bias(p.with_name(f"{meta.shard_id}.npz"))
        uid = _unique_shard_uid(meta, p)
        info = {
            "id": str(uid),
            "legacy_id": str(getattr(meta, "shard_id", p.stem)),
            "start": 0,  # placeholder; filled after we know offsets
            "stop": int(X_np.shape[0]),
            "dtraj": None if dtraj is None else np.asarray(dtraj, dtype=np.int32),
            "bias_potential": bias_arr,
            "temperature": float(meta.temperature),
            # also carry source metadata for validators and cache/cleanup
            "source": dict(getattr(meta, "source", {})),
        }
        shards_info.append(info)

    # Safety rails
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

    cv_names = tuple(cv_names_ref or tuple())
    periodic = tuple(periodic_ref or tuple())
    X_all = np.vstack(X_parts).astype(np.float64, copy=False)

    # Fill global start/stop offsets for shards_info to allow slice-based access.
    offset = 0
    for s in shards_info:
        length = int(s["stop"])  # currently holds local length
        if length <= 0:
            raise TemperatureConsistencyError("Shard length must be positive")
        s["start"] = offset
        s["stop"] = offset + length
        offset += length

    dataset = {
        "X": X_all,
        "cv_names": cv_names,
        "periodic": periodic,
        "dtrajs": [d for d in dtrajs if d is not None],
        "__shards__": shards_info,
    }

    # Optional unified progress callback forwarding (aliases accepted)
    cb = coerce_progress_callback(kwargs)
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
    out_bundle.parent.mkdir(parents=True, exist_ok=True)
    out_bundle.write_text(res.to_json())
    return res, ds_hash
