from __future__ import annotations

"""
Aggregate many shard files and build a global analysis envelope.

This module loads compatible shards (same cv_names and periodicity),
concatenates their CV matrices, assembles a dataset dict, and calls
``pmarlo.engine.build.build_result`` to produce MSM/FES/TRAM results.

Outputs a single JSON bundle via BuildResult.to_json() with a dataset hash
recorded into RunMetadata (when available) for end-to-end reproducibility.
"""

from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from typing import List, Sequence

import numpy as np

from pmarlo.data.shard import read_shard
from pmarlo.engine.build import AppliedOpts, BuildOpts, BuildResult, build_result
from pmarlo.progress import coerce_progress_callback
from pmarlo.transform.plan import TransformPlan


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
    """Load shards, aggregate a dataset, build with engine, and archive.

    Returns (BuildResult, dataset_hash_hex).
    """

    if not shard_jsons:
        raise ValueError("No shard JSONs provided")

    cv_names_ref: tuple[str, ...] | None = None
    periodic_ref: tuple[bool, ...] | None = None
    X_parts: List[np.ndarray] = []
    dtrajs: List[np.ndarray | None] = []
    shards_info: List[dict] = []

    for p in shard_jsons:
        p = Path(p)
        meta, X, dtraj = read_shard(p)
        cv_names_ref, periodic_ref = _validate_or_set_refs(
            meta, cv_names_ref, periodic_ref
        )
        X_np = np.asarray(X, dtype=np.float64)
        X_parts.append(X_np)
        dtrajs.append(None if dtraj is None else np.asarray(dtraj, dtype=np.int32))
        bias_arr = _maybe_read_bias(p.with_name(f"{meta.shard_id}.npz"))
        shards_info.append(
            {
                "start": 0,  # placeholder; filled after we know offsets
                "stop": int(X_np.shape[0]),
                "dtraj": None if dtraj is None else np.asarray(dtraj, dtype=np.int32),
                "bias_potential": bias_arr,
                "temperature": float(meta.temperature),
            }
        )

    cv_names = tuple(cv_names_ref or tuple())
    periodic = tuple(periodic_ref or tuple())
    X_all = np.vstack(X_parts).astype(np.float64, copy=False)

    # Fill global start/stop offsets for shards_info to allow slice-based access.
    offset = 0
    for s in shards_info:
        length = int(s["stop"])  # currently holds local length
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

    ds_hash = _dataset_hash(dtrajs, X_all, cv_names)
    try:
        new_md = replace(res.metadata, dataset_hash=ds_hash)
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
