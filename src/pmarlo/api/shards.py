from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import mdtraj as md
import numpy as np

from pmarlo import constants as const
from pmarlo.data.aggregate import aggregate_and_build as _aggregate_and_build
from pmarlo.data.emit import emit_shards_from_trajectories
from pmarlo.data.shard import read_shard, write_shard
from pmarlo.io import trajectory as traj_io
from pmarlo.shards.indexing import initialise_shard_indices
from pmarlo.transform.build import AppliedOpts as _AppliedOpts
from pmarlo.transform.build import BuildOpts as _BuildOpts
from pmarlo.transform.plan import TransformPlan as _TransformPlan
from pmarlo.transform.plan import TransformStep as _TransformStep
from pmarlo.transform.progress import ProgressReporter
from pmarlo.utils.array import concatenate_or_empty
from pmarlo.utils.path_utils import ensure_directory

logger = logging.getLogger("pmarlo")


def emit_shards_rg_rmsd(
    pdb_file: str | Path,
    traj_files: list[str | Path],
    out_dir: str | Path,
    *,
    reference: str | Path | None = None,
    stride: int = 1,
    temperature: float = 300.0,
    seed_start: int = 0,
    progress_callback=None,
    provenance: dict | None = None,
) -> list[Path]:
    """Stream trajectories and emit shards with Rg and RMSD to a reference.

    This is a convenience wrapper for UI apps. It handles quiet streaming via
    pmarlo.io.trajectory.iterload, alignment to a global reference, and writes
    deterministic shards under ``out_dir``.
    """
    logger.info(
        "[shards] Starting shard emission: n_trajectories=%d, stride=%d, temperature=%.1fK, out_dir=%s",
        len(traj_files),
        stride,
        temperature,
        out_dir,
    )

    pdb_file = Path(pdb_file)
    out_dir = Path(out_dir)
    ensure_directory(out_dir)
    logger.debug("[shards] Loading topology from: %s", pdb_file)
    top0 = md.load(str(pdb_file))
    ref = (
        md.load(str(reference), top=str(pdb_file))[0]
        if reference is not None and Path(reference).exists()
        else top0[0]
    )
    ca_sel = top0.topology.select("name CA")
    ca_sel = ca_sel if ca_sel.size else None

    if reference is not None and Path(reference).exists():
        logger.debug("[shards] Using custom reference structure: %s", reference)
    else:
        logger.debug("[shards] Using first frame as reference")

    logger.debug("[shards] Selected %d CA atoms for alignment", len(ca_sel) if ca_sel is not None else 0)

    def _extract(traj_path: Path):
        rg_parts = []
        rmsd_parts = []
        n = 0
        for chunk in traj_io.iterload(
            str(traj_path), top=str(pdb_file), stride=int(max(1, stride)), chunk=1000
        ):
            try:
                chunk = chunk.superpose(ref, atom_indices=ca_sel)
            except Exception:
                pass
            rg_parts.append(md.compute_rg(chunk).astype(np.float64))
            rmsd_parts.append(
                md.rmsd(chunk, ref, atom_indices=ca_sel).astype(np.float64)
            )
            n += int(chunk.n_frames)

        rg = (
            np.concatenate(rg_parts)
            if rg_parts
            else np.empty((0,), dtype=np.float64)
        )
        rmsd = (
            np.concatenate(rmsd_parts)
            if rmsd_parts
            else np.empty((0,), dtype=np.float64)
        )
        base_src = {"traj": str(traj_path), "n_frames": int(n)}
        if provenance:
            try:
                merged = dict(provenance)
                merged.update(base_src)
                base_src = merged
            except Exception:
                pass
        return (
            {"Rg": rg, "RMSD_ref": rmsd},
            None,
            base_src,
        )

    logger.info("[shards] Processing trajectories and emitting shards...")
    result = emit_shards_from_trajectories(
        [Path(p) for p in traj_files],
        out_dir=out_dir,
        extract_cvs=_extract,
        seed_start=int(seed_start),
        temperature=float(temperature),
        periodic_by_cv={"Rg": False, "RMSD_ref": False},
        progress_callback=progress_callback,
    )

    logger.info("[shards] Emission complete: generated %d shards in %s", len(result), out_dir)
    return result

def emit_shards_rg_rmsd_windowed(
    pdb_file: str | Path,
    traj_files: list[str | Path],
    out_dir: str | Path,
    *,
    reference: str | Path | None = None,
    stride: int = 1,
    temperature: float = 300.0,
    seed_start: int = 0,
    frames_per_shard: int = 5000,
    hop_frames: int | None = None,
    progress_callback=None,
    provenance: dict | None = None,
) -> list[Path]:
    """Emit many overlapping shards per trajectory via a sliding window."""
    logger.info(
        "[shards] Starting windowed shard emission: n_trajectories=%d, window_size=%d, hop=%s, "
        "stride=%d, temperature=%.1fK, out_dir=%s",
        len(traj_files),
        frames_per_shard,
        hop_frames if hop_frames is not None else frames_per_shard,
        stride,
        temperature,
        out_dir,
    )

    pdb_file = Path(pdb_file)
    out_dir = Path(out_dir)
    ensure_directory(out_dir)

    logger.debug("[shards] Loading topology and reference structure from: %s", pdb_file)
    ref, ca_sel = _load_reference_and_selection(md, pdb_file, reference)
    logger.debug("[shards] Selected %d CA atoms for alignment", len(ca_sel) if ca_sel is not None else 0)

    shard_state = initialise_shard_indices(out_dir, seed_start)
    next_idx = shard_state.next_index
    logger.debug("[shards] Initialized shard indexing: starting from index %d", next_idx)

    emit_progress = _make_emit_callback(ProgressReporter(progress_callback))
    shard_paths: list[Path] = []
    files = [Path(p) for p in traj_files]
    files.sort()
    emit_progress(
        "emit_begin",
        {
            "n_inputs": len(files),
            "out_dir": str(out_dir),
            "temperature": float(temperature),
            "current": 0,
            "total": len(files),
        },
    )

    window = max(1, int(frames_per_shard))
    hop = max(1, int(hop_frames) if hop_frames is not None else window)
    logger.info("[shards] Processing %d trajectories with window=%d, hop=%d", len(files), window, hop)

    for index, traj_path in enumerate(files):
        logger.debug("[shards] Processing trajectory %d/%d: %s", index + 1, len(files), traj_path.name)
        emit_progress(
            "emit_one_begin",
            {
                "index": int(index),
                "traj": str(traj_path),
                "current": int(index + 1),
                "total": int(len(files)),
            },
        )

        rg, rmsd, total_frames = _collect_rg_rmsd(
            traj_path,
            pdb_file,
            ref,
            ca_sel,
            stride,
            md,
            traj_io.iterload,
        )

        n_windows_before = len(shard_paths)
        window_paths, next_idx = _emit_windows(
            rg,
            rmsd,
            window,
            hop,
            next_idx,
            shard_state.seed_for,
            out_dir,
            traj_path,
            write_shard,
            temperature,
            replica_id=index,
            provenance=provenance,
        )
        shard_paths.extend(window_paths)
        n_windows_emitted = len(shard_paths) - n_windows_before

        logger.debug(
            "[shards] Trajectory %d: processed %d frames, emitted %d shards",
            index,
            total_frames,
            n_windows_emitted,
        )

        emit_progress(
            "emit_one_end",
            {
                "index": int(index),
                "traj": str(traj_path),
                "frames": int(total_frames),
                "current": int(index + 1),
                "total": int(len(files)),
            },
        )

    emit_progress(
        "emit_end",
        {
            "n_shards": len(shard_paths),
            "current": int(len(files)),
            "total": int(len(files)),
        },
    )

    logger.info("[shards] Windowed emission complete: generated %d shards in %s", len(shard_paths), out_dir)
    return shard_paths

def _load_reference_and_selection(
    md_module: Any,
    pdb_file: Path,
    reference: str | Path | None,
) -> tuple[Any, Any]:
    """Load reference frame and C-alpha selection indices."""

    top0 = md_module.load(str(pdb_file))
    if reference is not None and Path(reference).exists():
        ref = md_module.load(str(reference), top=str(pdb_file))[0]
    else:
        ref = top0[0]
    ca_sel = top0.topology.select("name CA")
    return ref, ca_sel if ca_sel.size else None


def _make_emit_callback(reporter: Any) -> Callable[[str, dict], None]:
    """Wrap progress reporter emission with best-effort error handling."""

    def _emit(event: str, data: dict) -> None:
        try:
            reporter.emit(event, data)
        except Exception:
            pass

    return _emit


def _collect_rg_rmsd(
    traj_path: Path,
    pdb_file: Path,
    reference: Any,
    ca_sel: Any,
    stride: int,
    md_module: Any,
    iterload: Callable[..., Iterable[Any]],
) -> tuple[np.ndarray, np.ndarray, int]:
    """Accumulate radius of gyration and RMSD arrays for a trajectory."""

    rg_parts: list[np.ndarray] = []
    rmsd_parts: list[np.ndarray] = []
    total_frames = 0
    raw_frames = 0
    invalid_total = 0
    stride_val = int(max(1, stride))
    for chunk in iterload(
        str(traj_path),
        top=str(pdb_file),
        stride=stride_val,
        chunk=1000,
    ):
        try:
            chunk = chunk.superpose(reference, atom_indices=ca_sel)
        except Exception:
            pass
        n_chunk = int(chunk.n_frames)
        chunk_start = raw_frames
        raw_frames += n_chunk
        rg_chunk = md_module.compute_rg(chunk).astype(np.float64)
        rmsd_chunk = md_module.rmsd(chunk, reference, atom_indices=ca_sel).astype(
            np.float64
        )
        finite_mask = np.isfinite(rg_chunk) & np.isfinite(rmsd_chunk)
        valid_count = int(np.count_nonzero(finite_mask))
        invalid_count = int(finite_mask.size - valid_count)
        if invalid_count:
            invalid_total += invalid_count
            bad_idx = np.where(~finite_mask)[0]
            for rel_idx in bad_idx[:10]:
                global_idx = chunk_start + int(rel_idx)
                rg_val = rg_chunk[rel_idx]
                rmsd_val = rmsd_chunk[rel_idx]
                issues: list[str] = []
                if not np.isfinite(rg_val):
                    issues.append(
                        "Rg="
                        + (
                            "NaN"
                            if np.isnan(rg_val)
                            else ("+inf" if rg_val > 0 else "-inf")
                        )
                    )
                if not np.isfinite(rmsd_val):
                    issues.append(
                        "RMSD_ref="
                        + (
                            "NaN"
                            if np.isnan(rmsd_val)
                            else ("+inf" if rmsd_val > 0 else "-inf")
                        )
                    )
                logger.warning(
                    "Discarding frame %d from '%s' due to non-finite CVs (%s)",
                    global_idx,
                    traj_path,
                    ", ".join(issues) if issues else "unknown issue",
                )
        if valid_count:
            rg_parts.append(rg_chunk[finite_mask])
            rmsd_parts.append(rmsd_chunk[finite_mask])
            total_frames += valid_count

    if invalid_total:
        logger.warning(
            "Discarded %d frames with invalid CV values while processing '%s'; "
            "retained %d of %d frames.",
            invalid_total,
            traj_path,
            total_frames,
            raw_frames,
        )

    rg = concatenate_or_empty(rg_parts, dtype=np.float64, copy=False)
    rmsd = concatenate_or_empty(rmsd_parts, dtype=np.float64, copy=False)
    return rg, rmsd, total_frames

def _emit_windows(
    rg: np.ndarray,
    rmsd: np.ndarray,
    window: int,
    hop: int,
    next_idx: int,
    seed_for: Callable[[int], int],
    out_dir: Path,
    traj_path: Path,
    write_shard: Callable[..., Path],
    temperature: float,
    replica_id: int,
    provenance: dict | None,
) -> tuple[list[Path], int]:
    """Write overlapping shards for the provided CV time-series."""

    shard_paths: list[Path] = []
    n_frames = int(rg.shape[0])
    if n_frames <= 0:
        return shard_paths, next_idx

    if provenance is None:
        raise ValueError("provenance metadata is required for shard emission")

    base_provenance = dict(provenance)
    required_keys = ("created_at", "kind", "run_id")
    missing = [key for key in required_keys if key not in base_provenance]
    if missing:
        keys = ", ".join(sorted(missing))
        raise ValueError(f"provenance missing required keys: {keys}")

    eff_window = min(window, n_frames)
    eff_hop = min(hop, eff_window)
    for start in range(0, n_frames - eff_window + 1, eff_hop):
        stop = start + eff_window
        segment_id = int(next_idx)
        shard_id = "T{temp}K_seg{segment:04d}_rep{replica:03d}".format(
            temp=int(round(float(temperature))),
            segment=segment_id,
            replica=int(replica_id),
        )
        cvs = {"Rg": rg[start:stop], "RMSD_ref": rmsd[start:stop]}
        source: dict[str, object] = {
            "traj": str(traj_path),
            "range": [int(start), int(stop)],
            "n_frames": int(stop - start),
            "segment_id": segment_id,
            "replica_id": int(replica_id),
            "exchange_window_id": int(base_provenance.get("exchange_window_id", 0)),
        }
        merged = dict(base_provenance)
        merged.update(source)
        source = merged
        shard_path = write_shard(
            out_dir=out_dir,
            shard_id=shard_id,
            cvs=cvs,
            dtraj=None,
            periodic={"Rg": False, "RMSD_ref": False},
            seed=int(seed_for(next_idx)),
            temperature=float(temperature),
            source=source,
        )
        shard_paths.append(shard_path.resolve())
        next_idx += 1

    return shard_paths, next_idx




def build_from_shards(
    shard_jsons: list[str | Path],
    out_bundle: str | Path,
    *,
    bins: dict[str, int],
    lag: int,
    seed: int,
    temperature: float,
    learn_cv: bool = False,
    deeptica_params: dict | None = None,
    n_macrostates: int | None = None,
    notes: dict | None = None,
    progress_callback=None,
    kmeans_kwargs: dict | None = None,
    n_microstates: int | None = None,
):
    """Aggregate shard JSONs and build a bundle with an app-friendly API.

    - Optional LEARN_CV(method="deeptica") is prepended to the plan when requested.
    - Adds SMOOTH_FES step to the plan by default.
    - Computes and records global bin edges into notes["cv_bin_edges"].
    - Optional ``kmeans_kwargs`` are forwarded to the clustering step to tune K-means.
    - ``n_microstates`` enforces the requested number of discrete microstates in the MSM
      build when provided.
    - Returns (BuildResult, dataset_hash).
    """
    logger.info(
        "[shards] Starting build from shards: n_shards=%d, lag=%d, temperature=%.1fK, "
        "learn_cv=%s, out_bundle=%s",
        len(shard_jsons),
        lag,
        temperature,
        learn_cv,
        out_bundle,
    )

    shard_paths = _normalise_shard_inputs(shard_jsons)
    logger.debug("[shards] Validated %d shard paths", len(shard_paths))

    logger.debug("[shards] Reading first shard to infer CV names")
    meta0, _, _ = read_shard(shard_paths[0])
    cv_pair = _infer_cv_pair(meta0)
    logger.debug("[shards] Detected CV pair: %s, %s", cv_pair[0], cv_pair[1])

    logger.info("[shards] Computing global CV bin edges across all shards...")
    edges = _compute_cv_edges(shard_paths, cv_pair, bins, read_shard, np)
    logger.info(
        "[shards] Computed bin edges: %s=[%.3f, %.3f] (%d bins), %s=[%.3f, %.3f] (%d bins)",
        cv_pair[0],
        edges[cv_pair[0]][0],
        edges[cv_pair[0]][-1],
        len(edges[cv_pair[0]]) - 1,
        cv_pair[1],
        edges[cv_pair[1]][0],
        edges[cv_pair[1]][-1],
        len(edges[cv_pair[1]]) - 1,
    )

    model_dir = _extract_model_dir(notes)
    plan = _build_transform_plan(learn_cv, deeptica_params, lag, model_dir)
    if learn_cv:
        logger.info("[shards] Transform plan includes CV learning with deeptica")
    logger.debug("[shards] Transform plan: %d steps", len(plan.steps))

    opts = _build_opts(
        seed,
        temperature,
        lag,
        kmeans_kwargs,
        n_microstates=n_microstates,
    )
    logger.debug(
        "[shards] Build options: n_clusters=%d, n_states=%d, lag_candidates=%s",
        opts.n_clusters,
        opts.n_states,
        opts.lag_candidates,
    )

    all_notes = _merge_notes_with_edges(notes, edges)
    n_states = _determine_macrostates(n_macrostates, deeptica_params)
    logger.debug("[shards] Target macrostates: %d", n_states)

    applied = _AppliedOpts(
        bins=bins,
        lag=int(lag),
        macrostates=n_states,
        notes=all_notes,
    )

    logger.info("[shards] Aggregating shards and building MSM bundle...")
    br, ds_hash = _aggregate_and_build(
        shard_paths,
        opts=opts,
        plan=plan,
        applied=applied,
        out_bundle=Path(out_bundle),
        progress_callback=progress_callback,
    )

    logger.info(
        "[shards] Build complete: bundle saved to %s, dataset_hash=%s",
        out_bundle,
        ds_hash[:16] if ds_hash else "none",
    )
    return br, ds_hash


def _normalise_shard_inputs(shard_jsons: list[str | Path]) -> list[Path]:
    """Validate shard inputs and return canonical Path objects."""
    if not shard_jsons:
        raise ValueError("No shard JSONs provided")
    return [Path(p) for p in shard_jsons]


def _infer_cv_pair(meta: Any) -> tuple[str, str]:
    """Derive the primary CV pair used for downstream binning."""

    names = tuple(meta.cv_names)
    if len(names) >= 2:
        return names[0], names[1]
    return "cv1", "cv2"


def _compute_cv_edges(
    shard_paths: list[Path],
    cv_pair: tuple[str, str],
    bins: Mapping[str, int],
    reader: Callable[[Path], tuple[Any, Any, Any]],
    np_module: Any,
) -> dict[str, np.ndarray]:
    """Compute global bin edges across all shards for the first two CVs."""

    mins = [np_module.inf, np_module.inf]
    maxs = [-np_module.inf, -np_module.inf]
    for path in shard_paths:
        meta, data, _ = reader(path)
        if tuple(meta.cv_names)[:2] != cv_pair:
            raise ValueError("Shard CV names mismatch")
        mins[0] = min(mins[0], float(np_module.nanmin(data[:, 0])))
        mins[1] = min(mins[1], float(np_module.nanmin(data[:, 1])))
        maxs[0] = max(maxs[0], float(np_module.nanmax(data[:, 0])))
        maxs[1] = max(maxs[1], float(np_module.nanmax(data[:, 1])))

    if not np_module.isfinite(mins[0]) or mins[0] == maxs[0]:
        maxs[0] = mins[0] + const.NUMERIC_RELATIVE_TOLERANCE
    if not np_module.isfinite(mins[1]) or mins[1] == maxs[1]:
        maxs[1] = mins[1] + const.NUMERIC_RELATIVE_TOLERANCE

    return {
        cv_pair[0]: np_module.linspace(
            mins[0],
            maxs[0],
            int(bins.get(cv_pair[0], 32)) + 1,
        ),
        cv_pair[1]: np_module.linspace(
            mins[1],
            maxs[1],
            int(bins.get(cv_pair[1], 32)) + 1,
        ),
    }

def _extract_model_dir(notes: dict | None) -> str | None:
    """Return the model directory hint from notes if present."""

    if not notes or not isinstance(notes, dict):
        return None
    try:
        model_dir = notes.get("model_dir")
    except Exception:
        model_dir = None
    return model_dir


def _build_transform_plan(
    learn_cv: bool,
    deeptica_params: dict | None,
    lag: int,
    model_dir: str | None,
) -> _TransformPlan:
    """Assemble the transform plan with optional Deeptica learning."""

    steps: list[_TransformStep] = []
    if learn_cv:
        params = dict(deeptica_params or {})
        params.setdefault("lag", int(max(1, lag)))
        if model_dir and "model_dir" not in params:
            params["model_dir"] = model_dir
        steps.append(_TransformStep("LEARN_CV", {"method": "deeptica", **params}))
    steps.append(_TransformStep("SMOOTH_FES", {"sigma": 0.6}))
    return _TransformPlan(steps=tuple(steps))


def _build_opts(
    seed: int,
    temperature: float,
    lag: int,
    kmeans_kwargs: dict | None = None,
    *,
    n_microstates: int | None = None,
) -> _BuildOpts:
    """Create BuildOpts with a simple lag candidate ladder."""

    DEFAULT_N_STATES = 50
    DEFAULT_N_CLUSTERS = 200

    microstate_kwargs = dict(kmeans_kwargs or {})
    resolved_states: int
    if n_microstates is not None:
        try:
            resolved_states = int(n_microstates)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"n_microstates must be an integer, received {n_microstates!r}."
            ) from exc
        if resolved_states <= 0:
            raise ValueError("n_microstates must be a positive integer.")
    else:
        resolved_states = DEFAULT_N_STATES

    resolved_clusters = (
        resolved_states if n_microstates is not None else DEFAULT_N_CLUSTERS
    )

    return _BuildOpts(
        seed=int(seed),
        temperature=float(temperature),
        lag_candidates=(int(lag), int(2 * lag), int(3 * lag)),
        n_clusters=int(resolved_clusters),
        n_states=int(resolved_states),
        kmeans_kwargs=microstate_kwargs,
    )


def _merge_notes_with_edges(
    notes: dict | None,
    edges: Mapping[str, np.ndarray],
) -> dict:
    """Merge user notes with computed CV bin edges."""

    merged = dict(notes or {})
    merged.setdefault("cv_bin_edges", {k: v.tolist() for k, v in edges.items()})
    return merged


def _determine_macrostates(
    n_macrostates: int | None,
    deeptica_params: dict | None,
) -> int:
    """Decide how many macrostates to request for downstream analysis."""

    if n_macrostates is not None:
        return int(n_macrostates)
    return int((deeptica_params or {}).get("n_states", 5))