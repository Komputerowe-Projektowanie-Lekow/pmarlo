from __future__ import annotations

"""Backend orchestration for the PMARLO sharded app.

Functions:
- run_short_sim: launch short REMD and return trajectory files
- extractor_factory: build a CV extractor for Rg + RMSD to a reference
- emit_from_trajs: write shards using pmarlo.data.emit
- aggregate_and_build_bundle: aggregate shards and build a bundle
- recompute_msm_from_shards: discretize globally and estimate MSM (for UI)
"""

from dataclasses import dataclass
import secrets
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from pmarlo.api import (
    run_replica_exchange,
    emit_shards_rg_rmsd as api_emit_rg_rmsd,
    build_from_shards as api_build_from_shards,
)
from pmarlo.data.shard import read_shard
from pmarlo.markov_state_model.free_energy import generate_2d_fes, FESResult
from pmarlo.transform.build import BuildResult, BuildOpts, AppliedOpts
from pmarlo.transform.plan import TransformPlan, TransformStep
from pmarlo.data.aggregate import aggregate_and_build
from pmarlo.io.catalog import build_catalog_from_paths
from pmarlo.io.shards import rescan_shards, prune_missing_shards
from pmarlo.workflow.validation import validate_build_result, format_validation_report
from pmarlo.markov_state_model.bridge import build_simple_msm
from pmarlo.transform.progress import console_progress_cb, tee_progress
from pmarlo.utils.seed import set_global_seed


# Module-level hook to allow chunk-level extract logging from emit calls
_EXTRACT_CB: Optional[Callable[[str, Dict[str, Any]], None]] = None

@dataclass
class SimResult:
    traj_files: List[Path]
    analysis_temperatures: List[float]
    run_dir: Path


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def set_all_seeds(seed: int) -> None:
    """Compatibility wrapper for older callers.

    Delegates to `pmarlo.utils.seed.set_global_seed`.
    """
    set_global_seed(seed)


def _write_log(workspace: Path, name: str, payload: Dict[str, Any]) -> Path:
    logs = Path(workspace) / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    p = logs / f"{name}.json"
    import json

    p.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    return p


def _progress_logger(workspace: Path, stem: str) -> Tuple[Callable[[str, Dict[str, Any]], None], Path]:
    """Return a progress_callback that appends NDJSON lines to logs/<stem>.log."""
    logs = Path(workspace) / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    out = logs / f"{stem}.log"

    def cb(event: str, payload: Dict[str, Any]) -> None:  # ProgressCB signature compatible
        try:
            import json, time

            rec = {"t": datetime.now().isoformat(timespec="seconds"), "event": event, "data": dict(payload)}
            with open(out, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, sort_keys=True, separators=(",", ":")) + "\n")
        except Exception:
            pass

    return cb, out


def run_short_sim(
    pdb: Path,
    base_out: Path,
    temperatures: List[float],
    steps: int,
    *,
    quick: bool = True,
    random_seed: int | None = None,
    start_mode: str = "none",  # "none" | "resume" | "last_frame" | "random_highT"
    start_run: Path | None = None,
    start_from: Path | None = None,
    jitter_start: bool = False,
    jitter_sigma_A: float = 0.05,
    velocity_reseed: bool = False,
    exchange_frequency_steps: int | None = None,
    temperature_schedule_mode: str | None = None,
) -> SimResult:
    """Run a short REMD and return trajectory info.

    Uses pmarlo.api.run_replica_exchange to perform the simulation.
    """
    base_out = Path(base_out)
    seed_tag = f"-seed{int(random_seed)}" if random_seed is not None else ""
    run_dir = base_out / "sims" / f"run-{_now_stamp()}{seed_tag}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # If a fixed seed is provided, seed all RNGs early for determinism.
    # Note: core REMD also receives the seed below via RemdConfig.
    if random_seed is not None:
        set_global_seed(int(random_seed))

    # progress log for simulation
    sim_cb, sim_log = _progress_logger(base_out, f"sim-{_now_stamp()}")
    sim_console = console_progress_cb()
    sim_cb_tee = tee_progress(sim_cb, sim_console)

    # Best-effort disk space check (warn in logs)
    try:
        import shutil

        free_gb = shutil.disk_usage(str(base_out)).free / (1024 ** 3)
        if free_gb < 1.0:
            _write_log(base_out, f"warn-{_now_stamp()}", {"low_disk_gb": round(free_gb, 3)})
    except Exception:
        pass

    # Enable checkpoint/resume for long runs
    try:
        from pmarlo.manager.checkpoint_manager import CheckpointManager  # type: ignore

        cm = CheckpointManager(output_base_dir=str(run_dir), auto_continue=True, max_retries=2)
        cm.setup_run_directory()
    except Exception:
        cm = None

    # Select starting conditions
    start_from_checkpoint = None
    start_from_pdb = None
    # Backward/compat convenience: if explicit start_from path provided, treat as resume-from-run
    if start_from is not None:
        start_mode = "resume"
        start_run = Path(start_from)
    if start_mode == "resume" and start_run is not None:
        cand_ckpts = sorted(Path(start_run).rglob("checkpoint_step_*.pkl"))
        if cand_ckpts:
            start_from_checkpoint = str(cand_ckpts[-1])
    elif start_mode == "last_frame" and start_run is not None:
        try:
            from pmarlo.api import extract_last_frame_to_pdb as _extract_last

            start_from_pdb = run_dir / "start_from_last.pdb"
            # Prefer demux; else fall back to highest replica DCD
            demux = sorted(Path(start_run).rglob("demux_*.dcd"))
            traj = demux[-1] if demux else sorted((Path(start_run) / "replica_exchange").glob("replica_*.dcd"))[-1]
            _extract_last(
                trajectory_file=str(traj),
                topology_pdb=str(pdb),
                out_pdb=str(start_from_pdb),
                jitter_sigma_A=float(jitter_sigma_A) if jitter_start else 0.0,
            )
        except Exception:
            start_from_pdb = None
    elif start_mode == "random_highT" and start_run is not None:
        try:
            from pmarlo.api import extract_random_highT_frame_to_pdb as _extract_rand

            start_from_pdb = run_dir / "start_from_random_highT.pdb"
            _extract_rand(
                run_dir=str(start_run),
                topology_pdb=str(pdb),
                out_pdb=str(start_from_pdb),
                jitter_sigma_A=float(jitter_sigma_A) if jitter_start else 0.0,
            )
        except Exception:
            start_from_pdb = None

    traj_files, analysis_temps = run_replica_exchange(
        pdb_file=str(pdb),
        output_dir=str(run_dir),
        temperatures=[float(t) for t in temperatures],
        total_steps=int(steps),
        quick=bool(quick),
        random_seed=int(random_seed) if random_seed is not None else None,
        start_from_checkpoint=start_from_checkpoint,
        start_from_pdb=str(start_from_pdb) if start_from_pdb else None,
        jitter_start=bool(jitter_start),
        jitter_sigma_A=float(jitter_sigma_A),
        velocity_reseed=bool(velocity_reseed),
        exchange_frequency_steps=int(exchange_frequency_steps) if exchange_frequency_steps is not None else None,
        temperature_schedule_mode=temperature_schedule_mode,
        progress_callback=sim_cb_tee,
        checkpoint_manager=cm,
    )

    log = {
        "time": _now_stamp(),
        "pdb": str(pdb),
        "run_dir": str(run_dir),
        "temperatures": [float(x) for x in temperatures],
        "steps": int(steps),
        "sim_seed": (int(random_seed) if random_seed is not None else None),
        "traj_files": [str(Path(t)) for t in traj_files],
        "analysis_temperatures": [float(x) for x in analysis_temps],
    }
    log["progress_log"] = str(sim_log)
    _write_log(base_out, f"run-{_now_stamp()}-sim", log)
    return SimResult([Path(t) for t in traj_files], [float(x) for x in analysis_temps], run_dir)


def extractor_factory(
    pdb: Path,
    dcd_ref: Optional[Path] = None,
    *,
    stride: int = 1,
) -> Callable[[Path], Tuple[Dict[str, np.ndarray], Optional[np.ndarray], Dict]]:
    """Return an extractor computing Rg and RMSD to a reference.

    - Uses the provided PDB as topology.
    - If dcd_ref is provided, uses its first frame as the global reference;
      otherwise uses the PDB-coordinates as reference.
    """
    import mdtraj as md  # type: ignore

    pdb = Path(pdb)
    topo = md.load(str(pdb))
    ref_traj = None
    if dcd_ref is not None and Path(dcd_ref).exists():
        try:
            ref_traj = md.load(str(dcd_ref), top=str(pdb))[0]
        except Exception:
            ref_traj = topo[0]
    else:
        ref_traj = topo[0]

    ca_sel = topo.topology.select("name CA")
    ca_sel = ca_sel if ca_sel.size else None

    # Use quiet streaming loader to avoid plugin chatter
    from pmarlo.io import trajectory as traj_io

    def _extract(traj_path: Path) -> Tuple[Dict[str, np.ndarray], Optional[np.ndarray], Dict]:
        rg_list: list[np.ndarray] = []
        rmsd_list: list[np.ndarray] = []
        n_frames = 0
        cb = _EXTRACT_CB
        if cb is not None:
            cb("extract_begin", {"traj": str(traj_path), "stride": int(max(1, stride))})
        for chunk in traj_io.iterload(
            str(traj_path), top=str(pdb), stride=int(max(1, stride)), atom_indices=None, chunk=1000
        ):
            try:
                chunk = chunk.superpose(ref_traj, atom_indices=ca_sel)
            except Exception:
                pass
            rg_block = md.compute_rg(chunk).astype(np.float64)
            if cb is not None:
                cb("extract_rg", {"traj": str(traj_path), "frames": int(getattr(chunk, "n_frames", 0))})
            rmsd_block = md.rmsd(chunk, ref_traj, atom_indices=ca_sel).astype(np.float64)
            if cb is not None:
                cb("extract_rmsd", {"traj": str(traj_path)})
            rg_list.append(rg_block)
            rmsd_list.append(rmsd_block)
            n_frames += int(chunk.n_frames)
        rg = np.concatenate(rg_list) if rg_list else np.zeros((0,), dtype=np.float64)
        rmsd = (
            np.concatenate(rmsd_list) if rmsd_list else np.zeros((0,), dtype=np.float64)
        )
        cvs = {"Rg": rg.reshape(-1), "RMSD_ref": rmsd.reshape(-1)}
        dtraj = None  # no discretization at emission time
        info = {"traj": str(traj_path), "n_frames": int(n_frames)}
        if cb is not None:
            cb("extract_end", {"traj": str(traj_path), "frames_total": int(n_frames)})
        return cvs, dtraj, info

    return _extract


def emit_from_trajs(
    traj_files: List[Path],
    shards_dir: Path,
    extractor: Callable[[Path], Tuple[Dict[str, np.ndarray], Optional[np.ndarray], Dict]],
    *,
    temperature: float,
    seed_start: int = 0,
    periodic_by_cv: Optional[Dict[str, bool]] = None,
) -> List[Path]:
    """Emit shards using pmarlo.data.emit.emit_shards_from_trajectories.

    Adds lightweight NDJSON progress log alongside shard emission for long runs.
    """
    periodic = dict(periodic_by_cv or {"Rg": False, "RMSD_ref": False})
    out_dir = Path(shards_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # progress log
    cb, logp = _progress_logger(out_dir.parent.parent if out_dir.parent.name.startswith("run-") else out_dir.parent, f"emit-{_now_stamp()}")
    cb("emit_begin", {"n_inputs": len(traj_files), "out_dir": str(out_dir)})
    try:
        # enable per-chunk extract logging for this emission
        global _EXTRACT_CB
        _EXTRACT_CB = cb
        out = emit_shards_from_trajectories(
            traj_files,
            out_dir=out_dir,
            extract_cvs=extractor,
            seed_start=int(seed_start),
            temperature=float(temperature),
            periodic_by_cv=periodic,
        )
        _EXTRACT_CB = None
        cb("emit_end", {"n_shards": len(out)})
        return out
    except Exception as e:
        _EXTRACT_CB = None
        cb("emit_error", {"error": str(e)})
        raise


def emit_from_trajs_simple(
    traj_files: List[Path],
    shards_dir: Path,
    *,
    pdb: Path,
    ref_dcd: Optional[Path],
    temperature: float,
    seed_start: int = 0,
    stride: int = 1,
    provenance: Dict[str, Any] | None = None,
) -> List[Path]:
    out_dir = Path(shards_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cb, _ = _progress_logger(
        out_dir.parent.parent if out_dir.parent.name.startswith("run-") else out_dir.parent,
        f"emit-{_now_stamp()}",
    )
    emit_console = console_progress_cb()
    emit_cb = tee_progress(cb, emit_console)
    return api_emit_rg_rmsd(
        pdb_file=str(pdb),
        traj_files=[str(p) for p in traj_files],
        out_dir=str(out_dir),
        reference=str(ref_dcd) if ref_dcd else None,
        stride=int(max(1, stride)),
        temperature=float(temperature),
        seed_start=int(seed_start),
        progress_callback=emit_cb,
        provenance=dict(provenance or {}),
    )


def choose_sim_seed(mode: str, fixed: int | None = None) -> int | None:
    """Choose a simulation seed based on mode.

    - "fixed": returns the provided value (normalized to 32-bit int)
    - "auto": generates a new 32-bit random seed
    - "none": returns None
    """
    mode = str(mode).strip().lower()
    if mode == "fixed":
        if fixed is None:
            raise ValueError("Fixed seed mode requires a seed value")
        return int(fixed) & 0xFFFFFFFF
    if mode == "auto":
        return secrets.randbits(32)
    if mode == "none":
        return None
    raise ValueError(f"Unknown seed mode: {mode}")


def _compute_global_edges(
    shard_jsons: List[Path],
    *,
    cv_names: Tuple[str, str],
    bins: Tuple[int, int],
) -> Dict[str, np.ndarray]:
    a_min = [np.inf, np.inf]
    a_max = [-np.inf, -np.inf]
    for p in shard_jsons:
        meta, X, _ = read_shard(p)
        assert tuple(meta.cv_names)[:2] == cv_names, "cv_names mismatch across shards"
        a_min[0] = min(a_min[0], float(np.nanmin(X[:, 0])))
        a_min[1] = min(a_min[1], float(np.nanmin(X[:, 1])))
        a_max[0] = max(a_max[0], float(np.nanmax(X[:, 0])))
        a_max[1] = max(a_max[1], float(np.nanmax(X[:, 1])))
    # guard identical ranges
    if not np.isfinite(a_min[0]) or not np.isfinite(a_max[0]) or a_min[0] == a_max[0]:
        a_max[0] = a_min[0] + 1e-8
    if not np.isfinite(a_min[1]) or not np.isfinite(a_max[1]) or a_min[1] == a_max[1]:
        a_max[1] = a_min[1] + 1e-8
    return {
        cv_names[0]: np.linspace(a_min[0], a_max[0], int(bins[0]) + 1),
        cv_names[1]: np.linspace(a_min[1], a_max[1], int(bins[1]) + 1),
    }


def aggregate_and_build_bundle(
    shard_jsons: List[Path],
    out_bundle: Path,
    *,
    bins: Dict[str, int],
    lag: int,
    seed: int,
    temperature: float,
    learn_cv: bool = False,
    deeptica_params: Optional[Dict[str, Any]] = None,
    workspace: Optional[Path] = None,
    build_type: Optional[str] = None,
) -> Tuple[BuildResult, str, Dict[str, np.ndarray]]:
    """Aggregate shards, build bundle, and return BuildResult + dataset hash.

    Notes:
    - We always append SMOOTH_FES(sigma=0.6) to the plan.
    - When learn_cv=True and extras are installed, the build pipeline will
      switch to learned CVs automatically (engine handles it).
    - We also compute and record global bin edges into applied.notes["cv_bin_edges"].
    """
    if not shard_jsons:
        raise ValueError("No shards to aggregate")

    # Enforce demux-only for Deep‑TICA path to ensure stationarity
    if learn_cv:
        try:
            from pmarlo.transform.build import select_shards as _select
            filtered = _select(shard_jsons, mode="demux")
            if filtered:
                shard_jsons = filtered
        except Exception:
            pass

    # Determine CV names order from first shard
    meta0, _, _ = read_shard(shard_jsons[0])
    names = tuple(meta0.cv_names)
    cv_pair = (names[0], names[1]) if len(names) >= 2 else ("cv1", "cv2")
    # Ensure bins mapping matches cv names; fallback to first two provided values
    if not (cv_pair[0] in bins and cv_pair[1] in bins):
        vals = [int(v) for v in bins.values()]
        if len(vals) >= 2:
            bins = {cv_pair[0]: vals[0], cv_pair[1]: vals[1]}
        else:
            bins = {cv_pair[0]: 32, cv_pair[1]: 32}
    bins_tuple = (int(bins.get(cv_pair[0], 32)), int(bins.get(cv_pair[1], 32)))
    edges = _compute_global_edges(shard_jsons, cv_names=cv_pair, bins=bins_tuple)

    steps: List[TransformStep] = []
    if learn_cv:
        params = dict(deeptica_params or {})
        if "lag" not in params:
            params["lag"] = int(max(1, lag))
        steps.append(TransformStep("LEARN_CV", {"method": "deeptica", **params}))
    steps.append(TransformStep("SMOOTH_FES", {"sigma": 0.6}))
    plan = TransformPlan(steps=tuple(steps))

    opts = BuildOpts(
        seed=int(seed),
        temperature=float(temperature),
        lag_candidates=[int(lag), int(2 * lag), int(3 * lag)],
    )
    notes = {
        "app": "pmarlo-sharded-app",
        "cv_bin_edges": {k: v.tolist() for k, v in edges.items()},
    }
    if workspace is not None:
        # Hint engine where to place learned CV artifacts
        notes["model_dir"] = str(Path(workspace) / "models")
    applied = AppliedOpts(bins=bins, lag=int(lag), macrostates=int((deeptica_params or {}).get("n_states", 5)), notes=notes)

    progress_cb = None
    log_path = None
    if workspace is not None:
        progress_cb, log_path = _progress_logger(Path(workspace), f"build-{_now_stamp()}")
    build_console = console_progress_cb()
    if progress_cb is not None:
        from pmarlo.transform.progress import tee_progress as _tee
        progress_cb = _tee(progress_cb, build_console)
    else:
        progress_cb = build_console
    extra_artifacts = {"build_type": str(build_type)} if build_type else None
    br, ds_hash = aggregate_and_build(
        shard_jsons,
        opts=opts,
        plan=plan,
        applied=applied,
        out_bundle=Path(out_bundle),
        progress_callback=progress_cb,  # forwarded via coerce in data.aggregate
        extra_artifacts=extra_artifacts,
    )

    if workspace is not None:
        info = {
            "bundle": str(out_bundle),
            "dataset_hash": ds_hash,
            "digest": br.metadata.digest,
            "flags": br.flags,
            "steps": [s.name for s in plan.steps],
        }
        if log_path is not None:
            info["progress_log"] = str(log_path)
        _write_log(Path(workspace), f"build-{_now_stamp()}", info)

    return br, ds_hash, edges


def _digitize_block(X: np.ndarray, edges: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    bx = np.digitize(X[:, 0], edges[0]) - 1
    by = np.digitize(X[:, 1], edges[1]) - 1
    bx = np.clip(bx, 0, len(edges[0]) - 2)
    by = np.clip(by, 0, len(edges[1]) - 2)
    n_x = len(edges[0]) - 1
    labels = bx * (len(edges[1]) - 1) + by
    labels = labels.astype(int)
    labels[(bx < 0) | (by < 0)] = -1
    # ensure labels < n_states
    labels = np.clip(labels, -1, n_x * (len(edges[1]) - 1) - 1)
    return labels


def recompute_msm_from_shards(
    shard_jsons: List[Path],
    *,
    edges_by_name: Dict[str, np.ndarray],
    lag: int,
    count_mode: str = "sliding",
) -> Tuple[np.ndarray, np.ndarray]:
    """Discretize shards with provided global edges and build MSM.

    Returns (T, pi). On failure, returns empty arrays.
    """
    if not shard_jsons:
        return np.zeros((0, 0), dtype=float), np.zeros((0,), dtype=float)

    # Determine order
    meta0, _, _ = read_shard(shard_jsons[0])
    names = tuple(meta0.cv_names)
    cv_pair = (names[0], names[1])
    E = (np.asarray(edges_by_name[cv_pair[0]]), np.asarray(edges_by_name[cv_pair[1]]))

    dtrajs: List[np.ndarray] = []
    for p in shard_jsons:
        _, X, _ = read_shard(p)
        d = _digitize_block(np.asarray(X, dtype=float), E)
        dtrajs.append(d)

    # number of states implied by edges
    n_states = (len(E[0]) - 1) * (len(E[1]) - 1)
    T, pi = build_simple_msm(dtrajs, n_states=n_states, lag=int(lag), count_mode=count_mode)
    return T, pi


def recompute_fes_from_shards(
    shard_jsons: List[Path],
    *,
    edges_by_name: Dict[str, np.ndarray],
    temperature: float,
) -> FESResult:
    """Recompute baseline FES from shards using provided global edges.

    Parameters
    ----------
    shard_jsons
        Paths to shard JSON files.
    edges_by_name
        Mapping from CV name to bin edges (1D arrays) for the first two CVs.
    temperature
        Temperature in Kelvin for kT conversion.
    """
    if not shard_jsons:
        raise ValueError("No shards to recompute FES")
    # Determine order and periodicity from first shard
    meta0, _, _ = read_shard(shard_jsons[0])
    names = tuple(meta0.cv_names)
    periodic = tuple(bool(x) for x in meta0.periodic)
    cv_pair = (names[0], names[1]) if len(names) >= 2 else ("cv1", "cv2")
    ex = np.asarray(edges_by_name[cv_pair[0]], dtype=float)
    ey = np.asarray(edges_by_name[cv_pair[1]], dtype=float)
    bins = (int(len(ex) - 1), int(len(ey) - 1))
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    for p in shard_jsons:
        _, X, _ = read_shard(p)
        A = np.asarray(X, dtype=float)
        if A.shape[1] < 2 or A.shape[0] == 0:
            continue
        xs.append(A[:, 0].reshape(-1))
        ys.append(A[:, 1].reshape(-1))
    if not xs or not ys:
        # Produce an empty FESResult-like object with minimal grids to avoid crashes
        arrx = ex
        arry = ey
        empty = np.full((bins[0], bins[1]), np.nan, dtype=float)
        return FESResult(F=empty, xedges=arrx, yedges=arry, metadata={"names": list(cv_pair), "periodic": periodic, "temperature": float(temperature)})
    X_all = np.concatenate(xs)
    Y_all = np.concatenate(ys)
    # Use explicit ranges from edges to match bins exactly
    ranges = ((float(ex[0]), float(ex[-1])), (float(ey[0]), float(ey[-1])))
    return generate_2d_fes(
        X_all,
        Y_all,
        bins=bins,
        temperature=float(temperature),
        periodic=(bool(periodic[0]), bool(periodic[1])),
        ranges=ranges,
        smooth=False,
        inpaint=False,
        min_count=1,
    )


@dataclass
class BuildValidationResult:
    """Comprehensive validation results for a build."""
    is_valid: bool
    messages: List[str]
    warnings: List[str]
    shard_stats: Dict[str, Any]
    weight_stats: Dict[str, Any]
    data_quality: Dict[str, Any]


def validate_build_quality(
    build_result: BuildResult,
    all_available_shards: List[Path],
    workspace: Path
) -> BuildValidationResult:
    """Perform intelligent validation of build quality and data usage.

    Checks:
    - All available shards were used
    - Weights/bias information is present and reasonable
    - Data quality metrics
    - Build completeness
    """
    messages = []
    warnings = []
    shard_stats = {}
    weight_stats = {}
    data_quality = {}

    # Extract shard information from build result
    build_artifacts = build_result.artifacts or {}
    shards_used = build_artifacts.get("shards_used", [])
    build_type = build_artifacts.get("build_type", "unknown")

    # Basic shard validation
    total_available = len(all_available_shards)
    total_used = len(shards_used)

    shard_stats.update({
        "total_available": total_available,
        "total_used": total_used,
        "usage_ratio": total_used / max(1, total_available),
        "build_type": build_type
    })

    if total_used < total_available:
        warnings.append(
            f"Build used {total_used}/{total_available} available shards "
            f"({shard_stats['usage_ratio']:.1%})"
        )
    else:
        messages.append(f"Build used all {total_used} available shards")

    # Analyze individual shard usage using canonical IDs
    try:
        # Build catalog from available shard paths for canonical ID validation
        catalog = build_catalog_from_paths(all_available_shards)
        used_ids = set(shards_used) if shards_used else set()

        # Use comprehensive validation with canonical IDs
        validation_results = validate_build_result(
            {"artifacts": {"shards_used": list(used_ids)}},
            all_available_shards
        )

        # Extract warnings and messages from comprehensive validation
        warnings.extend(validation_results["warnings"])
        messages.extend(validation_results["messages"])

        # Add detailed shard information if available
        if validation_results["shard_table"]:
            shard_table_info = []
            for shard in validation_results["shard_table"][:5]:  # Show first 5
                shard_table_info.append(
                    f"{shard['canonical_id']} ({shard['run_id']}, {shard['source_kind']})"
                )
            if shard_table_info:
                messages.append(f"Sample shards: {', '.join(shard_table_info)}")

    except Exception as e:
        # Fallback to legacy validation if canonical validation fails
        available_ids = {p.stem for p in all_available_shards}
        used_ids = set(shards_used) if shards_used else set()

        missing_shards = available_ids - used_ids
        extra_shards = used_ids - available_ids

        if missing_shards:
            warnings.append(f"Missing shards in build: {sorted(missing_shards)}")
        if extra_shards:
            warnings.append(f"Extra shards in build (shouldn't happen): {sorted(extra_shards)}")

        warnings.append(f"Canonical validation failed ({e}), using legacy validation")

    # Auto-rescan if we detect extra shards used but not found
    try:
        needs_rescan = any(
            (isinstance(w, str) and w.lower().startswith("extra shards in build"))
            for w in warnings
        ) or any(
            (isinstance(m, str) and m.lower().startswith("extra shards in build"))
            for m in messages
        )
        # Some validators record this as an error
        needs_rescan = needs_rescan or any(
            (isinstance(e, str) and "references shards not present" in e.lower())
            for e in []  # no errors collected here yet
        )
        if needs_rescan:
            idx = Path(workspace) / "shards_index.json"
            try:
                rescan_shards([Path(workspace) / "shards"], idx)
                prune_missing_shards(idx)
            except Exception:
                pass
            # Rebuild catalog with fresh listing and re-run validation to clear ghosts
            try:
                refreshed = sorted((Path(workspace) / "shards").rglob("*.json"))
                catalog2 = build_catalog_from_paths(refreshed)
                used_ids2 = set(shards_used) if shards_used else set()
                validation_results2 = validate_build_result(
                    {"artifacts": {"shards_used": list(used_ids2)}},
                    refreshed
                )
                # Replace messages/warnings with refreshed set (keep older informative lines)
                warnings = [w for w in warnings if not (isinstance(w, str) and w.lower().startswith("extra shards in build"))]
                warnings.extend(validation_results2["warnings"])
                messages.extend([m for m in validation_results2["messages"] if m not in messages])
            except Exception:
                pass
    except Exception:
        pass

    # Weight and bias validation
    weight_info = _analyze_weights_and_bias(all_available_shards, workspace)
    weight_stats.update(weight_info)

    if weight_info.get("shards_with_bias", 0) > 0:
        bias_ratio = weight_info["shards_with_bias"] / max(1, weight_info["total_shards"])
        messages.append(
            f"Bias information found in {weight_info['shards_with_bias']}/{weight_info['total_shards']} "
            f"shards ({bias_ratio:.1%})"
        )

        # Check for temperature distribution
        temps = weight_info.get("temperatures", [])
        if len(set(temps)) > 1:
            temp_range = f"{min(temps):.1f}K - {max(temps):.1f}K"
            messages.append(f"Temperature range: {temp_range} with {len(set(temps))} distinct temperatures")
        else:
            messages.append(f"Single temperature: {temps[0]:.1f}K" if temps else "No temperature info")

    # Data quality analysis
    quality_info = _analyze_data_quality(all_available_shards, build_result)
    data_quality.update(quality_info)

    if quality_info.get("total_frames", 0) > 0:
        messages.append(f"Total frames in dataset: {quality_info['total_frames']:,}")
        messages.append(f"Average frames per shard: {quality_info['avg_frames_per_shard']:.0f}")

    # Check for potential issues
    if quality_info.get("zero_variance_cvs"):
        warnings.append(f"Zero variance CVs detected: {quality_info['zero_variance_cvs']}")

    if quality_info.get("nan_values", 0) > 0:
        warnings.append(f"NaN values found in {quality_info['nan_values']} positions")

    # FES quality check
    if hasattr(build_result, 'fes') and build_result.fes is not None:
        fes_meta = getattr(build_result.fes, 'metadata', {})
        empty_bins = fes_meta.get('empty_bins_fraction', 0)
        if empty_bins > 0.3:
            warnings.append(f"High fraction of empty FES bins: {empty_bins:.1%}")
        else:
            messages.append(f"FES quality: {empty_bins:.1%} empty bins")

    # Overall validation
    is_valid = len(warnings) == 0 or all("Missing shards" not in w for w in warnings)

    return BuildValidationResult(
        is_valid=is_valid,
        messages=messages,
        warnings=warnings,
        shard_stats=shard_stats,
        weight_stats=weight_stats,
        data_quality=data_quality
    )


def _analyze_weights_and_bias(shard_paths: List[Path], workspace: Path) -> Dict[str, Any]:
    """Analyze weights and bias information across shards."""
    from pmarlo.data.shard import read_shard

    total_shards = len(shard_paths)
    shards_with_bias = 0
    temperatures = []
    weight_ranges = []

    for path in shard_paths:
        try:
            meta, X, dtraj = read_shard(path)

            # Check for bias information
            bias_path = path.with_name(f"{meta.shard_id}.npz")
            if bias_path.exists():
                shards_with_bias += 1

            # Collect temperature information
            if hasattr(meta, 'temperature') and meta.temperature is not None:
                temperatures.append(float(meta.temperature))

        except Exception:
            continue

    return {
        "total_shards": total_shards,
        "shards_with_bias": shards_with_bias,
        "bias_coverage": shards_with_bias / max(1, total_shards),
        "temperatures": sorted(temperatures) if temperatures else [],
        "unique_temperatures": len(set(temperatures)) if temperatures else 0
    }


def _analyze_data_quality(shard_paths: List[Path], build_result: BuildResult) -> Dict[str, Any]:
    """Analyze data quality metrics."""
    from pmarlo.data.shard import read_shard

    total_frames = 0
    zero_variance_cvs = []
    nan_positions = 0
    shard_sizes = []

    for path in shard_paths:
        try:
            meta, X, dtraj = read_shard(path)

            # Frame count
            frames = X.shape[0] if X is not None else 0
            total_frames += frames
            shard_sizes.append(frames)

            # Check for NaN values
            if X is not None and np.isnan(X).any():
                nan_positions += np.isnan(X).sum()

            # Check for zero variance (constant) CVs
            if X is not None:
                for i, cv_name in enumerate(meta.cv_names):
                    if np.std(X[:, i]) == 0:
                        zero_variance_cvs.append(cv_name)

        except Exception:
            continue

    return {
        "total_frames": total_frames,
        "avg_frames_per_shard": np.mean(shard_sizes) if shard_sizes else 0,
        "shard_sizes": shard_sizes,
        "zero_variance_cvs": list(set(zero_variance_cvs)),
        "nan_values": nan_positions,
        "cv_names": list(set(meta.cv_names)) if 'meta' in locals() else []
    }


def select_latest_baseline_and_deeptica(bundle_paths: List[Path]) -> Tuple[BuildResult | None, BuildResult | None]:
    """Select the latest baseline and Deep‑TICA bundles from a list of paths.

    Baseline:
      - artifacts.build_type == "baseline" OR no mlcv_deeptica applied
    Deep‑TICA:
      - artifacts.build_type == "deeptica" AND mlcv_deeptica.applied == True
    """
    baseline: BuildResult | None = None
    deeptica: BuildResult | None = None
    for p in sorted(bundle_paths, key=lambda q: str(q))[:]:
        pass
    for p in reversed(sorted(bundle_paths, key=lambda q: str(q))):
        try:
            obj = BuildResult.from_json(Path(p).read_text(encoding="utf-8"))
        except Exception:
            continue
        art = (obj.artifacts or {}) if hasattr(obj, "artifacts") else {}
        build_type = art.get("build_type") if isinstance(art, dict) else None
        mlcv = art.get("mlcv_deeptica") if isinstance(art, dict) else None
        applied = bool(isinstance(mlcv, dict) and mlcv.get("applied"))
        if deeptica is None and build_type == "deeptica" and applied:
            deeptica = obj
        if baseline is None and (
            build_type == "baseline" or (not isinstance(mlcv, dict) or not mlcv.get("applied", False))
        ):
            baseline = obj
        if baseline is not None and deeptica is not None:
            break
    return baseline, deeptica
