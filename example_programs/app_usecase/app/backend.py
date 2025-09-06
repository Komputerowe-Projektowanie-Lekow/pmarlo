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
from pmarlo.engine.build import BuildResult
from pmarlo.states.msm_bridge import build_simple_msm
from pmarlo.progress import console_progress_cb, tee_progress


# Module-level hook to allow chunk-level extract logging from emit calls
_EXTRACT_CB: Optional[Callable[[str, Dict[str, Any]], None]] = None

@dataclass
class SimResult:
    traj_files: List[Path]
    analysis_temperatures: List[float]
    run_dir: Path


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


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
) -> SimResult:
    """Run a short REMD and return trajectory info.

    Uses pmarlo.api.run_replica_exchange to perform the simulation.
    """
    base_out = Path(base_out)
    run_dir = base_out / "sims" / f"run-{_now_stamp()}"
    run_dir.mkdir(parents=True, exist_ok=True)

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

    traj_files, analysis_temps = run_replica_exchange(
        pdb_file=str(pdb),
        output_dir=str(run_dir),
        temperatures=[float(t) for t in temperatures],
        total_steps=int(steps),
        quick=bool(quick),
        progress_callback=sim_cb_tee,
        checkpoint_manager=cm,
    )

    log = {
        "time": _now_stamp(),
        "pdb": str(pdb),
        "run_dir": str(run_dir),
        "temperatures": [float(x) for x in temperatures],
        "steps": int(steps),
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
    )


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
        from pmarlo.progress import tee_progress as _tee
        progress_cb = _tee(progress_cb, build_console)
    else:
        progress_cb = build_console
    br, ds_hash = aggregate_and_build(
        shard_jsons,
        opts=opts,
        plan=plan,
        applied=applied,
        out_bundle=Path(out_bundle),
        progress_callback=progress_cb,  # forwarded via coerce in data.aggregate
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
