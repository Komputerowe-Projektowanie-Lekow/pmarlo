from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import mdtraj as md  # type: ignore
import numpy as np

from .analysis.ck import run_ck as _run_ck
from .cluster.micro import cluster_microstates as _cluster_microstates
from .features import get_feature
from .features.base import parse_feature_spec
from .fes.surfaces import FESResult
from .fes.surfaces import generate_2d_fes as _generate_2d_fes
from .markov_state_model.markov_state_model import EnhancedMSM as MarkovStateModel
from .reduce.reducers import pca_reduce, tica_reduce, vamp_reduce
from .replica_exchange.config import RemdConfig
from .replica_exchange.replica_exchange import ReplicaExchange
from .reporting.export import write_conformations_csv_json
from .reporting.plots import save_fes_contour, save_transition_matrix_heatmap
from .states.msm_bridge import build_simple_msm as _build_simple_msm
from .states.msm_bridge import compute_macro_mfpt as _compute_macro_mfpt
from .states.msm_bridge import compute_macro_populations as _compute_macro_populations
from .states.msm_bridge import lump_micro_to_macro_T as _lump_micro_to_macro_T
from .states.msm_bridge import pcca_like_macrostates as _pcca_like
from .states.picker import pick_frames_around_minima as _pick_frames_around_minima
from .utils.msm_utils import candidate_lag_ladder

logger = logging.getLogger("pmarlo")


def compute_features(
    traj: md.Trajectory, feature_specs: Sequence[str], cache_path: Optional[str] = None
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """Compute features for the given trajectory.

    Phase A: minimal implementation that supports ["phi_psi"].
    Caching is a no-op for now to preserve behavior.
    Returns (X, columns, periodic).
    """
    columns: List[str] = []
    feats: List[np.ndarray] = []
    periodic_flags: List[np.ndarray] = []

    for spec in feature_specs:
        # Phase B: parse spec into (feature_name, kwargs)
        feat_name, kwargs = parse_feature_spec(spec)
        fc = get_feature(feat_name)
        X = fc.compute(traj, **kwargs)
        if X.size == 0:
            continue
        feats.append(X)
        # Column labels: prefer feature-provided labels; else best-effort generic names
        n_cols = X.shape[1]
        labels = getattr(fc, "labels", None)
        if isinstance(labels, list) and len(labels) == n_cols:
            columns.extend(labels)
        elif feat_name == "phi_psi" and n_cols > 0:
            # Fallback for phi/psi if labels missing: packed as [phi..., psi...]
            half = max(0, n_cols // 2)
            columns.extend([f"phi_{i}" for i in range(half)])
            columns.extend([f"psi_{i}" for i in range(n_cols - half)])
        else:
            label_base = feat_name
            if feat_name == "distance_pair" and "i" in kwargs and "j" in kwargs:
                label_base = f"dist:atoms:{kwargs['i']}-{kwargs['j']}"
            for i in range(n_cols):
                columns.append(f"{label_base}_{i}" if n_cols > 1 else label_base)
        periodic_flags.append(fc.is_periodic())

    if feats:
        X_all = np.hstack(feats)
        periodic = (
            np.concatenate(periodic_flags)
            if periodic_flags
            else np.zeros((X_all.shape[1],), dtype=bool)
        )
    else:
        X_all = np.zeros((traj.n_frames, 0), dtype=float)
        periodic = np.zeros((0,), dtype=bool)

    return X_all, columns, periodic


def reduce_features(
    X: np.ndarray,
    method: Literal["pca", "tica", "vamp"] = "tica",
    lag: int = 10,
    n_components: int = 2,
) -> np.ndarray:
    if method == "pca":
        return pca_reduce(X, n_components=n_components)
    if method == "tica":
        return tica_reduce(X, lag=lag, n_components=n_components)
    if method == "vamp":
        # Try a small set of candidate dims to select by VAMP score
        candidates = [n_components, max(1, n_components - 1), n_components + 1]
        return vamp_reduce(X, lag=lag, n_components=n_components, score_dims=candidates)
    raise ValueError(f"Unknown reduction method: {method}")


def cluster_microstates(
    Y: np.ndarray,
    method: Literal["auto", "minibatchkmeans", "kmeans"] = "auto",
    n_states: int | Literal["auto"] = "auto",
    random_state: int | None = 42,
    minibatch_threshold: int = 5_000_000,
    **kwargs,
) -> np.ndarray:
    """Public wrapper around :func:`cluster.micro.cluster_microstates`.

    Parameters
    ----------
    Y:
        Reduced feature array.
    method:
        Clustering algorithm to use.  ``"auto"`` selects
        ``MiniBatchKMeans`` when the dataset size exceeds
        ``minibatch_threshold``.
    n_states:
        Number of states or ``"auto"`` to select via silhouette.
    random_state:
        Seed for deterministic clustering.  When ``None`` the global NumPy
        random state is used.
    minibatch_threshold:
        Product of frames and features above which ``MiniBatchKMeans`` is used
        when ``method="auto"``.

    Returns
    -------
    np.ndarray
        Integer labels per frame.
    """

    result = _cluster_microstates(
        Y,
        method=method,
        n_states=n_states,
        random_state=random_state,
        minibatch_threshold=minibatch_threshold,
        **kwargs,
    )
    return result.labels


def generate_free_energy_surface(
    cv1: np.ndarray,
    cv2: np.ndarray,
    bins: Tuple[int, int] = (100, 100),
    temperature: float = 300.0,
    periodic: Tuple[bool, bool] = (False, False),
    smoothing: Optional[Literal["cosine"]] = None,
) -> FESResult:
    """Generate a 2D free-energy surface.

    Parameters
    ----------
    cv1, cv2
        Collective variable samples.
    bins
        Number of histogram bins in ``(x, y)``.
    temperature
        Simulation temperature in Kelvin.
    periodic
        Flags indicating whether each dimension is periodic.
    smoothing
        Placeholder smoothing option; currently only ``"cosine"`` is
        recognised.

    Returns
    -------
    FESResult
        Dataclass containing the free-energy surface and bin edges.
    """

    # Map smoothing flag to gaussian sigma; cosine placeholder maps to 0.6
    sigma = 0.6 if smoothing == "cosine" else 0.6
    out = _generate_2d_fes(
        cv1,
        cv2,
        bins=bins,
        temperature=temperature,
        periodic=periodic,
        smoothing_sigma=sigma,
    )
    return out


def build_msm_from_labels(
    dtrajs: list[np.ndarray], n_states: Optional[int] = None, lag: int = 20
) -> tuple[np.ndarray, np.ndarray]:
    return _build_simple_msm(dtrajs, n_states=n_states, lag=lag)


def compute_macrostates(T: np.ndarray, n_macrostates: int = 4) -> Optional[np.ndarray]:
    return _pcca_like(T, n_macrostates=n_macrostates)


def macrostate_populations(
    pi_micro: np.ndarray, micro_to_macro: np.ndarray
) -> np.ndarray:
    return _compute_macro_populations(pi_micro, micro_to_macro)


def macro_transition_matrix(
    T_micro: np.ndarray, pi_micro: np.ndarray, micro_to_macro: np.ndarray
) -> np.ndarray:
    return _lump_micro_to_macro_T(T_micro, pi_micro, micro_to_macro)


def macro_mfpt(T_macro: np.ndarray) -> np.ndarray:
    return _compute_macro_mfpt(T_macro)


def _fes_pair_from_requested(
    cols: Sequence[str], requested: Optional[Tuple[str, str]]
) -> Tuple[int, int] | None:
    if requested is None:
        return None
    a, b = requested
    if a not in cols or b not in cols:
        raise ValueError(
            (
                f"Requested FES pair {requested} not found. Available columns "
                f"include: {cols[:12]} ..."
            )
        )
    return cols.index(a), cols.index(b)


def _fes_build_phi_psi_maps(
    cols: Sequence[str],
) -> tuple[dict[int, int], dict[int, int]]:
    phi_map_local: dict[int, int] = {}
    psi_map_local: dict[int, int] = {}
    for k, c in enumerate(cols):
        if c.startswith("phi:res"):
            try:
                rid = int(c.split("res")[-1])
                phi_map_local[rid] = k
            except Exception:
                continue
        if c.startswith("psi:res"):
            try:
                rid = int(c.split("res")[-1])
                psi_map_local[rid] = k
            except Exception:
                continue
    return phi_map_local, psi_map_local


def _fes_pair_from_phi_psi_maps(
    cols: Sequence[str],
) -> Tuple[int, int, int] | None:
    phi_map_local, psi_map_local = _fes_build_phi_psi_maps(cols)
    common_residues = sorted(set(phi_map_local).intersection(psi_map_local))
    if not common_residues:
        return None
    rid0 = common_residues[0]
    return phi_map_local[rid0], psi_map_local[rid0], rid0


def _fes_highest_variance_pair(X: np.ndarray) -> Tuple[int, int] | None:
    if X.shape[1] < 1:
        return None
    variances = np.var(X, axis=0)
    order = np.argsort(variances)[::-1]
    if order.shape[0] == 1:
        return int(order[0]), int(order[0]) if X.shape[1] == 1 else 0
    return int(order[0]), int(order[1])


def _fes_periodic_pair_flags(
    periodic: np.ndarray, i_idx: int, j_idx: int
) -> Tuple[bool, bool]:
    pi = bool(periodic[i_idx]) if len(periodic) > i_idx else False
    pj = bool(periodic[j_idx]) if len(periodic) > j_idx else False
    return pi, pj


def select_fes_pair(
    X: np.ndarray,
    cols: Sequence[str],
    periodic: np.ndarray,
    requested: Optional[Tuple[str, str]] = None,
    ensure: bool = True,
) -> Tuple[int, int, bool, bool]:
    """Select a pair of CV columns for FES.

    Preference order:
    1) If requested is provided, return those indices (or raise if missing).
    2) Pair phi:resN with psi:resN where available (lowest residue index).
    3) Fallback: highest-variance distinct pair if ensure=True.
    """

    # 1) Requested
    pair = _fes_pair_from_requested(cols, requested)
    if pair is not None:
        i, j = pair
        pi, pj = _fes_periodic_pair_flags(periodic, i, j)
        return i, j, pi, pj

    # 2) Residue-aware phi/psi pairing
    pair = _fes_pair_from_phi_psi_maps(cols)
    if pair is not None:
        i, j, rid = pair
        pi, pj = _fes_periodic_pair_flags(periodic, i, j)
        logger.info("FES φ/ψ pair selected: phi_res=%d, psi_res=%d", rid, rid)
        return i, j, pi, pj

    # 3) Highest-variance fallback
    if ensure:
        hv = _fes_highest_variance_pair(X)
        if hv is not None:
            i, j = hv
            pi, pj = _fes_periodic_pair_flags(periodic, i, j)
            return i, j, pi, pj

    raise RuntimeError("No suitable FES pair could be selected.")


def sanitize_label_for_filename(name: str) -> str:
    return name.replace(":", "-").replace(" ", "_")


def generate_fes_and_pick_minima(
    X: np.ndarray,
    cols: Sequence[str],
    periodic: np.ndarray,
    requested_pair: Optional[Tuple[str, str]] = None,
    bins: Tuple[int, int] = (60, 60),
    temperature: float = 300.0,
    smoothing: Optional[Literal["cosine"]] = "cosine",
    deltaF_kJmol: float = 3.0,
) -> Dict[str, Any]:
    """High-level helper to generate a 2D FES on selected pair and pick minima.

    Returns dict with keys: i, j, names, periodic_flags, fes (dict), minima (dict).
    """
    i, j, per_i, per_j = select_fes_pair(
        X, cols, periodic, requested=requested_pair, ensure=True
    )
    cv1 = X[:, i].reshape(-1)
    cv2 = X[:, j].reshape(-1)
    # Convert angles to degrees when labeling suggests dihedrals
    name_i = cols[i]
    name_j = cols[j]
    if name_i.startswith("phi") or name_i.startswith("psi"):
        cv1 = np.degrees(cv1)
    if name_j.startswith("phi") or name_j.startswith("psi"):
        cv2 = np.degrees(cv2)
    if np.allclose(cv1, cv2):
        raise RuntimeError(
            "Selected FES pair are identical; aborting to avoid diagonal artifact."
        )
    fes = generate_free_energy_surface(
        cv1,
        cv2,
        bins=bins,
        temperature=temperature,
        periodic=(per_i, per_j),
        smoothing=smoothing,
    )
    minima = _pick_frames_around_minima(
        cv1, cv2, fes.F, fes.xedges, fes.yedges, deltaF_kJmol=deltaF_kJmol
    )
    return {
        "i": int(i),
        "j": int(j),
        "names": (name_i, name_j),
        "periodic_flags": (bool(per_i), bool(per_j)),
        "fes": fes,
        "minima": minima,
    }


# ------------------------------ High-level wrappers ------------------------------


def run_replica_exchange(
    pdb_file: str | Path,
    output_dir: str | Path,
    temperatures: List[float],
    total_steps: int,
) -> Tuple[List[str], List[float]]:
    """Run REMD and return (trajectory_files, analysis_temperatures).

    Attempts demultiplexing to ~300 K; falls back to per-replica trajectories.
    """
    remd_out = Path(output_dir) / "replica_exchange"

    equil = min(total_steps // 10, 200 if total_steps <= 2000 else 2000)
    dcd_stride = max(1, int(total_steps // 5000))
    exchange_frequency = max(100, total_steps // 20)

    remd = ReplicaExchange.from_config(
        RemdConfig(
            pdb_file=str(pdb_file),
            temperatures=temperatures,
            output_dir=str(remd_out),
            exchange_frequency=exchange_frequency,
            auto_setup=False,
            dcd_stride=dcd_stride,
        )
    )
    remd.plan_reporter_stride(
        total_steps=int(total_steps), equilibration_steps=int(equil), target_frames=5000
    )
    remd.setup_replicas()
    remd.run_simulation(total_steps=int(total_steps), equilibration_steps=int(equil))

    # Demultiplex best-effort
    demuxed = remd.demux_trajectories(
        target_temperature=300.0, equilibration_steps=int(equil)
    )
    if demuxed:
        try:
            traj = md.load(str(demuxed), top=str(pdb_file))
            reporter_stride = getattr(remd, "reporter_stride", None)
            eff_stride = int(
                reporter_stride
                if reporter_stride
                else max(1, getattr(remd, "dcd_stride", 1))
            )
            production_steps = max(0, int(total_steps) - int(equil))
            expected = max(1, production_steps // eff_stride)
            if traj.n_frames >= expected:
                return [str(demuxed)], [300.0]
        except Exception:
            pass

    traj_files = [str(f) for f in remd.trajectory_files]
    return traj_files, temperatures


def analyze_msm(
    trajectory_files: List[str],
    topology_pdb: str | Path,
    output_dir: str | Path,
    feature_type: str = "phi_psi",
    analysis_temperatures: Optional[List[float]] = None,
    use_effective_for_uncertainty: bool = True,
    use_tica: bool = True,
    random_state: int | None = 42,
    traj_stride: int = 1,
    atom_selection: str | Sequence[int] | None = None,
    chunk_size: int = 1000,
) -> Path:
    """Build and analyze an MSM, saving plots and artifacts.

    Parameters
    ----------
    trajectory_files:
        Trajectory file paths.
    topology_pdb:
        Topology in PDB format.
    output_dir:
        Destination directory.
    feature_type:
        Feature specification string.
    analysis_temperatures:
        Optional list of temperatures for analysis.
    use_effective_for_uncertainty:
        Whether to use effective counts for uncertainty.
    use_tica:
        Whether to apply TICA reduction.
    random_state:
        Seed for deterministic clustering. ``None`` uses the global state.
    traj_stride:
        Stride for loading trajectory frames.
    atom_selection:
        MDTraj atom selection string or explicit atom indices used when
        loading trajectories.
    chunk_size:
        Number of frames per chunk when streaming trajectories from disk.

    Returns
    -------
    Path
        The analysis output directory.
    """
    msm_out = Path(output_dir) / "msm_analysis"

    msm = MarkovStateModel(
        trajectory_files=trajectory_files,
        topology_file=str(topology_pdb),
        temperatures=analysis_temperatures or [300.0],
        output_dir=str(msm_out),
        random_state=random_state,
    )
    if use_effective_for_uncertainty:
        msm.count_mode = "sliding"
    msm.load_trajectories(
        stride=traj_stride, atom_selection=atom_selection, chunk_size=chunk_size
    )
    ft = feature_type
    if use_tica and ("tica" not in feature_type.lower()):
        ft = f"{feature_type}_tica"
    msm.compute_features(feature_type=ft)

    # Cluster
    N_CLUSTERS = 8
    msm.cluster_features(n_states=int(N_CLUSTERS))

    # Method selection
    method = (
        "tram"
        if analysis_temperatures
        and len(analysis_temperatures) > 1
        and len(trajectory_files) > 1
        else "standard"
    )

    # ITS and lag selection
    try:
        total_frames = sum(t.n_frames for t in msm.trajectories)
    except Exception:
        total_frames = 0
    max_lag = 250
    try:
        if total_frames > 0:
            max_lag = int(min(500, max(150, total_frames // 5)))
    except Exception:
        max_lag = 250
    candidate_lags = candidate_lag_ladder(min_lag=1, max_lag=max_lag)
    msm.build_msm(lag_time=5, method=method)
    msm.compute_implied_timescales(lag_times=candidate_lags, n_timescales=3)

    chosen_lag = 10
    try:
        import numpy as _np  # type: ignore

        lags = _np.array(msm.implied_timescales["lag_times"])  # type: ignore[index]
        its = _np.array(msm.implied_timescales["timescales"])  # type: ignore[index]
        scores: List[float] = []
        for idx in range(len(lags)):
            if idx == 0:
                scores.append(float("inf"))
                continue
            prev = its[idx - 1]
            cur = its[idx]
            mask = _np.isfinite(prev) & _np.isfinite(cur) & (_np.abs(prev) > 0)
            if _np.count_nonzero(mask) == 0:
                scores.append(float("inf"))
                continue
            rel = float(_np.mean(_np.abs((cur[mask] - prev[mask]) / prev[mask])))
            scores.append(rel)
        start_idx = min(3, len(scores) - 1)
        region = scores[start_idx:]
        if region:
            min_idx = int(_np.nanargmin(region)) + start_idx
            chosen_lag = int(lags[min_idx])
    except Exception:
        chosen_lag = 10

    msm.build_msm(lag_time=chosen_lag, method=method)

    # CK test with macro → micro fallback
    try:
        _run_ck(msm.dtrajs, msm.lag_time, msm.output_dir, macro_k=3)
    except Exception:
        pass

    try:
        total_frames_fes = sum(t.n_frames for t in msm.trajectories)
    except Exception:
        total_frames_fes = 0
    adaptive_bins = max(20, min(50, int((total_frames_fes or 0) ** 0.5))) or 20
    msm.generate_free_energy_surface(
        cv1_name="phi", cv2_name="psi", bins=int(adaptive_bins), temperature=300.0
    )
    msm.plot_free_energy_surface(save_file="free_energy_surface", interactive=False)
    msm.plot_implied_timescales(save_file="implied_timescales")
    msm.plot_free_energy_profile(save_file="free_energy_profile")
    msm.create_state_table()
    msm.extract_representative_structures(save_pdb=True)
    msm.save_analysis_results()

    return msm_out


def find_conformations(
    topology_pdb: str | Path,
    trajectory_choice: str | Path,
    output_dir: str | Path,
    feature_specs: Optional[List[str]] = None,
    requested_pair: Optional[Tuple[str, str]] = None,
    traj_stride: int = 1,
    atom_selection: str | Sequence[int] | None = None,
    chunk_size: int = 1000,
) -> Path:
    """Find MSM- and FES-based representative conformations.

    Parameters
    ----------
    topology_pdb:
        Topology file in PDB format.
    trajectory_choice:
        Trajectory file to analyze.
    output_dir:
        Directory where results are written.
    feature_specs:
        Feature specification strings.
    requested_pair:
        Optional pair of feature names for FES plotting.
    traj_stride:
        Stride for loading trajectory frames.
    atom_selection:
        MDTraj atom selection string or indices used when loading the
        trajectory.
    chunk_size:
        Frames per chunk when streaming the trajectory.

    Returns
    -------
    Path
        The output directory path.
    """
    out = Path(output_dir)

    atom_indices: Sequence[int] | None = None
    if atom_selection is not None:
        topo = md.load_topology(str(topology_pdb))
        if isinstance(atom_selection, str):
            atom_indices = topo.select(atom_selection)
        else:
            atom_indices = list(atom_selection)

    logger.info(
        "Streaming trajectory %s with stride=%d, chunk=%d%s",
        trajectory_choice,
        traj_stride,
        chunk_size,
        f", selection={atom_selection}" if atom_selection else "",
    )
    traj: md.Trajectory | None = None
    from pmarlo.io import trajectory as traj_io

    for chunk in traj_io.iterload(
        str(trajectory_choice),
        top=str(topology_pdb),
        stride=traj_stride,
        atom_indices=atom_indices,
        chunk=chunk_size,
    ):
        traj = chunk if traj is None else traj.join(chunk)
    if traj is None:
        raise ValueError("No frames loaded from trajectory")

    specs = feature_specs if feature_specs is not None else ["phi_psi"]
    X, cols, periodic = compute_features(traj, feature_specs=specs)
    Y = reduce_features(X, method="vamp", lag=10, n_components=3)
    labels = cluster_microstates(Y, method="minibatchkmeans", n_states=8)

    dtrajs = [labels]
    observed_states = int(np.max(labels)) + 1 if labels.size else 0
    T, pi = build_msm_from_labels(dtrajs, n_states=observed_states, lag=10)
    macrostates = compute_macrostates(T, n_macrostates=4)
    _ = save_transition_matrix_heatmap(T, str(out), name="transition_matrix.png")

    items: List[dict] = []
    if macrostates is not None:
        macro_of_micro = macrostates
        macro_per_frame = macro_of_micro[labels]
        pi_macro = macrostate_populations(pi, macro_of_micro)
        T_macro = macro_transition_matrix(T, pi, macro_of_micro)
        mfpt = macro_mfpt(T_macro)

        for macro_id in sorted(set(int(m) for m in macro_per_frame)):
            idxs = np.where(macro_per_frame == macro_id)[0]
            if idxs.size == 0:
                continue
            centroid = np.mean(Y[idxs], axis=0)
            deltas = np.linalg.norm(Y[idxs] - centroid, axis=1)
            best_local = int(idxs[int(np.argmin(deltas))])
            best_local = int(best_local % max(1, traj.n_frames))
            rep_path = out / f"macrostate_{macro_id:02d}_rep.pdb"
            try:
                traj[best_local].save_pdb(str(rep_path))
            except Exception:
                pass
            items.append(
                {
                    "type": "MSM",
                    "macrostate": int(macro_id),
                    "representative_frame": int(best_local),
                    "population": (
                        float(pi_macro[macro_id])
                        if pi_macro.size > macro_id
                        else float("nan")
                    ),
                    "mfpt_to": {
                        str(int(j)): float(mfpt[int(macro_id), int(j)])
                        for j in range(mfpt.shape[1])
                    },
                    "rep_pdb": str(rep_path),
                }
            )

    adaptive_bins = max(30, min(80, int((getattr(traj, "n_frames", 0) or 1) ** 0.5)))
    fes_info = generate_fes_and_pick_minima(
        X,
        cols,
        periodic,
        requested_pair=requested_pair,
        bins=(adaptive_bins, adaptive_bins),
        temperature=300.0,
        smoothing="cosine",
        deltaF_kJmol=3.0,
    )
    names = fes_info["names"]
    fes = fes_info["fes"]
    minima = fes_info["minima"]
    fname = f"fes_{sanitize_label_for_filename(names[0])}_vs_{sanitize_label_for_filename(names[1])}.png"
    _ = save_fes_contour(
        fes.F, fes.xedges, fes.yedges, names[0], names[1], str(out), fname
    )

    for idx, entry in enumerate(minima.get("minima", [])):
        frames = entry.get("frames", [])
        if not frames:
            continue
        best_local = int(frames[0])
        rep_path = out / f"state_{idx:02d}_rep.pdb"
        try:
            traj[best_local].save_pdb(str(rep_path))
        except Exception:
            pass
        items.append(
            {
                "type": "FES_MIN",
                "state": int(idx),
                "representative_frame": int(best_local),
                "num_frames": int(entry.get("num_frames", 0)),
                "pair": {"x": names[0], "y": names[1]},
                "rep_pdb": str(rep_path),
            }
        )

    write_conformations_csv_json(str(out), items)
    return out
