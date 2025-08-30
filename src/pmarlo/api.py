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
from .markov_state_model.enhanced_msm import EnhancedMSM as MarkovStateModel
from .progress import coerce_progress_callback
from .reduce.reducers import pca_reduce, tica_reduce, vamp_reduce
from .replica_exchange.config import RemdConfig
from .replica_exchange.replica_exchange import ReplicaExchange
from .reporting.export import write_conformations_csv_json
from .reporting.plots import (
    save_fes_contour,
    save_pmf_line,
    save_transition_matrix_heatmap,
)
from .states.msm_bridge import build_simple_msm as _build_simple_msm
from .states.msm_bridge import compute_macro_mfpt as _compute_macro_mfpt
from .states.msm_bridge import compute_macro_populations as _compute_macro_populations
from .states.msm_bridge import lump_micro_to_macro_T as _lump_micro_to_macro_T
from .states.msm_bridge import pcca_like_macrostates as _pcca_like
from .states.picker import pick_frames_around_minima as _pick_frames_around_minima
from .utils.msm_utils import candidate_lag_ladder

logger = logging.getLogger("pmarlo")


def _align_trajectory(
    traj: md.Trajectory,
    atom_selection: str | Sequence[int] | None = "name CA",
) -> md.Trajectory:
    """Return an aligned copy of the trajectory using the provided atom selection.

    For invariance across frames, we superpose all frames to the first frame
    on C-alpha atoms by default. If the selection fails, the input trajectory
    is returned unchanged.
    """
    try:
        top = traj.topology
        if isinstance(atom_selection, str):
            atom_indices = top.select(atom_selection)
        elif atom_selection is None:
            atom_indices = top.select("name CA")
        else:
            atom_indices = list(atom_selection)
        if atom_indices is None or len(atom_indices) == 0:
            return traj
        ref = traj[0]
        aligned = traj.superpose(ref, atom_indices=atom_indices)
        return aligned
    except Exception:
        return traj


def _trig_expand_periodic(
    X: np.ndarray, periodic: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Expand periodic columns of ``X`` into cos/sin pairs.

    Parameters
    ----------
    X:
        Feature matrix of shape ``(n_frames, n_features)``.
    periodic:
        Boolean array indicating which columns of ``X`` are periodic.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A pair ``(Xe, mapping)`` where ``Xe`` is the expanded feature matrix
        and ``mapping`` is an integer array such that ``mapping[k]`` gives the
        original column index of ``Xe[:, k]``.  Non-periodic columns map 1:1,
        while periodic columns appear twice in ``Xe`` (cos and sin) and thus
        duplicate their original index in ``mapping``.
    """

    if X.size == 0:
        return X, np.array([], dtype=int)
    if periodic.size != X.shape[1]:
        # Best-effort: assume non-periodic if mismatch
        periodic = np.zeros((X.shape[1],), dtype=bool)

    cols: List[np.ndarray] = []
    mapping: List[int] = []
    for j in range(X.shape[1]):
        col = X[:, j]
        if bool(periodic[j]):
            cols.append(np.cos(col))
            cols.append(np.sin(col))
            mapping.extend([j, j])
        else:
            cols.append(col)
            mapping.append(j)

    Xe = np.vstack(cols).T if cols else X
    return Xe, np.asarray(mapping, dtype=int)


def compute_universal_metric(
    traj: md.Trajectory,
    feature_specs: Optional[Sequence[str]] = None,
    align: bool = True,
    atom_selection: str | Sequence[int] | None = "name CA",
    method: Literal["vamp", "tica", "pca"] = "vamp",
    lag: int = 10,
    *,
    cache_path: Optional[str] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Compute a universal 1D metric from multiple CVs with alignment and reduction.

    Steps:
    - Optional superposition of trajectory frames (default: C-alpha atoms)
    - Compute a broad set of default features if none are specified
      (phi/psi, chi1, Rg, SASA, HBond count, secondary-structure fractions)
    - Trig-expand periodic columns to handle angular wrap-around
    - Reduce to a single component via VAMP/TICA/PCA

    Returns the 1D metric array (n_frames,) and metadata.
    """
    logger.info(
        "[universal] Starting computation (align=%s, method=%s, lag=%s)",
        bool(align),
        method,
        int(lag),
    )
    traj_in = _align_trajectory(traj, atom_selection=atom_selection) if align else traj
    if align:
        try:
            logger.info("[universal] Alignment complete: %d frames", traj_in.n_frames)
        except Exception:
            logger.info("[universal] Alignment complete")
    specs = (
        list(feature_specs)
        if feature_specs is not None
        else ["phi_psi", "chi1", "Rg", "sasa", "hbonds_count", "ssfrac"]
    )
    logger.info("[universal] Computing features: %s", ", ".join(specs))
    X, cols, periodic = compute_features(
        traj_in, feature_specs=specs, cache_path=cache_path
    )
    logger.info(
        "[universal] Features computed: shape=%s, columns=%d", tuple(X.shape), len(cols)
    )
    if X.size == 0:
        return np.zeros((traj.n_frames,), dtype=float), {
            "columns": cols,
            "periodic": periodic,
            "reduction": method,
            "lag": int(lag),
            "aligned": bool(align),
            "specs": specs,
        }
    logger.info("[universal] Trig-expanding periodic columns")
    Xe, index_map = _trig_expand_periodic(X, periodic)
    logger.info("[universal] Expanded shape=%s", tuple(Xe.shape))
    if method == "pca":
        logger.info("[universal] Reducing with PCA → 1D")
        Y = pca_reduce(Xe, n_components=1)
    elif method == "tica":
        logger.info("[universal] Reducing with TICA(lag=%d) → 1D", int(max(1, lag)))
        Y = tica_reduce(Xe, lag=int(max(1, lag)), n_components=1)
    else:
        # VAMP default
        logger.info("[universal] Reducing with VAMP(lag=%d) → 1D", int(max(1, lag)))
        Y = vamp_reduce(Xe, lag=int(max(1, lag)), n_components=1, score_dims=[1])
    metric = Y.reshape(-1)
    logger.info("[universal] Metric ready: %d frames", metric.shape[0])
    meta: Dict[str, Any] = {
        "columns": cols,
        "periodic": periodic,
        "reduction": method,
        "lag": int(lag),
        "aligned": bool(align),
        "specs": specs,
        "index_map": index_map,
    }
    return metric, meta


def compute_universal_embedding(
    traj: md.Trajectory,
    feature_specs: Optional[Sequence[str]] = None,
    align: bool = True,
    atom_selection: str | Sequence[int] | None = "name CA",
    method: Literal["vamp", "tica", "pca"] = "vamp",
    lag: int = 10,
    n_components: int = 2,
    *,
    cache_path: Optional[str] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Compute a universal low-dimensional embedding (≥1D) from many CVs.

    Returns array of shape (n_frames, n_components) and metadata.
    """
    specs = (
        list(feature_specs)
        if feature_specs is not None
        else ["phi_psi", "chi1", "Rg", "sasa", "hbonds_count", "ssfrac"]
    )
    traj_in = _align_trajectory(traj, atom_selection=atom_selection) if align else traj
    X, cols, periodic = compute_features(
        traj_in, feature_specs=specs, cache_path=cache_path
    )
    Xe, index_map = _trig_expand_periodic(X, periodic)
    k = int(max(1, n_components))
    if method == "pca":
        Y = pca_reduce(Xe, n_components=k)
    elif method == "tica":
        Y = tica_reduce(Xe, lag=int(max(1, lag)), n_components=k)
    else:
        Y = vamp_reduce(Xe, lag=int(max(1, lag)), n_components=k)
    meta: Dict[str, Any] = {
        "columns": cols,
        "periodic": periodic,
        "reduction": method,
        "lag": int(lag),
        "aligned": bool(align),
        "specs": specs,
        "n_components": k,
        "index_map": index_map,
    }
    return Y, meta


# ------------------------------ Feature helpers (refactor) ------------------------------


def _init_feature_accumulators() -> (
    tuple[List[str], List[np.ndarray], List[np.ndarray]]
):
    columns: List[str] = []
    feats: List[np.ndarray] = []
    periodic_flags: List[np.ndarray] = []
    return columns, feats, periodic_flags


def _parse_spec(spec: str) -> tuple[str, Dict[str, Any]]:
    feat_name, kwargs = parse_feature_spec(spec)
    return feat_name, kwargs


def _compute_feature_block(
    traj: md.Trajectory, feat_name: str, kwargs: Dict[str, Any]
) -> tuple[Any, np.ndarray]:
    fc = get_feature(feat_name)
    X = fc.compute(traj, **kwargs)
    return fc, X


def _log_feature_progress(feat_name: str, X: np.ndarray) -> None:
    try:
        logger.info("[features] %-14s → shape=%s", feat_name, tuple(X.shape))
    except Exception:
        logger.info("[features] %s computed", feat_name)


def _feature_labels(
    fc: Any, feat_name: str, n_cols: int, kwargs: Dict[str, Any]
) -> List[str]:
    labels = getattr(fc, "labels", None)
    if isinstance(labels, list) and len(labels) == n_cols:
        return list(labels)
    if feat_name == "phi_psi" and n_cols > 0:
        half = max(0, n_cols // 2)
        return [f"phi_{i}" for i in range(half)] + [
            f"psi_{i}" for i in range(n_cols - half)
        ]
    label_base = feat_name
    if feat_name == "distance_pair" and "i" in kwargs and "j" in kwargs:
        label_base = f"dist:atoms:{kwargs['i']}-{kwargs['j']}"
    return [f"{label_base}_{i}" if n_cols > 1 else label_base for i in range(n_cols)]


def _append_feature_outputs(
    feats: List[np.ndarray],
    periodic_flags: List[np.ndarray],
    columns: List[str],
    fc: Any,
    X: np.ndarray,
    feat_name: str,
    kwargs: Dict[str, Any],
) -> None:
    if X.size == 0:
        return
    feats.append(X)
    n_cols = X.shape[1]
    columns.extend(_feature_labels(fc, feat_name, n_cols, kwargs))
    periodic_flags.append(fc.is_periodic())


def _frame_mismatch_info(feats: List[np.ndarray]) -> tuple[int, bool, List[int]]:
    lengths = [int(f.shape[0]) for f in feats]
    min_frames = min(lengths) if lengths else 0
    mismatch = any(length != min_frames for length in lengths)
    return min_frames, mismatch, lengths


def _truncate_to_min_frames(
    feats: List[np.ndarray], min_frames: int
) -> List[np.ndarray]:
    return [f[:min_frames] for f in feats]


def _stack_and_build_periodic(
    feats: List[np.ndarray], periodic_flags: List[np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    X_all = np.hstack(feats)
    if periodic_flags:
        periodic = np.concatenate(periodic_flags)
    else:
        periodic = np.zeros((X_all.shape[1],), dtype=bool)
    return X_all, periodic


def _empty_feature_matrix(traj: md.Trajectory) -> tuple[np.ndarray, np.ndarray]:
    return np.zeros((traj.n_frames, 0), dtype=float), np.zeros((0,), dtype=bool)


def _resolve_cache_file(
    traj: md.Trajectory, feature_specs: Sequence[str], cache_path: Optional[str]
) -> Optional[Path]:
    if not cache_path:
        return None
    try:
        import hashlib as _hashlib
        import json as _json
        from pathlib import Path as _Path

        p = _Path(cache_path)
        p.mkdir(parents=True, exist_ok=True)
        meta: Dict[str, Any] = {
            "n_frames": int(getattr(traj, "n_frames", 0) or 0),
            "n_atoms": int(getattr(traj, "n_atoms", 0) or 0),
            "specs": list(feature_specs),
            "top_hash": None,
            "pos_hash": None,
        }
        try:
            top = getattr(traj, "topology", None)
            if top is not None:
                # Build a light-weight hash from atom/residue counts and names
                atoms = [a.name for a in top.atoms]
                residues = [r.name for r in top.residues]
                chains = [c.index for c in top.chains]
                meta["top_hash"] = _hashlib.sha1(
                    _json.dumps(
                        [
                            len(atoms),
                            len(residues),
                            len(chains),
                            atoms[:50],
                            residues[:50],
                        ],
                        separators=(",", ":"),
                    ).encode()
                ).hexdigest()
            # Include a small digest of coordinates to prevent stale cache
            try:
                xyz = getattr(traj, "xyz", None)
                if xyz is not None and xyz.size:
                    nf = int(min(getattr(traj, "n_frames", 0) or 0, 10)) or 1
                    na = int(min(getattr(traj, "n_atoms", 0) or 0, 50)) or 1
                    step = max(1, (getattr(traj, "n_frames", 1) or 1) // nf)
                    sample = xyz[::step, :na, :].astype("float32")
                    # Quantize for stability
                    sample_q = (sample * 1000.0).round().astype("int32")
                    meta["pos_hash"] = _hashlib.sha1(sample_q.tobytes()).hexdigest()
            except Exception:
                pass
        except Exception:
            pass
        key = _hashlib.sha1(
            _json.dumps(meta, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return p / f"features_{key}.npz"
    except Exception:
        return None


def _try_load_cached_features(
    cache_file: Path,
) -> Optional[tuple[np.ndarray, List[str], np.ndarray]]:
    try:
        data = np.load(cache_file)
        X_cached = data["X"]
        cols_cached = list(data["columns"].astype(str).tolist())
        periodic_cached = data["periodic"]
        try:
            logger.info(
                "[features] Loaded from cache %s: shape=%s, columns=%d",
                str(cache_file),
                tuple(X_cached.shape),
                len(cols_cached),
            )
        except Exception:
            pass
        return X_cached, cols_cached, periodic_cached
    except Exception:
        return None


def _compute_features_without_cache(
    traj: md.Trajectory, feature_specs: Sequence[str]
) -> tuple[np.ndarray, List[str], np.ndarray]:
    columns, feats, periodic_flags = _init_feature_accumulators()
    for spec in feature_specs:
        feat_name, kwargs = _parse_spec(spec)
        fc, X = _compute_feature_block(traj, feat_name, kwargs)
        _log_feature_progress(feat_name, X)
        _append_feature_outputs(
            feats, periodic_flags, columns, fc, X, feat_name, kwargs
        )
    if feats:
        min_frames, mismatch, lengths = _frame_mismatch_info(feats)
        if mismatch:
            logger.warning(
                "[features] Frame count mismatch across features: %s → truncating to %d",
                lengths,
                min_frames,
            )
        feats = _truncate_to_min_frames(feats, min_frames)
        X_all, periodic = _stack_and_build_periodic(feats, periodic_flags)
    else:
        X_all, periodic = _empty_feature_matrix(traj)
    return X_all, columns, periodic


def _maybe_save_cached_features(
    cache_file: Optional[Path],
    X_all: np.ndarray,
    columns: List[str],
    periodic: np.ndarray,
) -> None:
    if cache_file is None:
        return
    try:
        np.savez_compressed(
            cache_file,
            X=X_all,
            columns=np.array(columns, dtype=np.str_),
            periodic=periodic,
        )
    except Exception:
        pass


def compute_features(
    traj: md.Trajectory, feature_specs: Sequence[str], cache_path: Optional[str] = None
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """Compute features for the given trajectory.

    Returns (X, columns, periodic). If cache_path is provided, features will
    be loaded/saved using a hash of inputs to avoid redundant computation.
    """
    cache_file = _resolve_cache_file(traj, feature_specs, cache_path)
    if cache_file is not None and cache_file.exists():
        cached = _try_load_cached_features(cache_file)
        if cached is not None:
            return cached

    X_all, columns, periodic = _compute_features_without_cache(traj, feature_specs)
    _maybe_save_cached_features(cache_file, X_all, columns, periodic)
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
    smooth: bool = False,
    inpaint: bool = False,
    min_count: int = 1,
    kde_bw_deg: Tuple[float, float] = (20.0, 20.0),
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
    smooth
        If ``True``, smooth the density with a periodic KDE.
    inpaint
        If ``True``, fill empty bins using the KDE estimate.
    min_count
        Histogram bins with fewer samples are marked as empty unless ``inpaint``
        is ``True``.
    kde_bw_deg
        Bandwidth in degrees for the periodic KDE when smoothing or inpainting.

    Returns
    -------
    FESResult
        Dataclass containing the free-energy surface and bin edges.
    """

    out = _generate_2d_fes(
        cv1,
        cv2,
        bins=bins,
        temperature=temperature,
        periodic=periodic,
        smooth=smooth,
        inpaint=inpaint,
        min_count=min_count,
        kde_bw_deg=kde_bw_deg,
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
    """Return indices of the highest-variance CV columns.

    Constant (zero-variance) columns are ignored. If fewer than two
    non-constant columns remain, the lone surviving index is paired with
    itself. ``None`` is returned when ``X`` has no columns.
    """

    if X.shape[1] < 1:
        return None
    variances = np.var(X, axis=0)
    non_const = np.where(variances > 0)[0]
    if non_const.size == 0:
        return None
    order = non_const[np.argsort(variances[non_const])[::-1]]
    if order.size == 1:
        idx = int(order[0])
        return idx, idx
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
    pair_phi_psi = _fes_pair_from_phi_psi_maps(cols)
    if pair_phi_psi is not None:
        i, j, rid = pair_phi_psi
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
        if X.shape[1] > 0:
            # Fold: use first axis for both coordinates
            pi, pj = _fes_periodic_pair_flags(periodic, 0, 0)
            return 0, 0, pi, pj

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
    smooth: bool = True,
    min_count: int = 1,
    kde_bw_deg: Tuple[float, float] = (20.0, 20.0),
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
        smooth=smooth,
        min_count=min_count,
        kde_bw_deg=kde_bw_deg,
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
    **kwargs: Any,
) -> Tuple[List[str], List[float]]:
    """Run REMD and return (trajectory_files, analysis_temperatures).

    Attempts demultiplexing to ~300 K; falls back to per-replica trajectories.
    """
    remd_out = Path(output_dir) / "replica_exchange"

    equil = min(total_steps // 10, 200 if total_steps <= 2000 else 2000)
    dcd_stride = max(1, int(total_steps // 5000))
    exchange_frequency = max(100, total_steps // 20)
    # Optional quick-run preset for interactive demos
    if bool(kwargs.get("quick", False)):
        equil = min(total_steps // 20, 100)
        exchange_frequency = max(50, total_steps // 40)
        dcd_stride = max(1, int(total_steps // 1000))

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
    # Optional unified progress callback (many alias names accepted)
    cb = coerce_progress_callback(kwargs)
    remd.run_simulation(
        total_steps=int(total_steps),
        equilibration_steps=int(equil),
        progress_callback=cb,
        cancel_token=kwargs.get("cancel_token"),
    )

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


def analyze_msm(  # noqa: C901
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

    # Plot FES/PMF based on feature_type
    if feature_type.lower().startswith("universal"):
        try:
            # Build one universal embedding and reuse for PMF(1D) and FES(2D)
            traj_all = None
            for t in msm.trajectories:
                traj_all = t if traj_all is None else traj_all.join(t)
            if traj_all is not None:
                # Choose method with Literal-typed variable for mypy
                if "vamp" in feature_type.lower():
                    red_method: Literal["vamp", "tica", "pca"] = "vamp"
                elif "tica" in feature_type.lower():
                    red_method = "tica"
                else:
                    red_method = "pca"
                # Reuse cached features for the concatenated trajectory as well
                from pathlib import Path as _Path

                cache_dir = _Path(str(msm.output_dir)) / "feature_cache"
                cache_dir.mkdir(parents=True, exist_ok=True)
                Y2, _ = compute_universal_embedding(
                    traj_all,
                    feature_specs=None,
                    align=True,
                    method=red_method,
                    lag=int(max(1, msm.lag_time or 10)),
                    n_components=2,
                    cache_path=str(cache_dir),
                )
                # 1) PMF on IC1
                from .fes.surfaces import generate_1d_pmf

                pmf = generate_1d_pmf(
                    Y2[:, 0], bins=int(max(30, adaptive_bins)), temperature=300.0
                )
                _ = save_pmf_line(
                    pmf.F,
                    pmf.edges,
                    xlabel="universal IC1",
                    output_dir=str(msm.output_dir),
                    filename="pmf_universal_ic1.png",
                )
                # 2) 2D FES on (IC1, IC2)
                fes2 = generate_free_energy_surface(
                    Y2[:, 0],
                    Y2[:, 1],
                    bins=(int(adaptive_bins), int(adaptive_bins)),
                    temperature=300.0,
                    periodic=(False, False),
                    smooth=True,
                    min_count=1,
                )
                _ = save_fes_contour(
                    fes2.F,
                    fes2.xedges,
                    fes2.yedges,
                    "universal IC1",
                    "universal IC2",
                    str(msm.output_dir),
                    "fes_universal_ic1_vs_ic2.png",
                )
        except Exception:
            pass
    else:
        # Disable phi/psi-specific FES in analyze_msm default path
        pass
    msm.plot_implied_timescales(save_file="implied_timescales")
    msm.plot_free_energy_profile(save_file="free_energy_profile")
    msm.create_state_table()
    msm.extract_representative_structures(save_pdb=True)
    msm.save_analysis_results()

    return msm_out


def find_conformations(  # noqa: C901
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

    loaded_frames = 0
    for chunk in traj_io.iterload(
        str(trajectory_choice),
        top=str(topology_pdb),
        stride=traj_stride,
        atom_indices=atom_indices,
        chunk=chunk_size,
    ):
        traj = chunk if traj is None else traj.join(chunk)
        loaded_frames += int(chunk.n_frames)
        if loaded_frames % max(1, chunk_size) == 0:
            logger.info("[stream] Loaded %d frames so far...", loaded_frames)
    if traj is None:
        raise ValueError("No frames loaded from trajectory")

    specs = feature_specs if feature_specs is not None else ["phi_psi"]
    # Use on-disk cache to avoid recomputing expensive CVs
    from pathlib import Path as _Path

    cache_dir = _Path(str(out)) / "feature_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    X, cols, periodic = compute_features(
        traj, feature_specs=specs, cache_path=str(cache_dir)
    )
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
    try:
        fes_info = generate_fes_and_pick_minima(
            X,
            cols,
            periodic,
            requested_pair=requested_pair,
            bins=(adaptive_bins, adaptive_bins),
            temperature=300.0,
            smooth=True,
            min_count=1,
            kde_bw_deg=(20.0, 20.0),
            deltaF_kJmol=3.0,
        )
    except RuntimeError as e:
        # Gracefully skip when selected pair is identical or unsuitable
        logger.warning("Skipping FES minima picking: %s", e)
        fes_info = {"names": ("N/A", "N/A"), "fes": None, "minima": {"minima": []}}
    names = fes_info["names"]
    fes = fes_info["fes"]
    minima = fes_info["minima"]
    fname = f"fes_{sanitize_label_for_filename(names[0])}_vs_{sanitize_label_for_filename(names[1])}.png"
    if fes is not None:
        _ = save_fes_contour(
            fes.F,
            fes.xedges,
            fes.yedges,
            names[0],
            names[1],
            str(out),
            fname,
            mask=fes.metadata.get("mask"),
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


def find_conformations_with_msm(
    topology_pdb: str | Path,
    trajectory_file: str | Path,
    output_dir: str | Path,
    feature_specs: Optional[List[str]] = None,
    requested_pair: Optional[Tuple[str, str]] = None,
    traj_stride: int = 1,
    atom_selection: str | Sequence[int] | None = None,
    chunk_size: int = 1000,
) -> Path:
    """One-line convenience wrapper to find representative conformations.

    This is a thin alias around :func:`find_conformations` to mirror the
    example program name and make the public API more discoverable.
    """
    return find_conformations(
        topology_pdb=topology_pdb,
        trajectory_choice=trajectory_file,
        output_dir=output_dir,
        feature_specs=feature_specs,
        requested_pair=requested_pair,
        traj_stride=traj_stride,
        atom_selection=atom_selection,
        chunk_size=chunk_size,
    )
