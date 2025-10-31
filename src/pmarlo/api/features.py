import logging
import hashlib
import json
import numpy as np
import mdtraj as md

from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple
from pathlib import Path

# Import reduction methods directly
from pmarlo.markov_state_model.reduction import (
    pca_reduce,
    tica_reduce,
    vamp_reduce,
)
from pmarlo.utils.path_utils import ensure_directory

from pmarlo.features import get_feature
from pmarlo.features.base import parse_feature_spec


logger = logging.getLogger("pmarlo")

def _resolve_cache_file(
    traj: md.Trajectory, feature_specs: Sequence[str], cache_path: Optional[str]
) -> Optional[Path]:
    if not cache_path:
        return None

    p = Path(cache_path)
    ensure_directory(p)
    meta: Dict[str, Any] = {
        "n_frames": int(traj.n_frames),
        "n_atoms": int(traj.n_atoms),
        "specs": list(feature_specs),
        "top_hash": None,
        "pos_hash": None,
    }

    top = traj.topology
    # Build a light-weight hash from atom/residue counts and names
    atoms = [a.name for a in top.atoms]
    residues = [r.name for r in top.residues]
    chains = [c.index for c in top.chains]
    meta["top_hash"] = hashlib.sha1(
        json.dumps(
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
    xyz = traj.xyz
    if xyz is not None and xyz.size:
        nf = int(min(traj.n_frames, 10))
        na = int(min(traj.n_atoms, 50))
        step = max(1, traj.n_frames // nf)
        sample = xyz[::step, :na, :].astype("float32")
        # Quantize for stability
        sample_q = (sample * 1000.0).round().astype("int32")
        meta["pos_hash"] = hashlib.sha1(sample_q.tobytes()).hexdigest()

    key = hashlib.sha1(
        json.dumps(meta, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return p / f"features_{key}.npz"

def _try_load_cached_features(
    cache_file: Path,
) -> Optional[tuple[np.ndarray, List[str], np.ndarray]]:
    data = np.load(cache_file)
    X_cached = data["X"]
    cols_cached = list(data["columns"].astype(str).tolist())
    periodic_cached = data["periodic"]
    logger.info(
        "[features] Loaded from cache %s: shape=%s, columns=%d",
        str(cache_file),
        tuple(X_cached.shape),
        len(cols_cached),
    )
    return X_cached, cols_cached, periodic_cached


def _maybe_save_cached_features(
    cache_file: Optional[Path],
    X_all: np.ndarray,
    columns: List[str],
    periodic: np.ndarray,
) -> None:
    if cache_file is None:
        return
    np.savez_compressed(
        cache_file,
        X=X_all,
        columns=np.array(columns, dtype=np.str_),
        periodic=periodic,
    )


def align_trajectory(
    traj: md.Trajectory,
    atom_selection: str | Sequence[int] | None = "name CA",
) -> md.Trajectory:
    """Return an aligned copy of the trajectory using the provided atom selection.

    For invariance across frames, we superpose all frames to the first frame
    on C-alpha atoms by default. Failures to determine the atom selection or to
    perform the alignment raise errors instead of silently returning the
    unaligned trajectory so that issues can be detected early in the pipeline.
    """
    top = traj.topology
    if isinstance(atom_selection, str):
        atom_indices = top.select(atom_selection)
    elif atom_selection is None:
        atom_indices = top.select("name CA")
    else:
        atom_indices = list(atom_selection)

    if atom_indices is None or len(atom_indices) == 0:
        raise ValueError(
            "No atoms were selected for trajectory alignment; check the atom selection."
        )

    ref = traj[0]
    return traj.superpose(ref, atom_indices=atom_indices)


def trig_expand_periodic(
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
        raise ValueError(
            f"periodic array size ({periodic.size}) must match number of features ({X.shape[1]})"
        )

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

def _init_feature_accumulators() -> (
    tuple[List[str], List[np.ndarray], List[np.ndarray]]
):
    columns: List[str] = []
    feats: List[np.ndarray] = []
    periodic_flags: List[np.ndarray] = []
    return columns, feats, periodic_flags


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
    logger.info("[features] %-14s → shape=%s", feat_name, tuple(X.shape))

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
    return np.empty((traj.n_frames, 0), dtype=float), np.empty((0,), dtype=bool)


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
    traj_in = align_trajectory(traj, atom_selection=atom_selection) if align else traj
    if align:
        logger.info("[universal] Alignment complete: %d frames", traj_in.n_frames)
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
    Xe, index_map = trig_expand_periodic(X, periodic)
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
        Y = vamp_reduce(Xe, lag=int(max(1, lag)), n_components=1)
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
    traj_in = align_trajectory(traj, atom_selection=atom_selection) if align else traj
    X, cols, periodic = compute_features(
        traj_in, feature_specs=specs, cache_path=cache_path
    )
    Xe, index_map = trig_expand_periodic(X, periodic)
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
        # Use VAMP reduction with specified components
        return vamp_reduce(X, lag=lag, n_components=n_components)
    raise ValueError(f"Unknown reduction method: {method}")

def _fes_build_phi_psi_maps(
    cols: Sequence[str],
) -> tuple[dict[int, int], dict[int, int]]:
    phi_map_local: dict[int, int] = {}
    psi_map_local: dict[int, int] = {}
    for k, c in enumerate(cols):
        if c.startswith("phi:res"):
            rid = int(c.split("res")[-1])
            phi_map_local[rid] = k
        if c.startswith("psi:res"):
            rid = int(c.split("res")[-1])
            psi_map_local[rid] = k
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
