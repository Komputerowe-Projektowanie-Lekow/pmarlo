from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import mdtraj as md  # type: ignore
import numpy as np

from .cluster.micro import cluster_microstates as _cluster_microstates
from .features import get_feature
from .features.base import parse_feature_spec
from .fes.surfaces import generate_2d_fes as _generate_2d_fes
from .reduce.reducers import pca_reduce, tica_reduce
from .states.msm_bridge import build_simple_msm as _build_simple_msm
from .states.msm_bridge import compute_macro_mfpt as _compute_macro_mfpt
from .states.msm_bridge import compute_macro_populations as _compute_macro_populations
from .states.msm_bridge import lump_micro_to_macro_T as _lump_micro_to_macro_T
from .states.msm_bridge import pcca_like_macrostates as _pcca_like
from .states.picker import pick_frames_around_minima as _pick_frames_around_minima


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
    method: Literal["pca", "tica"] = "tica",
    lag: int = 10,
    n_components: int = 2,
) -> np.ndarray:
    if method == "pca":
        return pca_reduce(X, n_components=n_components)
    if method == "tica":
        return tica_reduce(X, lag=lag, n_components=n_components)
    raise ValueError(f"Unknown reduction method: {method}")


def cluster_microstates(
    Y: np.ndarray,
    method: Literal["minibatchkmeans", "kmeans"] = "minibatchkmeans",
    **kwargs,
) -> np.ndarray:
    return _cluster_microstates(Y, method=method, **kwargs)


def generate_free_energy_surface(
    cv1: np.ndarray,
    cv2: np.ndarray,
    bins: Tuple[int, int] = (100, 100),
    temperature: float = 300.0,
    periodic: Tuple[bool, bool] = (False, False),
    smoothing: Optional[Literal["cosine"]] = None,
) -> dict:
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


def _fes_pair_from_phi_psi_maps(cols: Sequence[str]) -> Tuple[int, int] | None:
    phi_map_local, psi_map_local = _fes_build_phi_psi_maps(cols)
    common_residues = sorted(set(phi_map_local).intersection(psi_map_local))
    if not common_residues:
        return None
    rid0 = common_residues[0]
    return phi_map_local[rid0], psi_map_local[rid0]


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
        i, j = pair
        pi, pj = _fes_periodic_pair_flags(periodic, i, j)
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
        cv1, cv2, fes["F"], fes["xedges"], fes["yedges"], deltaF_kJmol=deltaF_kJmol
    )
    return {
        "i": int(i),
        "j": int(j),
        "names": (name_i, name_j),
        "periodic_flags": (bool(per_i), bool(per_j)),
        "fes": fes,
        "minima": minima,
    }
