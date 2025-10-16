from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import mdtraj as md  # type: ignore
import numpy as np

from pmarlo import constants as const

if TYPE_CHECKING:
    from .io.trajectory_writer import MDTrajDCDWriter

from .config import JOINT_USE_REWEIGHT
from .data.aggregate import aggregate_and_build as _aggregate_and_build
from .features import get_feature
from .features.base import parse_feature_spec
from .markov_state_model._msm_utils import build_simple_msm as _build_simple_msm
from .markov_state_model._msm_utils import (
    candidate_lag_ladder,
)
from .markov_state_model._msm_utils import compute_macro_mfpt as _compute_macro_mfpt
from .markov_state_model._msm_utils import (
    compute_macro_populations as _compute_macro_populations,
)
from .markov_state_model._msm_utils import (
    lump_micro_to_macro_T as _lump_micro_to_macro_T,
)
from .markov_state_model._msm_utils import pcca_like_macrostates as _pcca_like
from .shards.indexing import initialise_shard_indices

_run_ck: Any = None
try:  # pragma: no cover - optional plotting dependency
    from .markov_state_model.ck_runner import (
        run_ck as _run_ck,  # type: ignore[no-redef]
    )
except Exception:  # pragma: no cover - executed without matplotlib
    pass

try:  # pragma: no cover - optional sklearn dependency
    from .markov_state_model.clustering import (
        cluster_microstates as _cluster_microstates,
    )
except Exception:  # pragma: no cover - executed without sklearn

    def _cluster_microstates(*_args: object, **_kwargs: object):  # type: ignore
        raise ImportError(
            "cluster_microstates requires scikit-learn. Install with `pip install 'pmarlo[analysis]'`."
        )


try:  # pragma: no cover - optional ML stack
    from .markov_state_model.enhanced_msm import EnhancedMSM as _EnhancedMSM

    MarkovStateModel: type[Any] = _EnhancedMSM
except Exception:  # pragma: no cover - executed without sklearn/torch

    class _MarkovStateModelStub:  # type: ignore[misc]
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise ImportError(
                "EnhancedMSM requires optional dependencies. Install with `pip install 'pmarlo[analysis]'`."
            )

    MarkovStateModel = _MarkovStateModelStub

try:  # pragma: no cover - optional plotting dependency
    from .markov_state_model.free_energy import FESResult
    from .markov_state_model.free_energy import generate_2d_fes as _generate_2d_fes
except Exception:  # pragma: no cover - executed without analysis extras
    FESResult = Any  # type: ignore

    def _generate_2d_fes(*_args: object, **_kwargs: object) -> Any:  # type: ignore
        raise ImportError("generate_2d_fes requires optional analysis dependencies.")


try:  # pragma: no cover - optional matplotlib dependency
    from .markov_state_model.picker import (
        pick_frames_around_minima as _pick_frames_around_minima_impl,
    )

    _pick_frames_around_minima = _pick_frames_around_minima_impl
except Exception:  # pragma: no cover - executed without plotting

    def _pick_frames_around_minima_stub(
        *_args: object, **_kwargs: object
    ) -> dict[str, Any]:
        raise ImportError(
            "pick_frames_around_minima requires optional plotting dependencies."
        )

    _pick_frames_around_minima = _pick_frames_around_minima_stub

try:  # pragma: no cover - optional sklearn dependency
    from .markov_state_model.reduction import pca_reduce, tica_reduce, vamp_reduce
except Exception:  # pragma: no cover - executed without sklearn

    def _missing_reduction(*_args: object, **_kwargs: object) -> np.ndarray:
        raise ImportError(
            "Dimensionality reduction requires scikit-learn. Install with `pip install 'pmarlo[analysis]'`."
        )

    pca_reduce = tica_reduce = vamp_reduce = _missing_reduction  # type: ignore

try:  # pragma: no cover - optional OpenMM dependency
    from .replica_exchange.config import RemdConfig
    from .replica_exchange.replica_exchange import ReplicaExchange as _ReplicaExchange

    ReplicaExchange: type[Any] = _ReplicaExchange
except Exception:  # pragma: no cover - executed without OpenMM stack
    RemdConfig = Any  # type: ignore

    class _ReplicaExchangeStub:  # type: ignore[misc]
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise ImportError("ReplicaExchange requires OpenMM and optional extras.")

    ReplicaExchange = _ReplicaExchangeStub

try:  # pragma: no cover - optional pandas/matplotlib dependency
    from .reporting.export import (
        write_conformations_csv_json as _write_conformations_csv_json,
    )

    write_conformations_csv_json = _write_conformations_csv_json
except Exception:  # pragma: no cover - executed without reporting extras

    def _write_conformations_csv_json_stub(*_args: object, **_kwargs: object) -> None:
        raise ImportError("Reporting export helpers require optional dependencies.")

    write_conformations_csv_json = _write_conformations_csv_json_stub

try:  # pragma: no cover - optional matplotlib dependency
    from .reporting.plots import save_fes_contour as _save_fes_contour
    from .reporting.plots import save_pmf_line as _save_pmf_line
    from .reporting.plots import (
        save_transition_matrix_heatmap as _save_transition_matrix_heatmap,
    )

    save_fes_contour = _save_fes_contour
    save_pmf_line = _save_pmf_line
    save_transition_matrix_heatmap = _save_transition_matrix_heatmap
except Exception:  # pragma: no cover - executed without plotting

    def _save_fes_contour_stub(*_args: object, **_kwargs: object) -> None:
        raise ImportError("Plotting helpers require matplotlib.")

    save_fes_contour = _save_fes_contour_stub
    save_pmf_line = _save_fes_contour_stub
    save_transition_matrix_heatmap = _save_fes_contour_stub
try:  # pragma: no cover - optional workflow dependency chain
    from .transform.build import AppliedOpts as _AppliedOpts
    from .transform.build import BuildOpts as _BuildOpts
    from .transform.plan import TransformPlan as _TransformPlan
    from .transform.plan import TransformStep as _TransformStep
    from .transform.progress import (
        coerce_progress_callback as _coerce_progress_callback,
    )
    from .workflow.joint import JointWorkflow as _JointWorkflow
    from .workflow.joint import WorkflowConfig as _JointWorkflowConfig

    coerce_progress_callback = _coerce_progress_callback
    JointWorkflow: type[Any] = _JointWorkflow
    JointWorkflowConfig: type[Any] = _JointWorkflowConfig
except Exception:  # pragma: no cover - executed without transform extras
    _AppliedOpts = _BuildOpts = _TransformPlan = _TransformStep = None  # type: ignore

    def _coerce_progress_callback_stub(*_args: object, **_kwargs: object):  # type: ignore
        raise ImportError("Transform workflow requires optional dependencies.")

    class _JointWorkflowStub:  # type: ignore[misc]
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise ImportError("Joint workflow requires optional dependencies.")

    class _JointWorkflowConfigStub:  # type: ignore[misc]
        pass

    coerce_progress_callback = _coerce_progress_callback_stub
    JointWorkflow = _JointWorkflowStub
    JointWorkflowConfig = _JointWorkflowConfigStub

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
        # Use VAMP reduction with specified components
        return vamp_reduce(X, lag=lag, n_components=n_components)
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
    *,
    random_seed: int | None = None,
    random_state: int | None = None,
    start_from_checkpoint: str | Path | None = None,
    start_from_pdb: str | Path | None = None,
    jitter_start: bool = False,
    jitter_sigma_A: float = 0.05,
    velocity_reseed: bool = False,
    exchange_frequency_steps: int | None = None,
    save_state_frequency: int | None = None,
    temperature_schedule_mode: str | None = None,
    **kwargs: Any,
) -> Tuple[List[str], List[float]]:
    """Run REMD and return (trajectory_files, analysis_temperatures).

    Attempts demultiplexing to ~300 K; falls back to per-replica trajectories.
    When ``random_state`` or ``random_seed`` is provided, the seed is forwarded
    to the underlying :class:`ReplicaExchange` for deterministic behavior. If
    both are provided, ``random_state`` takes precedence for backward
    compatibility.
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
    # Allow explicit override from caller
    if exchange_frequency_steps is not None and int(exchange_frequency_steps) > 0:
        exchange_frequency = int(exchange_frequency_steps)

    # Prefer random_state when both provided for back-compat
    _seed = (
        int(random_state)
        if random_state is not None
        else (int(random_seed) if random_seed is not None else None)
    )
    remd = ReplicaExchange.from_config(
        RemdConfig(
            pdb_file=str(pdb_file),
            temperatures=temperatures,
            output_dir=str(remd_out),
            exchange_frequency=exchange_frequency,
            auto_setup=False,
            dcd_stride=dcd_stride,
            random_seed=_seed,
            start_from_checkpoint=(
                str(start_from_checkpoint) if start_from_checkpoint else None
            ),
            start_from_pdb=str(start_from_pdb) if start_from_pdb else None,
            jitter_sigma_A=float(jitter_sigma_A) if jitter_start else 0.0,
            reseed_velocities=bool(velocity_reseed),
            temperature_schedule_mode=temperature_schedule_mode,
        )
    )
    remd.plan_reporter_stride(
        total_steps=int(total_steps), equilibration_steps=int(equil), target_frames=5000
    )
    remd.setup_replicas()
    # Optional resume from checkpoint state
    if start_from_checkpoint:
        try:
            import pickle as _pkl

            with open(start_from_checkpoint, "rb") as fh:
                ckpt = _pkl.load(fh)
            remd.restore_from_checkpoint(ckpt)
            # Skip equilibration if resuming from a checkpoint
            equil = 0
        except Exception as exc:
            logger.warning(
                "Failed to restore from checkpoint %s: %s",
                str(start_from_checkpoint),
                exc,
            )
    # Optional unified progress callback (many alias names accepted)
    cb = coerce_progress_callback(kwargs)
    remd.run_simulation(
        total_steps=int(total_steps),
        equilibration_steps=int(equil),
        save_state_frequency=int(save_state_frequency or 10_000),
        progress_callback=cb,
        cancel_token=kwargs.get("cancel_token"),
    )

    # Demultiplex best-effort
    demuxed = remd.demux_trajectories(
        target_temperature=300.0, equilibration_steps=int(equil), progress_callback=cb
    )
    if demuxed:
        try:
            from pmarlo.io.trajectory_reader import MDTrajReader as _MDTReader

            nframes = _MDTReader(topology_path=str(pdb_file)).probe_length(str(demuxed))
            reporter_stride = getattr(remd, "reporter_stride", None)
            eff_stride = int(
                reporter_stride
                if reporter_stride
                else max(1, getattr(remd, "dcd_stride", 1))
            )
            production_steps = max(0, int(total_steps) - int(equil))
            expected = max(1, production_steps // eff_stride)
            # Accept demux if it's close to expected or at least a safe fraction.
            # Large runs may have minor segment repairs or stride mismatches.
            threshold = max(1, expected // 5)  # 20% of expected frames
            if int(nframes) >= threshold:
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
    # Configure MSM parameters
    if use_effective_for_uncertainty and hasattr(msm, "count_mode"):
        msm.count_mode = "sliding"  # type: ignore[attr-defined]

    # Load trajectories
    if hasattr(msm, "load_trajectories"):
        msm.load_trajectories(  # type: ignore[attr-defined]
            stride=traj_stride, atom_selection=atom_selection, chunk_size=chunk_size
        )

    # Compute features
    ft = feature_type
    if use_tica and ("tica" not in feature_type.lower()):
        ft = f"{feature_type}_tica"
    if hasattr(msm, "compute_features"):
        msm.compute_features(feature_type=ft)  # type: ignore[attr-defined]

    # Cluster
    N_CLUSTERS = 8
    if hasattr(msm, "cluster_features"):
        msm.cluster_features(n_states=int(N_CLUSTERS))  # type: ignore[attr-defined]

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
        total_frames = sum(
            getattr(t, "n_frames", 0) for t in getattr(msm, "trajectories", [])
        )
    except Exception:
        total_frames = 0
    max_lag = 250
    try:
        if total_frames > 0:
            max_lag = int(min(500, max(150, total_frames // 5)))
    except Exception:
        max_lag = 250
    candidate_lags = candidate_lag_ladder(min_lag=1, max_lag=max_lag)
    if hasattr(msm, "build_msm"):
        msm.build_msm(lag_time=5, method=method)  # type: ignore[attr-defined]
    if hasattr(msm, "compute_implied_timescales"):
        msm.compute_implied_timescales(lag_times=candidate_lags, n_timescales=3)  # type: ignore[attr-defined]

    chosen_lag = 10
    try:
        import numpy as _np  # type: ignore

        its_data = getattr(msm, "implied_timescales", None)
        if its_data is not None and hasattr(its_data, "__getitem__"):
            lags = _np.array(its_data["lag_times"])  # type: ignore[index]
            its = _np.array(its_data["timescales"])  # type: ignore[index]
        else:
            lags = _np.array(candidate_lags)
            its = _np.ones((len(candidate_lags), 3)) * 10.0
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

    if hasattr(msm, "build_msm"):
        msm.build_msm(lag_time=chosen_lag, method=method)  # type: ignore[attr-defined]

    # CK test with macro → micro fallback
    try:
        dtrajs = getattr(msm, "dtrajs", None)
        lag_time = getattr(msm, "lag_time", chosen_lag)
        output_dir = getattr(msm, "output_dir", output_dir)
        if _run_ck is not None and dtrajs is not None:
            _run_ck(dtrajs, lag_time, output_dir, macro_k=3)
    except Exception:
        pass

    try:
        total_frames_fes = sum(
            getattr(t, "n_frames", 0) for t in getattr(msm, "trajectories", [])
        )
    except Exception:
        total_frames_fes = 0
    adaptive_bins = max(20, min(50, int((total_frames_fes or 0) ** 0.5))) or 20

    # Plot FES/PMF based on feature_type
    if feature_type.lower().startswith("universal"):
        try:
            # Build one universal embedding and reuse for PMF(1D) and FES(2D)
            traj_all = None
            trajectories = getattr(msm, "trajectories", [])
            for t in trajectories:
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

                cache_dir = (
                    _Path(str(getattr(msm, "output_dir", output_dir))) / "feature_cache"
                )
                cache_dir.mkdir(parents=True, exist_ok=True)
                Y2, _ = compute_universal_embedding(
                    traj_all,
                    feature_specs=None,
                    align=True,
                    method=red_method,
                    lag=int(max(1, getattr(msm, "lag_time", None) or 10)),
                    n_components=2,
                    cache_path=str(cache_dir),
                )
                # 1) PMF on IC1
                from .markov_state_model.free_energy import generate_1d_pmf

                pmf = generate_1d_pmf(
                    Y2[:, 0], bins=int(max(30, adaptive_bins)), temperature=300.0
                )
                _ = save_pmf_line(
                    pmf.F,
                    pmf.edges,
                    xlabel="universal IC1",
                    output_dir=str(getattr(msm, "output_dir", output_dir)),
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
                    str(getattr(msm, "output_dir", output_dir)),
                    "fes_universal_ic1_vs_ic2.png",
                )
        except Exception:
            pass
    else:
        # Disable phi/psi-specific FES in analyze_msm default path
        pass
    # Generate plots and analysis results with attribute checks
    if hasattr(msm, "plot_implied_timescales"):
        msm.plot_implied_timescales(save_file="implied_timescales")  # type: ignore[attr-defined]
    if hasattr(msm, "plot_free_energy_profile"):
        msm.plot_free_energy_profile(save_file="free_energy_profile")  # type: ignore[attr-defined]
    if hasattr(msm, "create_state_table"):
        msm.create_state_table()  # type: ignore[attr-defined]
    if hasattr(msm, "extract_representative_structures"):
        msm.extract_representative_structures(save_pdb=True)  # type: ignore[attr-defined]
    if hasattr(msm, "save_analysis_results"):
        msm.save_analysis_results()  # type: ignore[attr-defined]

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


# ------------------------------ App-friendly wrappers ------------------------------


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

    import mdtraj as md  # type: ignore

    from pmarlo.data.shard import write_shard  # type: ignore
    from pmarlo.io import trajectory as _traj_io  # type: ignore
    from pmarlo.transform.progress import ProgressReporter  # type: ignore

    pdb_file = Path(pdb_file)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ref, ca_sel = _load_reference_and_selection(md, pdb_file, reference)
    shard_state = initialise_shard_indices(out_dir, seed_start)
    next_idx = shard_state.next_index
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

    for index, traj_path in enumerate(files):
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
            _traj_io.iterload,
        )
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
            provenance,
        )
        shard_paths.extend(window_paths)

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

    rg = np.concatenate(rg_parts) if rg_parts else np.zeros((0,), dtype=np.float64)
    rmsd = (
        np.concatenate(rmsd_parts) if rmsd_parts else np.zeros((0,), dtype=np.float64)
    )
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
    provenance: dict | None,
) -> tuple[list[Path], int]:
    """Write overlapping shards for the provided CV time-series."""

    shard_paths: list[Path] = []
    n_frames = int(rg.shape[0])
    if n_frames <= 0:
        return shard_paths, next_idx

    eff_window = min(window, n_frames)
    eff_hop = min(hop, eff_window)
    for start in range(0, n_frames - eff_window + 1, eff_hop):
        stop = start + eff_window
        shard_id = f"shard_{next_idx:04d}"
        cvs = {"Rg": rg[start:stop], "RMSD_ref": rmsd[start:stop]}
        source: dict[str, object] = {
            "traj": str(traj_path),
            "range": [int(start), int(stop)],
            "n_frames": int(stop - start),
        }
        if provenance:
            try:
                merged = dict(provenance)
                merged.update(source)
                source = merged
            except Exception:
                pass
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
    import mdtraj as md  # type: ignore

    # Quiet streaming iterator for reading DCDs without plugin chatter
    from pmarlo.io import trajectory as _traj_io

    pdb_file = Path(pdb_file)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    top0 = md.load(str(pdb_file))
    ref = (
        md.load(str(reference), top=str(pdb_file))[0]
        if reference is not None and Path(reference).exists()
        else top0[0]
    )
    ca_sel = top0.topology.select("name CA")
    ca_sel = ca_sel if ca_sel.size else None

    def _extract(traj_path: Path):
        rg_parts = []
        rmsd_parts = []
        n = 0
        for chunk in _traj_io.iterload(
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
        import numpy as _np

        rg = (
            _np.concatenate(rg_parts)
            if rg_parts
            else _np.zeros((0,), dtype=_np.float64)
        )
        rmsd = (
            _np.concatenate(rmsd_parts)
            if rmsd_parts
            else _np.zeros((0,), dtype=_np.float64)
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

    from .data.emit import emit_shards_from_trajectories as _emit

    return _emit(
        [Path(p) for p in traj_files],
        out_dir=out_dir,
        extract_cvs=_extract,
        seed_start=int(seed_start),
        temperature=float(temperature),
        periodic_by_cv={"Rg": False, "RMSD_ref": False},
        progress_callback=progress_callback,
    )


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
):
    """Aggregate shard JSONs and build a bundle with an app-friendly API.

    - Optional LEARN_CV(method="deeptica") is prepended to the plan when requested.
    - Adds SMOOTH_FES step to the plan by default.
    - Computes and records global bin edges into notes["cv_bin_edges"].
    - Optional ``kmeans_kwargs`` are forwarded to the clustering step to tune K-means.
    - Returns (BuildResult, dataset_hash).
    """
    import numpy as _np

    from .data.shard import read_shard as _read_shard

    shard_paths = _normalise_shard_inputs(shard_jsons)
    meta0, _, _ = _read_shard(shard_paths[0])
    cv_pair = _infer_cv_pair(meta0)
    edges = _compute_cv_edges(shard_paths, cv_pair, bins, _read_shard, _np)

    model_dir = _extract_model_dir(notes)
    plan = _build_transform_plan(learn_cv, deeptica_params, lag, model_dir)
    opts = _build_opts(seed, temperature, lag, kmeans_kwargs)
    all_notes = _merge_notes_with_edges(notes, edges)
    n_states = _determine_macrostates(n_macrostates, deeptica_params)
    applied = _AppliedOpts(
        bins=bins,
        lag=int(lag),
        macrostates=n_states,
        notes=all_notes,
    )

    br, ds_hash = _aggregate_and_build(
        shard_paths,
        opts=opts,
        plan=plan,
        applied=applied,
        out_bundle=Path(out_bundle),
        progress_callback=progress_callback,
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
) -> _BuildOpts:
    """Create BuildOpts with a simple lag candidate ladder."""

    return _BuildOpts(
        seed=int(seed),
        temperature=float(temperature),
        lag_candidates=(int(lag), int(2 * lag), int(3 * lag)),
        kmeans_kwargs=dict(kmeans_kwargs or {}),
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


def demultiplex_run(
    run_id: str,
    replica_traj_paths: list[str | Path],
    exchange_log_path: str | Path,
    topology_path: str | Path,
    ladder_K: list[float] | str,
    dt_ps: float,
    out_dir: str | Path,
    fmt: str = "dcd",
    chunk_size: int = 5000,
) -> list[str]:
    """Demultiplex a REMD run into per-temperature trajectories and manifests.

    .. deprecated:: 0.0.42
        This function is deprecated. Use :func:`pmarlo.demultiplexing.demux.demux_trajectories`
        or the streaming demux functions directly.

    Returns list of DemuxShard JSON paths.
    """
    import warnings

    warnings.warn(
        "demultiplex_run is deprecated; use pmarlo.demultiplexing.demux.demux_trajectories "
        "or streaming demux functions directly",
        DeprecationWarning,
        stacklevel=2,
    )

    from .io.trajectory_reader import MDTrajReader
    from .replica_exchange.demux_compat import (
        parse_exchange_log,
        parse_temperature_ladder,
    )

    out_dir_path, topo_path, replica_paths = _prepare_demux_paths(
        out_dir,
        topology_path,
        replica_traj_paths,
    )
    temperatures = _parse_temperature_ladder_safe(ladder_K, parse_temperature_ladder)
    exchange_records = _load_exchange_records_safe(
        exchange_log_path, parse_exchange_log
    )

    if not exchange_records:
        return []

    _validate_demux_inputs(temperatures, replica_paths, exchange_records)

    reader = MDTrajReader(topology_path=str(topo_path))
    replica_frames = _collect_replica_frames(reader, replica_paths)
    writers, dcd_paths = _open_demux_writers(
        out_dir_path,
        topo_path,
        temperatures,
        fmt,
    )

    try:
        segments_per_temp, dst_positions = _demux_exchange_segments(
            exchange_records,
            replica_frames,
            writers,
        )
    finally:
        _close_demux_writers(writers)

    return _write_demux_manifests(
        run_id,
        temperatures,
        dcd_paths,
        dst_positions,
        segments_per_temp,
        dt_ps,
        topology_path=topo_path,
    )


def _prepare_demux_paths(
    out_dir: str | Path,
    topology_path: str | Path,
    replica_traj_paths: list[str | Path],
) -> tuple[Path, Path, list[Path]]:
    """Create output directory and normalise key input paths."""

    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    topo_path = Path(topology_path)
    replica_paths = [Path(p) for p in replica_traj_paths]
    return out_dir_path, topo_path, replica_paths


def _parse_temperature_ladder_safe(
    ladder: list[float] | str,
    parser: Callable[[list[float] | str], Sequence[float]],
) -> list[float]:
    """Parse the temperature ladder and surface friendlier errors."""

    try:
        values = list(parser(ladder))
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError("Failed to parse temperature ladder") from exc
    return [float(val) for val in values]


def _load_exchange_records_safe(
    exchange_log_path: str | Path,
    loader: Callable[[str], Sequence[Any]],
) -> list[Any]:
    """Load exchange records with consistent error handling."""

    try:
        records = list(loader(str(exchange_log_path)))
    except FileNotFoundError as exc:
        raise ValueError(f"Exchange log not found: {exchange_log_path}") from exc
    except ValueError:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError("Failed to parse exchange log") from exc
    records.sort(key=lambda rec: rec.step_index)
    return records


def _validate_demux_inputs(
    temperatures: Sequence[float],
    replica_paths: Sequence[Path],
    exchange_records: Sequence[Any],
) -> None:
    """Sanity-check parsed inputs before demultiplexing frames."""

    if len(temperatures) != len(replica_paths):
        raise ValueError(
            "Temperature ladder length does not match number of replica trajectories"
        )
    if not exchange_records:
        raise ValueError("Exchange log contained no exchanges")
    n_temps = len(temperatures)
    if any(len(record.temp_to_replica) != n_temps for record in exchange_records):
        raise ValueError("Exchange log column count does not match temperature ladder")


def _collect_replica_frames(
    reader: Any,
    replica_paths: Sequence[Path],
) -> list[list[np.ndarray]]:
    """Load all frames for each replica using the shared reader."""

    frames_per_replica: list[list[np.ndarray]] = []
    for path in replica_paths:
        count = reader.probe_length(str(path))
        frames = list(reader.iter_frames(str(path), start=0, stop=count, stride=1))
        frames_per_replica.append(frames)
    return frames_per_replica


def _open_demux_writers(
    out_dir_path: Path,
    topo_path: Path,
    temperatures: Sequence[float],
    fmt: str,
) -> tuple[list[MDTrajDCDWriter], list[Path]]:
    """Open one trajectory writer per temperature and return their paths."""

    from .io.trajectory_writer import MDTrajDCDWriter

    writers: list[MDTrajDCDWriter] = []
    paths: list[Path] = []
    for temp in temperatures:
        demux_path = out_dir_path / f"demux_T{float(temp):.0f}K.{fmt}"
        writer = MDTrajDCDWriter()
        writer.open(str(demux_path), topology_path=str(topo_path), overwrite=True)
        writers.append(writer)
        paths.append(demux_path)
    return writers, paths


def _demux_exchange_segments(
    exchange_records: Sequence[Any],
    replica_frames: Sequence[Sequence[np.ndarray]],
    writers: Sequence[MDTrajDCDWriter],
) -> tuple[list[list[Dict[str, Any]]], list[int]]:
    """Replay exchanges and write per-temperature segments."""

    n_temps = len(writers)
    segments_per_temp: list[list[Dict[str, Any]]] = [list() for _ in range(n_temps)]
    dst_positions = [0] * n_temps
    segments_consumed = [0] * len(replica_frames)

    for seg_index, record in enumerate(exchange_records):
        mapping = list(record.temp_to_replica)
        _validate_exchange_mapping(mapping, len(replica_frames), seg_index)
        frame_index = seg_index // max(1, len(replica_frames))

        for temp_index, rep_idx in enumerate(mapping):
            frames_for_replica = replica_frames[rep_idx]
            if frame_index >= len(frames_for_replica):
                raise ValueError(
                    f"Replica {rep_idx} exhausted after {frame_index} frames"
                )

            segments_consumed[rep_idx] += 1
            if segments_consumed[rep_idx] > len(frames_for_replica):
                raise ValueError(
                    f"Replica {rep_idx} consumed more segments than available frames"
                )

            frame = frames_for_replica[frame_index]
            writers[temp_index].write_frames(frame[np.newaxis, :, :])

            src_start = frame_index
            dst_start = dst_positions[temp_index]
            segment_info = {
                "segment_index": int(seg_index),
                "slice_index": int(record.step_index),
                "source_replica": int(rep_idx),
                "src_frame_start": int(src_start),
                "src_frame_stop": int(src_start + 1),
                "dst_frame_start": int(dst_start),
                "dst_frame_stop": int(dst_start + 1),
            }
            segments_per_temp[temp_index].append(segment_info)
            dst_positions[temp_index] += 1

    return segments_per_temp, dst_positions


def _validate_exchange_mapping(
    mapping: Sequence[int],
    n_replicas: int,
    seg_index: int,
) -> None:
    """Ensure each exchange row is a permutation of replica indices."""

    if sorted(mapping) != list(range(n_replicas)):
        raise ValueError("Exchange log rows must be permutations")
    for rep_idx in mapping:
        if rep_idx < 0 or rep_idx >= n_replicas:
            raise ValueError(
                f"Replica index {rep_idx} out of range for segment {seg_index}"
            )


def _close_demux_writers(writers: Sequence[MDTrajDCDWriter]) -> None:
    """Close all trajectory writers, suppressing cleanup issues."""

    for writer in writers:
        try:
            writer.close()
        except Exception:  # pragma: no cover - defensive
            pass


def _write_demux_manifests(
    run_id: str,
    temperatures: Sequence[float],
    dcd_paths: Sequence[Path],
    dst_positions: Sequence[int],
    segments_per_temp: Sequence[Sequence[Dict[str, Any]]],
    dt_ps: float,
    topology_path: Path | None = None,
) -> list[str]:
    """Write JSON manifests for each demultiplexed temperature trajectory."""

    def _compute_digest(traj_path: Path) -> str:
        if topology_path is None:
            return hashlib.sha256(traj_path.read_bytes()).hexdigest()
        try:
            from .io.trajectory_reader import MDTrajReader

            reader = MDTrajReader(topology_path=str(topology_path))
            total = reader.probe_length(str(traj_path))
            if total <= 0:
                return hashlib.sha256(b"").hexdigest()
            digest = hashlib.sha256()
            for frame in reader.iter_frames(
                str(traj_path), start=0, stop=total, stride=1
            ):
                data = np.ascontiguousarray(np.asarray(frame, dtype=np.float32))
                digest.update(data.tobytes())
            return digest.hexdigest()
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.debug("Falling back to file digest for %s: %s", traj_path, exc)
            return hashlib.sha256(traj_path.read_bytes()).hexdigest()

    json_paths: list[str] = []
    run_id_str = str(run_id)
    dt_ps_value = float(dt_ps)
    for temp_index, temp in enumerate(temperatures):
        dcd_path = dcd_paths[temp_index]
        digest = _compute_digest(dcd_path)
        metadata = {
            "schema_version": "2.0",
            "kind": "demux",
            "run_id": run_id_str,
            "temperature_K": float(temp),
            "n_frames": int(dst_positions[temp_index]),
            "dt_ps": dt_ps_value,
            "trajectory": dcd_path.name,
            "segments": list(segments_per_temp[temp_index]),
            "integrity": {"traj_sha256": digest},
        }
        json_path = dcd_path.with_suffix(".json")
        json_path.write_text(json.dumps(metadata, sort_keys=True))
        json_paths.append(str(json_path))
    return json_paths


def extract_last_frame_to_pdb(
    *,
    trajectory_file: str | Path,
    topology_pdb: str | Path,
    out_pdb: str | Path,
    jitter_sigma_A: float = 0.0,
) -> Path:
    """Extract the last frame from a trajectory and write as a PDB.

    Parameters
    ----------
    trajectory_file:
        Path to the input trajectory (e.g., DCD).
    topology_pdb:
        PDB topology defining atom order.
    out_pdb:
        Destination PDB path to write.
    jitter_sigma_A:
        Optional Gaussian noise sigma in Angstroms applied to positions.

    Returns
    -------
    Path
        The output PDB path.
    """
    import mdtraj as _md  # type: ignore
    import numpy as _np

    traj = _md.load(str(trajectory_file), top=str(topology_pdb))
    if traj.n_frames <= 0:
        raise ValueError("Trajectory has no frames to extract")
    last = traj[traj.n_frames - 1]
    if jitter_sigma_A and float(jitter_sigma_A) > 0.0:
        noise = _np.random.normal(0.0, float(jitter_sigma_A), size=last.xyz.shape)
        # MDTraj units are nm; 1 Å = 0.1 nm
        last.xyz = last.xyz + (noise * 0.1)
    out_p = Path(out_pdb)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    last.save_pdb(str(out_p))
    return out_p


def extract_random_highT_frame_to_pdb(
    *,
    run_dir: str | Path,
    topology_pdb: str | Path,
    out_pdb: str | Path,
    jitter_sigma_A: float = 0.0,
    rng_seed: int | None = None,
) -> Path:
    """Extract a random frame from the highest-temperature replica of a run.

    Falls back to the last `replica_*.dcd` when metadata is missing.
    """
    import json as _json

    import mdtraj as _md  # type: ignore
    import numpy as _np

    rd = Path(run_dir)
    analysis_json = rd / "replica_exchange" / "analysis_results.json"
    traj_path: Path | None = None
    if analysis_json.exists():
        try:
            data = _json.loads(analysis_json.read_text())
            remd = data.get("remd", {})
            temps = remd.get("temperatures", [])
            tfiles = remd.get("trajectory_files", [])
            if temps and tfiles and len(temps) == len(tfiles):
                # Choose highest temperature index
                # temps may be nested list from metadata-only; coerce
                temps_f = [float(x) for x in temps]
                i_max = int(_np.argmax(temps_f))
                cand = Path(tfiles[i_max])
                traj_path = cand if cand.is_absolute() else (rd / cand)
        except Exception:
            traj_path = None
    if traj_path is None:
        # Fallback: pick highest replica index .dcd
        dcds = sorted((rd / "replica_exchange").glob("replica_*.dcd"))
        if not dcds:
            raise FileNotFoundError(
                f"No replica_*.dcd found under {rd / 'replica_exchange'}"
            )
        traj_path = dcds[-1]

    traj = _md.load(str(traj_path), top=str(topology_pdb))
    if traj.n_frames <= 0:
        raise ValueError("Trajectory has no frames to extract")
    rng = _np.random.default_rng(rng_seed)
    idx = int(rng.integers(0, traj.n_frames))
    frame = traj[idx]
    if jitter_sigma_A and float(jitter_sigma_A) > 0.0:
        noise = _np.random.normal(0.0, float(jitter_sigma_A), size=frame.xyz.shape)
        frame.xyz = frame.xyz + (noise * 0.1)
    out_p = Path(out_pdb)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    frame.save_pdb(str(out_p))
    return out_p


def build_joint_workflow(
    shards_root: Path,
    temperature_ref_K: float,
    tau_steps: int,
    n_clusters: int,
    *,
    use_reweight: Optional[bool] = None,
) -> Any:
    """Construct a :class:`JointWorkflow` using library defaults only."""

    if use_reweight is None:
        use_reweight = JOINT_USE_REWEIGHT.get()

    cfg = JointWorkflowConfig(
        shards_root=Path(shards_root),
        temperature_ref_K=temperature_ref_K,
        tau_steps=int(tau_steps),
        n_clusters=int(n_clusters),
        use_reweight=bool(use_reweight),
    )
    return JointWorkflow(cfg)  # type: ignore[no-any-return]
