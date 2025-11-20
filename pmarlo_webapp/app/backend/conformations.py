import json
import logging
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np

from pmarlo.analysis.project_cv import apply_whitening_from_metadata
from pmarlo.api.fes import generate_free_energy_surface
from pmarlo.conformations import find_conformations as _DEFAULT_FIND_CONFORMATIONS
from pmarlo.conformations.representative_picker import (
    TrajectoryFrameLocator,
    TrajectorySegment,
)
from pmarlo.conformations.visualizations import (
    plot_pcca_states,
    plot_pcca_states_on_fes,
    plot_tpt_summary as _DEFAULT_PLOT_TPT_SUMMARY,
)
from pmarlo.data.aggregate import load_shards_as_dataset as _DEFAULT_LOAD_SHARDS_AS_DATASET
from pmarlo.markov_state_model.clustering import (
    cluster_microstates as _DEFAULT_CLUSTER_MICROSTATES,
)
from pmarlo.markov_state_model.free_energy import FESResult
from pmarlo.markov_state_model.reduction import reduce_features as _DEFAULT_REDUCE_FEATURES
from pmarlo.utils.msm_utils import (
    build_simple_msm as _DEFAULT_BUILD_SIMPLE_MSM,
)
from pmarlo.utils.path_utils import ensure_directory

from .types import (
    ConformationsConfig,
    ConformationsResult,
    ConformationsResultSchema,
)
from .utils import (
    _is_transition_matrix_reversible,
    _load_metadata_mapping,
    _load_projection_matrix,
    _resolve_workspace_path,
    _sanitize_artifacts,
    _STRUCTURE_EXTENSIONS,
    _timestamp,
)

logger = logging.getLogger(__name__)
_BACKEND_PACKAGE = __name__.rsplit(".", 1)[0]


def _backend_attr(name: str, default: Any) -> Any:
    package = sys.modules.get(_BACKEND_PACKAGE)
    attr = getattr(package, name, None) if package else None
    return attr or default


# Module-level helper functions (can be imported by other modules and frontend tabs)


def resolve_workspace_path(base_dir: Path, candidate: Path) -> Path:
    """Resolve a path that may be relative to the workspace directory."""
    if candidate.is_absolute():
        return candidate.expanduser().resolve()
    return (base_dir / candidate).expanduser().resolve()


def extract_trajectory_names(source: Mapping[str, Any]) -> List[str]:
    """Extract trajectory file names from shard source metadata."""
    names: List[str] = []
    for key in ("traj_files", "trajectories"):
        entries = source.get(key)
        if isinstance(entries, (list, tuple)):
            for entry in entries:
                if isinstance(entry, str) and entry:
                    names.append(entry)
    primary = source.get("traj") or source.get("trajectory") or source.get("path")
    if isinstance(primary, str) and primary:
        names.append(primary)

    deduped: List[str] = []
    seen: set[str] = set()
    for name in names:
        if name not in seen:
            seen.add(name)
            deduped.append(name)
    return deduped


def _extract_frame_range(source: Mapping[str, Any] | None) -> tuple[int, int] | None:
    if not isinstance(source, Mapping):
        return None
    for key in ("range", "frame_range"):
        candidate = source.get(key)
        if (
            isinstance(candidate, Sequence)
            and len(candidate) == 2
            and all(isinstance(v, (int, float)) for v in candidate)
        ):
            start, stop = int(candidate[0]), int(candidate[1])
            if stop > start:
                return start, stop
    return None


def _resolve_shard_source(shard_meta: Mapping[str, Any]) -> Mapping[str, Any]:
    candidate = shard_meta.get("source")
    if isinstance(candidate, Mapping):
        return candidate
    provenance = shard_meta.get("provenance")
    if isinstance(provenance, Mapping):
        nested = provenance.get("source")
        if isinstance(nested, Mapping):
            return nested
        return provenance
    raise ValueError(
        "Shard metadata must declare a provenance source mapping for trajectory resolution"
    )


def _safe_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return None
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return None
    return None


def _segment_tracking_key(shard_path: Path, source_meta: Mapping[str, Any]) -> tuple[str, str, tuple[str, ...]]:
    run_id = source_meta.get("run_id") or source_meta.get("run_uid") or source_meta.get("run_dir")
    run_id = str(run_id) if isinstance(run_id, (str, int, float)) and run_id not in (None, "") else shard_path.parent.name

    replica_raw = source_meta.get("replica_id")
    if replica_raw is None:
        replica_raw = source_meta.get("replica")
    replica_key = (
        str(int(replica_raw))
        if isinstance(replica_raw, (int, float, np.integer, np.floating))
        else (str(replica_raw) if isinstance(replica_raw, str) and replica_raw else "replica-unknown")
    )

    trajectory_names = extract_trajectory_names(source_meta)
    if not trajectory_names:
        fallback = source_meta.get("trajectory") or source_meta.get("path")
        if isinstance(fallback, str) and fallback:
            trajectory_names = [fallback]
        else:
            trajectory_names = [shard_path.stem]

    return (run_id, replica_key, tuple(trajectory_names))


def _maybe_stride_from_ratio(value: Any, frames_loaded: int) -> int | None:
    """Convert a ratio to an integer stride when it divides the feature count."""

    if value is None or frames_loaded <= 0:
        return None

    numerator = _safe_int(value)
    if numerator is None or numerator <= 0:
        return None

    ratio = float(numerator) / float(frames_loaded)
    stride = int(round(ratio))
    if stride <= 0:
        return None
    if not math.isclose(ratio, float(stride), rel_tol=1e-6, abs_tol=1e-6):
        return None
    return stride


def _infer_segment_stride(
    shard_label: str,
    shard_meta: Mapping[str, Any],
    source_meta: Mapping[str, Any],
    frames_loaded: int,
    *,
    frame_span: int | None = None,
) -> int:
    """Infer the physical stride between consecutive frames represented by a shard."""

    if frames_loaded <= 0:
        raise ValueError(
            f"Shard {shard_label} cannot infer stride because it reports no loaded frames"
        )

    hints: List[Tuple[str, int]] = []

    def _append_hint(label: str, value: Any) -> None:
        stride = _safe_int(value)
        if stride is not None and stride > 0:
            hints.append((label, stride))

    _append_hint("shard.effective_frame_stride", shard_meta.get("effective_frame_stride"))
    _append_hint("shard.frame_stride", shard_meta.get("frame_stride"))
    _append_hint("source.frame_stride", source_meta.get("frame_stride"))
    _append_hint("source.stride", source_meta.get("stride"))

    frames_declared_stride = _maybe_stride_from_ratio(
        shard_meta.get("frames_declared"), frames_loaded
    )
    if frames_declared_stride is not None:
        hints.append(("shard.frames_declared", frames_declared_stride))

    if frame_span is not None:
        range_stride = _maybe_stride_from_ratio(frame_span, frames_loaded)
        if range_stride is not None:
            hints.append(("source.frame_range", range_stride))
        elif frame_span != frames_loaded:
            raise ValueError(
                f"Shard {shard_label} frame range span {frame_span} is not divisible by feature count "
                f"{frames_loaded}; cannot infer trajectory frame stride."
            )

    if not hints:
        return 1

    stride_value = hints[0][1]
    for label, value in hints[1:]:
        if value != stride_value:
            raise ValueError(
                f"Shard {shard_label} reports conflicting stride metadata: "
                f"{hints[0][0]}={stride_value} vs {label}={value}."
            )
    return stride_value


def _derive_frame_range_from_metadata(
    shard_path: Path,
    shard_meta: Mapping[str, Any],
    source_meta: Mapping[str, Any],
    tracker: Dict[tuple[str, str, tuple[str, ...]], int],
    *,
    stride: int = 1,
) -> tuple[int, int]:
    """Derive a best-effort frame range when legacy shards omit ``source.range``."""

    frames_loaded = _safe_int(shard_meta.get("frames_loaded"))
    if frames_loaded is None or frames_loaded <= 0:
        start_raw = _safe_int(shard_meta.get("start")) or 0
        stop_raw = _safe_int(shard_meta.get("stop")) or 0
        frames_loaded = stop_raw - start_raw
    if frames_loaded <= 0:
        raise ValueError(
            f"Shard {shard_path.name} cannot derive frame range because it reports no frames"
        )

    stride_value = _safe_int(stride) or 1
    stride_value = max(1, int(stride_value))

    key = _segment_tracking_key(shard_path, source_meta)

    if key not in tracker:
        initial = (
            _safe_int(source_meta.get("segment_start"))
            or _safe_int(source_meta.get("frame_start"))
            or _safe_int(source_meta.get("range_start"))
            or _safe_int(source_meta.get("start_frame"))
            or _safe_int(shard_meta.get("start"))
            or 0
        )
        tracker[key] = int(initial)

    local_start = tracker[key]
    local_stop = local_start + int(frames_loaded) * stride_value
    tracker[key] = local_stop

    if (
        isinstance(source_meta, dict)
        and "range" not in source_meta
        and "frame_range" not in source_meta
    ):
        source_meta["range"] = [local_start, local_stop]

    return local_start, local_stop


def _ensure_fes_variation(values: np.ndarray) -> np.ndarray:
    """Guarantee a finite span for FES plotting by injecting minimal jitter."""

    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        return arr
    finite = arr[np.isfinite(arr)]
    span = float(np.max(finite) - np.min(finite)) if finite.size else 0.0
    if span > 0.0:
        return arr
    baseline = float(finite[0]) if finite.size else 0.0
    scale = max(1.0, abs(baseline)) * 1e-6
    if scale == 0.0:
        scale = 1e-6
    jitter = np.linspace(-0.5, 0.5, arr.size, dtype=float) * scale
    if arr.size == 1:
        jitter[0] = scale
    return arr + jitter


def _infer_cluster_centers(features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Derive cluster centers directly from frame assignments."""

    matrix = np.asarray(features, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("Cannot derive cluster centers from non-2D feature matrix")
    assignments = np.asarray(labels)
    if assignments.ndim != 1:
        assignments = assignments.reshape(-1)
    if assignments.size != matrix.shape[0]:
        raise ValueError(
            "Cluster labels length does not match number of frames; cannot derive centers"
        )
    centers: List[np.ndarray] = []
    for state in np.unique(assignments):
        mask = assignments == state
        if not np.any(mask):
            raise ValueError(
                f"Cluster label {state} has no member frames; cannot derive medoid."
            )
        centers.append(np.mean(matrix[mask], axis=0))
    return np.vstack(centers)


def _load_shards_dataset(shards: Sequence[Path]) -> Dict[str, Any]:
    """Resolve the dataset loader through the backend package to honor patches."""

    loader = _backend_attr("load_shards_as_dataset", _DEFAULT_LOAD_SHARDS_AS_DATASET)
    return loader(shards)


class ConformationsMixin:
    """Methods for TPT conformations analysis.

    This class is mixed into the Backend class to provide conformations
    analysis operations including loading, running, and managing TPT results.
    """

    def run_conformations_analysis(
            self,
            shard_jsons: Sequence[Path],
            config: ConformationsConfig,
    ) -> ConformationsResult:
        """Run TPT conformations analysis on shards.

        Args:
            shard_jsons: Paths to shard JSON files
            config: Configuration for conformations analysis

        Returns:
            ConformationsResult with outputs and metadata
        """
        stamp = _timestamp()
        output_dir = self.layout.bundles_dir / f"conformations-{stamp}"
        ensure_directory(output_dir)
        config_dict: Dict[str, Any] = ConformationsResultSchema.serialize_config(
            config
        )
        summary_path = output_dir / "conformations_summary.json"

        fes_result: Optional[FESResult] = None

        try:
            logger.info(f"Loading {len(shard_jsons)} shards for conformations analysis")
            shards = [Path(p).resolve() for p in shard_jsons]
            if not shards:
                raise ValueError("No shards selected for conformations analysis")

            dataset = _load_shards_dataset(shards)

            if "X" not in dataset or len(dataset["X"]) == 0:
                raise ValueError("No feature data found in shards")

            features = np.asarray(dataset["X"], dtype=float)
            logger.info(f"Loaded {features.shape[0]} frames with {features.shape[1]} features")

            # Extract feature names for explainability in plots
            cv_names = dataset.get("cv_names", [])
            if not cv_names or len(cv_names) != features.shape[1]:
                cv_names = [f"Feature {i+1}" for i in range(features.shape[1])]
            feature_names_str = ", ".join(cv_names[:5])
            if len(cv_names) > 5:
                feature_names_str += f", ... ({len(cv_names)} total)"
            logger.info(f"Feature names: {feature_names_str}")

            if config.topology_pdb is None:
                raise ValueError(
                    "A topology PDB must be specified for conformations analysis."
                )

            raw_topology = Path(config.topology_pdb)
            if raw_topology.is_absolute():
                topology_pdb = raw_topology.expanduser().resolve()
            else:
                topology_pdb = (self.layout.workspace_dir / raw_topology).expanduser().resolve()

            if not topology_pdb.exists():
                raise FileNotFoundError(
                    f"Topology PDB {topology_pdb} does not exist."
                )

            shard_meta_list = dataset.get("__shards__", [])
            if not shard_meta_list:
                raise ValueError(
                    "Shard metadata missing from aggregated dataset; cannot locate trajectories."
                )

            cv_method = (config.cv_method or "tica").strip().lower()
            tica_dim = (
                int(config.tica_dim)
                if config.tica_dim is not None
                else int(config.n_components)
            )
            if cv_method == "deeptica":
                if config.deeptica_projection_path is None:
                    raise ValueError(
                        "deeptica_projection_path is required when cv_method='deeptica'"
                    )
                projection_path = _resolve_workspace_path(
                    self.layout.workspace_dir,
                    Path(config.deeptica_projection_path),
                )
                logger.info("Loading precomputed DeepTICA projection from %s", projection_path)
                features_reduced = _load_projection_matrix(projection_path)
                if features_reduced.shape[0] != features.shape[0]:
                    raise ValueError(
                        "DeepTICA projection frame count does not match loaded features"
                    )
                if config.deeptica_metadata_path is not None:
                    metadata_path = _resolve_workspace_path(
                        self.layout.workspace_dir,
                        Path(config.deeptica_metadata_path),
                    )
                    logger.info("Applying DeepTICA whitening metadata from %s", metadata_path)
                    metadata = _load_metadata_mapping(metadata_path)
                    features_reduced, _ = apply_whitening_from_metadata(
                        np.asarray(features_reduced, dtype=float), metadata
                    )
                else:
                    features_reduced = np.asarray(features_reduced, dtype=float)
            elif cv_method == "tica":
                logger.info(
                    "Reducing features with TICA (n_components=%d)",
                    tica_dim,
                )
                reducer = _backend_attr("reduce_features", _DEFAULT_REDUCE_FEATURES)
                features_reduced = reducer(
                    features,
                    method="tica",
                    lag=config.lag,
                    n_components=tica_dim,
                )
            else:
                raise ValueError(f"Unsupported cv_method '{config.cv_method}' for conformations")

            if features_reduced.shape[1] < 2:
                raise ValueError(
                    "At least two collective variable components are required to compute the FES overlay."
                )
            cv1 = _ensure_fes_variation(features_reduced[:, 0])
            cv2 = _ensure_fes_variation(features_reduced[:, 1])
            frame_count = int(cv1.size)
            adaptive_bins = max(30, min(150, int(max(1, frame_count) ** 0.5)))
            fes_label_prefix = "DeepTICA" if cv_method == "deeptica" else "TICA"

            # Generate descriptive axis labels with feature information
            def _format_tica_label(component_idx: int, method: str, features: list) -> str:
                """Create a descriptive label showing TICA component and input features."""
                if not features:
                    return f"{method} {component_idx}"

                # Show up to 3 feature names for readability
                max_features = 3
                if len(features) <= max_features:
                    feature_list = ", ".join(features)
                else:
                    feature_list = ", ".join(features[:max_features]) + f", ... +{len(features) - max_features} more"

                return f"{method} {component_idx} ({feature_list})"

            cv1_label = _format_tica_label(1, fes_label_prefix, cv_names)
            cv2_label = _format_tica_label(2, fes_label_prefix, cv_names)

            try:
                fes_result = generate_free_energy_surface(
                    cv1,
                    cv2,
                    bins=(adaptive_bins, adaptive_bins),
                    temperature=float(config.temperature),
                    periodic=(False, False),
                    smooth=True,
                    min_count=1,
                )
            except Exception as exc:
                raise RuntimeError(
                    "Failed to compute the Free Energy Surface required for the metastable overlay plot."
                ) from exc

            if fes_result is None or fes_result.F is None:
                raise RuntimeError("Free Energy Surface generation returned an empty result.")

            if fes_result.cv1_name is None:
                fes_result.cv1_name = cv1_label
                fes_result.metadata["cv1_name"] = fes_result.cv1_name
            if fes_result.cv2_name is None:
                fes_result.cv2_name = cv2_label
                fes_result.metadata["cv2_name"] = fes_result.cv2_name

            cluster_mode = (config.cluster_mode or "kmeans").strip().lower()
            method_alias = {
                "kmeans": "kmeans",
                "minibatchkmeans": "minibatchkmeans",
                "auto": "auto",
                "dbscan": "dbscan",
            }
            if cluster_mode not in method_alias:
                raise ValueError(
                    "Unsupported cluster_mode for conformations analysis: "
                    f"{config.cluster_mode!r}."
                )
            cluster_method = method_alias[cluster_mode]

            cluster_kwargs: Mapping[str, Any]
            if config.kmeans_kwargs is None:
                cluster_kwargs = {}
            elif isinstance(config.kmeans_kwargs, Mapping):
                cluster_kwargs = dict(config.kmeans_kwargs)
            else:
                raise TypeError(
                    "ConformationsConfig.kmeans_kwargs must be a mapping; "
                    f"received {type(config.kmeans_kwargs).__name__}."
                )

            n_clusters = int(config.n_clusters)
            if n_clusters <= 0:
                raise ValueError(
                    "ConformationsConfig.n_clusters must be a positive integer"
                )

            total_frames = int(features_reduced.shape[0])
            if total_frames == 0:
                raise ValueError("No frames available after dimensionality reduction")

            if cluster_method != "dbscan":
                frames_per_cluster = total_frames / float(n_clusters)
                if frames_per_cluster > 10_000:
                    logger.warning(
                        (
                            "Using %d clusters for %d frames (~%.0f frames/cluster) may be too coarse. "
                            "Increase n_clusters to improve transition state resolution."
                        ),
                        n_clusters,
                        total_frames,
                        frames_per_cluster,
                    )
            if cluster_method == "dbscan":
                cluster_kwargs.pop("n_init", None)
                n_states_requested: int | Literal["auto"] = "auto"
                n_init_kwargs: Dict[str, Any] = {}
                logger.info(
                    "Clustering with DBSCAN (seed=%s, kwargs=%s)",
                    "None" if config.cluster_seed is None else int(config.cluster_seed),
                    cluster_kwargs,
                )
            else:
                n_states_requested = n_clusters
                n_init_kwargs = {"n_init": int(config.kmeans_n_init)}
                logger.info(
                    "Clustering into %d microstates using %s (seed=%s, n_init=%d, kwargs=%s)",
                    n_clusters,
                    cluster_method,
                    "None" if config.cluster_seed is None else int(config.cluster_seed),
                    int(config.kmeans_n_init),
                    cluster_kwargs,
                )

            clusterer = _backend_attr("cluster_microstates", _DEFAULT_CLUSTER_MICROSTATES)
            clustering_result = clusterer(
                features_reduced,
                method=cluster_method,
                n_states=n_states_requested,
                random_state=(
                    None
                    if config.cluster_seed is None
                    else int(config.cluster_seed)
                ),
                **n_init_kwargs,
                **cluster_kwargs,
            )
            labels_raw = getattr(clustering_result, "labels", None)
            if labels_raw is None:
                raise ValueError(
                    "Clustering did not return microstate labels required for MSM construction."
                )
            labels = np.asarray(labels_raw, dtype=int).reshape(-1)

            centers = getattr(clustering_result, "centers", None)
            if centers is None:
                logger.info(
                    "Clustering result is missing centers; deriving medoids from frame assignments."
                )
                centers = _infer_cluster_centers(features_reduced, labels)
            cluster_centers = np.asarray(centers, dtype=float)
            if cluster_centers.ndim != 2:
                raise ValueError(
                    "Cluster centers must be a 2D array to generate the PCCA visualization."
                )
            if cluster_centers.shape[1] < 2:
                raise ValueError(
                    "At least two TICA dimensions are required to plot PCCA metastable states."
                )
            tica_cluster_coords = cluster_centers[:, :2]
            n_states = int(np.max(labels) + 1)
            if n_states < 2:
                raise ValueError(
                    "Clustering produced fewer than two microstates; increase the "
                    "number of clusters or relax DBSCAN parameters (eps, min_samples)."
                )
            if cluster_method == "dbscan" and n_states < int(config.n_metastable):
                raise ValueError(
                    "DBSCAN clustering yielded only %d microstate(s), but the analysis "
                    "requested %d metastable states. Adjust the DBSCAN parameters "
                    "(e.g., larger eps or smaller min_samples) or lower the metastable count."
                    % (n_states, int(config.n_metastable))
                )

            logger.info("Building MSM (lag=%s) via backend helper", config.lag)
            msm_builder = _backend_attr("build_simple_msm", _DEFAULT_BUILD_SIMPLE_MSM)
            T, pi = msm_builder([labels], n_states=n_states, lag=int(config.lag))

            if not _is_transition_matrix_reversible(T, pi):
                raise ValueError(
                    "Transition matrix is not reversible; TPT requires detailed balance."
                )

            locator = self._build_trajectory_locator(shards, shard_meta_list)
            logger.info(
                "Resolved %d trajectory segments for representative extraction",
                len(locator.segments),
            )

            logger.info("Running TPT conformations analysis")

            msm_data = {
                'T': T,
                'pi': pi,
                'dtrajs': [labels],
                'features': features_reduced,
                'fes': fes_result,
            }

            conformer = _backend_attr("find_conformations", _DEFAULT_FIND_CONFORMATIONS)
            conf_result = conformer(
                msm_data=msm_data,
                source_states=np.array(config.source_states) if config.source_states else None,
                sink_states=np.array(config.sink_states) if config.sink_states else None,
                auto_detect=config.auto_detect_states,
                auto_detect_method='auto',
                find_transition_states=True,
                find_metastable_states=True,
                find_pathway_intermediates=True,
                compute_kis=config.compute_kis,
                uncertainty_analysis=config.uncertainty_analysis,
                n_bootstrap=config.bootstrap_samples,
                lag=int(config.lag),
                representative_selection='medoid',
                output_dir=str(output_dir),
                save_structures=True,
                topology_path=str(topology_pdb),
                trajectory_locator=locator,
                tica__dim=tica_dim,
                committor_thresholds=tuple(config.committor_thresholds),
                n_metastable=config.n_metastable,
                temperature=float(config.temperature),
            )

            macro_memberships_data = conf_result.metadata.get("macrostate_memberships")
            if macro_memberships_data is None:
                raise ValueError(
                    "Conformations analysis did not return PCCA memberships required for visualization."
                )
            pcca_memberships = np.asarray(macro_memberships_data, dtype=float)
            if pcca_memberships.ndim != 2:
                raise ValueError(
                    "PCCA memberships must be a 2D array to generate the metastable state plot."
                )
            if pcca_memberships.shape[0] != tica_cluster_coords.shape[0]:
                raise ValueError(
                    "The number of PCCA membership rows does not match the number of microstate clusters."
                )

            pcca_plot_path = output_dir / "pcca_states.png"
            plot_pcca_states(
                tica_cluster_coords,
                pcca_memberships,
                str(pcca_plot_path),
                xlabel=cv1_label,
                ylabel=cv2_label,
            )
            if fes_result is None:
                raise RuntimeError("Free Energy Surface was not computed; cannot build overlay plot.")
            pcca_fes_plot_path = output_dir / "pcca_states_on_fes.png"
            plot_pcca_states_on_fes(
                fes_result,
                tica_cluster_coords,
                pcca_memberships,
                str(pcca_fes_plot_path),
            )

            tpt_result = conf_result.tpt_result
            if tpt_result is None:
                raise RuntimeError(
                    "TPT analysis did not produce a result. Ensure the transition matrix is reversible and source/sink states are valid."
                )

            logger.info("Generating visualizations")
            plotter = _backend_attr("plot_tpt_summary", _DEFAULT_PLOT_TPT_SUMMARY)
            plotter(tpt_result, str(output_dir))
            plots = {
                "pcca_states": pcca_plot_path,
                "pcca_states_on_fes": pcca_fes_plot_path,
            }
            for plot_name in ("committors", "flux_network", "pathways"):
                plot_path = output_dir / f"{plot_name}.png"
                if plot_path.exists():
                    plots[plot_name] = plot_path

            tpt_summary = {
                "rate": float(tpt_result.rate),
                "mfpt": float(tpt_result.mfpt),
                "total_flux": float(tpt_result.total_flux),
                "n_pathways": len(tpt_result.pathways),
                "source_states": tpt_result.source_states.tolist(),
                "sink_states": tpt_result.sink_states.tolist(),
                "tpt_converged": bool(tpt_result.tpt_converged),
                "pathway_iterations": int(tpt_result.pathway_iterations),
                "pathway_max_iterations": int(tpt_result.pathway_max_iterations),
            }

            metastable_states: Dict[str, Dict[str, Any]] = {}
            for conf in conf_result.get_metastable_states():
                macro_id = (
                    int(conf.macrostate_id)
                    if conf.macrostate_id is not None
                    else int(conf.state_id)
                )
                micro_ids = conf.metadata.get("microstate_ids", [])
                n_states = len(micro_ids) if isinstance(micro_ids, list) else 0
                metastable_states[str(macro_id)] = {
                    "population": float(conf.population),
                    "n_states": n_states,
                    "representative_pdb": (
                        str(conf.structure_path)
                        if conf.structure_path is not None
                        else None
                    ),
                }

            transition_states: List[Dict[str, Any]] = []
            for conf in conf_result.get_transition_states():
                transition_states.append(
                    {
                        "committor": float(conf.committor) if conf.committor is not None else 0.0,
                        "state_index": int(conf.state_id),
                        "representative_pdb": (
                            str(conf.structure_path)
                            if conf.structure_path is not None
                            else None
                        ),
                    }
                )

            pathways: List[List[int]] = []
            for path in tpt_result.pathways:
                pathways.append([int(state) for state in path])

            representative_pdbs = [p.resolve() for p in sorted(output_dir.glob("*.pdb"))]

            result_schema = ConformationsResultSchema(
                output_dir=output_dir.resolve(),
                tpt=tpt_summary,
                metastable_states=metastable_states,
                transition_states=transition_states,
                pathways=pathways,
                config=config,
                created_at=stamp,
                plots={name: Path(path).resolve() for name, path in plots.items()},
                representative_pdbs=representative_pdbs,
                tpt_converged=bool(tpt_result.tpt_converged),
                tpt_pathway_iterations=int(tpt_result.pathway_iterations),
                tpt_pathway_max_iterations=int(tpt_result.pathway_max_iterations),
            )

            summary_path.write_text(
                result_schema.model_dump_json(indent=2),
                encoding="utf-8",
            )

            logger.info(f"Conformations analysis complete. Output saved to {output_dir}")

            conf_result_obj = result_schema.to_result()

            state_entry = result_schema.model_dump(mode="json")
            state_entry.update(
                {
                    "output_dir": str(output_dir.resolve()),
                    "summary": str(summary_path.resolve()),
                    "tpt_summary": state_entry.get("tpt"),
                }
            )

            self.state.append_conformations(state_entry)

            return conf_result_obj

        except Exception as e:
            logger.error(f"Conformations analysis failed: {e}", exc_info=True)
            self.state.append_conformations(
                {
                    "output_dir": str(output_dir.resolve()),
                    "summary": str(summary_path.resolve()),
                    "created_at": stamp,
                    "config": _sanitize_artifacts(config_dict),
                    "error": str(e),
                }
            )
            return ConformationsResult(
                output_dir=output_dir,
                tpt_summary={},
                metastable_states={},
                transition_states=[],
                pathways=[],
                representative_pdbs=[],
                plots={},
                created_at=stamp,
                config=config,
                error=str(e),
                tpt_converged=True,
                tpt_pathway_iterations=None,
                tpt_pathway_max_iterations=None,
            )

    def load_conformations(
            self, handle: int | Mapping[str, Any]
    ) -> Optional[ConformationsResult]:
        """Load conformations result from state by handle or index."""
        if isinstance(handle, Mapping):
            entry = dict(handle)
            state_idx = entry.get("state_index")
            try:
                idx = int(state_idx) if state_idx is not None else None
            except (TypeError, ValueError):
                idx = None
            if idx is not None and 0 <= idx < len(self.state.conformations):
                entry = dict(self.state.conformations[idx])
                entry["state_index"] = idx
                return self._load_conformations_from_entry(entry)
            return self._load_conformations_from_entry(entry)

        index = int(handle)
        if index < 0 or index >= len(self.state.conformations):
            return None
        entry = dict(self.state.conformations[index])
        entry["state_index"] = index
        return self._load_conformations_from_entry(entry)

    def list_conformations(self) -> List[Dict[str, Any]]:
        """Return recorded conformations analyses with resolved paths."""
        self._reconcile_conformation_state()
        entries: List[Dict[str, Any]] = []
        for idx, entry in enumerate(self.state.conformations):
            data = dict(entry)
            data["state_index"] = idx
            output_dir_raw = data.get("output_dir") or data.get("directory")
            output_dir_path = self._path_from_value(output_dir_raw)
            if output_dir_path is not None:
                data["output_dir"] = str(output_dir_path)
            summary_raw = data.get("summary") or data.get("summary_path")
            summary_path = self._path_from_value(summary_raw)
            if summary_path is not None:
                data["summary"] = str(summary_path)
            entries.append(data)

        entries.sort(key=lambda e: str(e.get("created_at", "")), reverse=True)
        return entries

    def _reconcile_conformation_state(self) -> None:
        """Drop conformations entries whose artifacts no longer exist."""
        try:
            to_delete: List[int] = []
            for i, entry in enumerate(list(self.state.conformations)):
                output_dir = self._path_from_value(
                    entry.get("output_dir") or entry.get("directory")
                )
                summary_path = self._path_from_value(
                    entry.get("summary") or entry.get("summary_path")
                )

                # If summary_path is None, try default location
                if summary_path is None and output_dir is not None:
                    summary_path = output_dir / "conformations_summary.json"

                # Entry is valid only if the summary file exists
                # (directory alone is not sufficient, as it might be empty from failed runs)
                exists = summary_path is not None and summary_path.exists()

                if not exists:
                    to_delete.append(i)

            for idx in reversed(to_delete):
                try:
                    self.state.remove_conformations(idx)
                except Exception:
                    pass
        except Exception:
            pass

    def _load_conformations_from_entry(
            self, entry: Mapping[str, Any]
    ) -> Optional[ConformationsResult]:
        """Load conformations result from entry dictionary."""
        output_dir = self._path_from_value(
            entry.get("output_dir") or entry.get("directory")
        )
        if output_dir is None:
            logger.warning(f"Could not resolve output_dir from entry: {entry.get('output_dir') or entry.get('directory')}")
            return None

        summary_raw = entry.get("summary") or entry.get("summary_path")
        summary_path = self._path_from_value(summary_raw)
        if summary_path is None:
            summary_path = output_dir / "conformations_summary.json"

        if not summary_path.exists():
            alt_summary = output_dir / "conformations_summary.json"
            if alt_summary.exists():
                summary_path = alt_summary
            else:
                logger.warning(
                    f"Conformations summary not found. Tried:\n"
                    f"  - {summary_path}\n"
                    f"  - {alt_summary}\n"
                    f"  output_dir exists: {output_dir.exists() if output_dir else 'N/A'}"
                )
                return None

        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            return None

        tpt_summary = dict(payload.get("tpt") or {})
        metastable_states = dict(payload.get("metastable_states") or {})
        transition_states = list(payload.get("transition_states") or [])
        pathways = list(payload.get("pathways") or [])
        created_at = str(payload.get("created_at", entry.get("created_at", _timestamp())))

        config_payload = payload.get("config") or entry.get("config") or {}
        config_obj = self._conformations_config_from_entry(config_payload)

        plots: Dict[str, Path] = {}
        entry_plots = entry.get("plots")
        if isinstance(entry_plots, Mapping):
            for name, plot_path in entry_plots.items():
                candidate = self._path_from_value(plot_path)
                if candidate is not None and candidate.exists():
                    plots[str(name)] = candidate.resolve()

        try:
            for plot_path in output_dir.glob("*.png"):
                plots.setdefault(plot_path.stem, plot_path.resolve())
        except Exception:
            pass

        try:
            representative_pdbs = [p.resolve() for p in sorted(output_dir.glob("*.pdb"))]
        except Exception:
            representative_pdbs = []

        tpt_converged = bool(
            entry.get("tpt_converged", tpt_summary.get("tpt_converged", True))
        )

        tpt_iterations = entry.get("tpt_pathway_iterations")
        if tpt_iterations is None:
            tpt_iterations = tpt_summary.get("pathway_iterations")

        tpt_max_iterations = entry.get("tpt_pathway_max_iterations")
        if tpt_max_iterations is None:
            tpt_max_iterations = tpt_summary.get("pathway_max_iterations")

        error_message = entry.get("error")
        error_str = str(error_message) if error_message is not None else None

        try:
            result_schema = ConformationsResultSchema(
                output_dir=output_dir.resolve(),
                tpt=tpt_summary,
                metastable_states=metastable_states,
                transition_states=transition_states,
                pathways=pathways,
                representative_pdbs=representative_pdbs,
                plots=plots,
                created_at=created_at,
                config=config_obj,
                tpt_converged=bool(tpt_converged),
                tpt_pathway_iterations=(
                    int(tpt_iterations) if tpt_iterations is not None else None
                ),
                tpt_pathway_max_iterations=(
                    int(tpt_max_iterations) if tpt_max_iterations is not None else None
                ),
            )
        except Exception as exc:
            logger.warning(
                "Unable to parse conformations summary at %s: %s", summary_path, exc
            )
            return None

        result = result_schema.to_result()
        result.error = error_str
        return result

    def _conformations_config_from_entry(
            self, payload: Any
    ) -> ConformationsConfig:
        """Parse ConformationsConfig from entry payload."""
        if not isinstance(payload, Mapping):
            return ConformationsConfig()

        data: Dict[str, Any] = dict(payload)
        for key in (
                "topology_pdb",
                "deeptica_projection_path",
                "deeptica_metadata_path",
        ):
            value = data.get(key)
            if value:
                try:
                    candidate = Path(value).expanduser()
                except Exception:
                    candidate = Path(str(value)).expanduser()
                if candidate.is_absolute():
                    candidate = self.layout.rebase_legacy_path(candidate)
                data[key] = candidate
            else:
                data[key] = None

        thresholds = data.get("committor_thresholds")
        if isinstance(thresholds, (list, tuple)):
            try:
                data["committor_thresholds"] = tuple(float(v) for v in thresholds)
            except Exception:
                data.pop("committor_thresholds", None)

        for state_key in ("source_states", "sink_states"):
            states = data.get(state_key)
            if isinstance(states, (list, tuple)):
                try:
                    data[state_key] = [int(s) for s in states]
                except Exception:
                    data[state_key] = [int(float(s)) for s in states if s is not None]

        cluster_seed = data.get("cluster_seed")
        if cluster_seed is not None:
            try:
                data["cluster_seed"] = int(cluster_seed)
            except (TypeError, ValueError):
                data["cluster_seed"] = None

        try:
            return ConformationsConfig(**data)
        except Exception:
            return ConformationsConfig()

    def _build_trajectory_locator(
            self,
            shard_paths: Sequence[Path],
            shard_meta_list: Sequence[Mapping[str, Any]],
    ) -> TrajectoryFrameLocator:
        """Build trajectory locator from shard metadata."""
        if len(shard_paths) != len(shard_meta_list):
            raise ValueError(
                "Shard metadata length mismatch; cannot map trajectories reliably."
            )

        segments: List[TrajectorySegment] = []
        derived_ranges: Dict[tuple[str, str, tuple[str, ...]], int] = {}
        for idx, (shard_path, shard_meta) in enumerate(zip(shard_paths, shard_meta_list)):
            if not isinstance(shard_meta, Mapping):
                raise TypeError(
                    f"Shard metadata entry {idx} is not a mapping; unable to resolve trajectories."
                )

            start_raw = shard_meta.get("start")
            stop_raw = shard_meta.get("stop")
            if start_raw is None or stop_raw is None:
                raise ValueError(
                    f"Shard metadata for {shard_path.name} must include start/stop offsets"
                )
            start = int(start_raw)
            stop = int(stop_raw)
            if stop <= start:
                raise ValueError(
                    f"Shard {shard_path.name} reports non-positive frame span ({start}->{stop})"
                )

            frames_loaded = int(shard_meta.get("frames_loaded", stop - start))
            if frames_loaded != stop - start:
                raise ValueError(
                    f"Shard {shard_path.name} has inconsistent frame counts (loaded={frames_loaded}, span={stop - start})"
                )

            source_meta = _resolve_shard_source(shard_meta)
            frame_range = _extract_frame_range(source_meta)
            if frame_range is None:
                local_stride = _infer_segment_stride(
                    shard_path.name, shard_meta, source_meta, frames_loaded
                )
                local_start, local_stop = _derive_frame_range_from_metadata(
                    shard_path,
                    shard_meta,
                    source_meta,
                    derived_ranges,
                    stride=local_stride,
                )
                logger.warning(
                    "[conformations] Shard %s is missing explicit frame range metadata; "
                    "deriving span [%d, %d) from shard ordering. This may be approximate when "
                    "multiple trajectory files are interleaved.",
                    shard_path.name,
                    local_start,
                    local_stop,
                )
            else:
                local_start = int(frame_range[0])
                local_stop = int(frame_range[1])
                frame_span = local_stop - local_start
                if frame_span <= 0:
                    raise ValueError(
                        f"Shard {shard_path.name} frame range ({local_start}->{local_stop}) "
                        "is not a valid positive span"
                    )
                local_stride = _infer_segment_stride(
                    shard_path.name,
                    shard_meta,
                    source_meta,
                    frames_loaded,
                    frame_span=frame_span,
                )
                expected_span = frames_loaded * local_stride
                if frame_span != expected_span:
                    raise ValueError(
                        f"Shard {shard_path.name} frame range ({local_start}->{local_stop}) span {frame_span} "
                        f"does not match feature count {frames_loaded} with stride {local_stride}"
                    )

            trajectory_names = extract_trajectory_names(source_meta)
            trajectory_path = self._resolve_trajectory_path(shard_path, trajectory_names)

            segments.append(
                TrajectorySegment(
                    path=trajectory_path,
                    start=start,
                    stop=stop,
                    local_start=local_start,
                    local_stride=local_stride,
                )
            )

        segments.sort(key=lambda seg: seg.start)
        for prev, current in zip(segments, segments[1:]):
            if current.start < prev.stop:
                raise ValueError(
                    "Shard frame intervals overlap; cannot resolve representative frames to unique trajectories."
                )

        return TrajectoryFrameLocator(tuple(segments))

    def _resolve_trajectory_path(
            self, shard_path: Path, raw_names: Sequence[str]
    ) -> Path:
        """Resolve trajectory path from shard metadata."""
        if not raw_names:
            raise ValueError(
                f"Shard {shard_path} does not declare any trajectory file references"
            )

        shard_dir = shard_path.parent.resolve()
        search_bases = [shard_dir, self.layout.workspace_dir.resolve()]

        for name in raw_names:
            candidate = self.layout.rebase_legacy_path(Path(name))
            stem = candidate.stem
            if candidate.is_absolute():
                targets = [candidate.resolve()]
            else:
                targets = [(base / candidate).resolve() for base in search_bases]

            for target in targets:
                resolved = self._maybe_resolve_structure_file(target, stem, shard_dir)
                if resolved is not None:
                    if not resolved.exists():
                        raise FileNotFoundError(
                            f"Trajectory file {resolved} referenced by {shard_path} does not exist"
                        )
                    return resolved.resolve()

        raise FileNotFoundError(
            f"Could not resolve trajectory file for shard {shard_path.name}."
        )

    def _maybe_resolve_structure_file(
            self, target: Path, stem: str, shard_dir: Path
    ) -> Optional[Path]:
        """Try to resolve a structure file path."""
        suffix = target.suffix.lower()
        if target.exists():
            if suffix in _STRUCTURE_EXTENSIONS:
                return target.resolve()
            if suffix in {".npz", ".npy"}:
                alt = self._search_structure_by_stem(target.parent, stem)
                if alt is None:
                    raise FileNotFoundError(
                        f"Feature archive {target} does not have a matching structural trajectory"
                    )
                return alt
            raise ValueError(
                f"Unsupported trajectory file extension '{target.suffix}' for {target}"
            )

        alt = self._search_structure_by_stem(target.parent, stem)
        if alt is not None:
            return alt
        if target.parent != shard_dir:
            alt = self._search_structure_by_stem(shard_dir, stem)
            if alt is not None:
                return alt
        return None

    def _search_structure_by_stem(self, base: Path, stem: str) -> Optional[Path]:
        """Search for structure file by stem in directory."""
        if not stem:
            return None
        base = base.resolve()
        for ext in _STRUCTURE_EXTENSIONS:
            candidate = (base / f"{stem}{ext}").resolve()
            if candidate.exists():
                return candidate
        return None
