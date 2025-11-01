import json
import logging
import shutil
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from pmarlo.analysis import export_analysis_debug
from pmarlo.analysis.debug_export import (
    AnalysisDebugData,
    _coverage_fraction,
    _strongly_connected_components,
    _transition_diag_mass,
    _compute_dwell_times,
    _compute_occupancy_tail,
)
from pmarlo.api.shards import build_from_shards
from pmarlo.data.aggregate import load_shards_as_dataset

from .types import BuildArtifact, BuildConfig, _AnalysisMSMStats
from .utils import _sanitize_artifacts, _timestamp
from .metrics_logger import MetricsLogger

logger = logging.getLogger(__name__)


# Module-level helper functions (can be imported by other modules and frontend tabs)
def analysis_total_pairs(
        msm_obj: Mapping[str, Any], counts: Any | None = None
) -> int:
    """Derive the total number of (t, t+tau) pairs from the MSM payload."""
    counted_pairs = msm_obj.get("counted_pairs")
    if isinstance(counted_pairs, Mapping) and counted_pairs:
        if "all" in counted_pairs and counted_pairs["all"] is not None:
            try:
                return int(counted_pairs["all"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid 'all' counted_pairs value {counted_pairs['all']!r}"
                ) from exc
        totals: list[int] = []
        for value in counted_pairs.values():
            if value is None:
                continue
            try:
                totals.append(int(value))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid counted_pairs entry {value!r}; cannot determine total pairs"
                ) from exc
        if totals:
            return int(sum(totals))

    counts_payload = counts if counts is not None else msm_obj.get("counts")
    if counts_payload is None:
        raise ValueError(
            "MSM payload lacks counted_pairs and counts; cannot derive total transition pairs"
        )
    arr = np.asarray(counts_payload, dtype=np.float64)
    if arr.size == 0:
        return 0
    return int(np.rint(arr.sum()))


def compute_analysis_diag_mass(
    transition_matrix: Any,
) -> tuple[float, np.ndarray | None, dict[str, Any] | None]:
    """Compute the mean diagonal mass of a transition matrix.

    Parameters
    ----------
    transition_matrix:
        Final MSM transition matrix or an object convertible to an array.

    Returns
    -------
    diag_mass, matrix, guardrail
        A tuple containing the diagonal mass as a float (``nan`` when
        unavailable), the normalised transition matrix (or ``None`` if it could
        not be coerced), and an optional guardrail violation to record.
    """

    if transition_matrix is None:
        return (
            float("nan"),
            None,
            {"code": "diag_mass_unavailable", "actual": "missing_transition_matrix"},
        )

    try:
        matrix = np.asarray(transition_matrix, dtype=np.float64)
    except Exception:
        return (
            float("nan"),
            None,
            {"code": "diag_mass_unavailable", "actual": "non_numeric_transition_matrix"},
        )

    if matrix.size == 0:
        return (
            float("nan"),
            None,
            {"code": "diag_mass_unavailable", "actual": "empty_transition_matrix"},
        )

    if not np.all(np.isfinite(matrix)):
        return (
            float("nan"),
            matrix,
            {"code": "diag_mass_unavailable", "actual": "nonfinite_transition_matrix"},
        )

    diag = np.diag(matrix)
    if diag.size == 0:
        return (
            float("nan"),
            matrix,
            {"code": "diag_mass_unavailable", "actual": "empty_diagonal"},
        )

    diag_mass = float(np.nanmean(diag))
    if not np.isfinite(diag_mass):
        return (
            float("nan"),
            matrix,
            {"code": "diag_mass_unavailable", "actual": "nonfinite_diag_mass"},
        )

    return diag_mass, matrix, None


# AnalysisMixin class containing all methods for analysis operations
class AnalysisMixin:
    """Methods for MSM analysis and build operations."""

    def _load_analysis_from_entry(
            self, entry: Dict[str, Any]
    ) -> Optional[BuildArtifact]:
        bundle_path = self._path_from_value(entry.get("bundle"))
        if bundle_path is None:
            return None
        br = self._load_build_result_from_path(bundle_path)
        if br is None:
            return None
        dataset_hash = str(entry.get("dataset_hash", "")) or (
            str(getattr(br.metadata, "dataset_hash", "")) if br.metadata else ""
        )
        created_at = str(entry.get("created_at", "")) or _timestamp()
        debug_dir_raw = entry.get("debug_dir")
        debug_dir = self._path_from_value(debug_dir_raw)
        debug_summary = entry.get("debug_summary")
        if debug_summary is None and debug_dir:
            summary_name = entry.get("debug_summary_file") or "summary.json"
            summary_path = debug_dir / summary_name
            if summary_path.exists():
                try:
                    debug_summary = json.loads(summary_path.read_text(encoding="utf-8"))
                except Exception:
                    logger.debug("Failed to load analysis summary from %s", summary_path)
        fingerprint = entry.get("discretizer_fingerprint")
        if fingerprint is None and isinstance(br.flags, Mapping):
            fingerprint = br.flags.get("discretizer_fingerprint")
        tau_frames = entry.get("tau_frames")
        if tau_frames is None and isinstance(br.flags, Mapping):
            tau_frames = br.flags.get("analysis_tau_frames")
        effective_tau_frames = entry.get("effective_tau_frames")
        if effective_tau_frames is None and isinstance(br.flags, Mapping):
            effective_tau_frames = br.flags.get("analysis_effective_tau_frames")
        effective_stride_max = entry.get("effective_stride_max")
        if effective_stride_max is None and isinstance(br.flags, Mapping):
            effective_stride_max = br.flags.get("analysis_effective_stride_max")
        msm_n_states = entry.get("analysis_msm_n_states")
        if msm_n_states is None and isinstance(br.flags, Mapping):
            msm_n_states = br.flags.get("analysis_msm_n_states")
        return BuildArtifact(
            bundle_path=bundle_path.resolve(),
            dataset_hash=dataset_hash,
            build_result=br,
            created_at=created_at,
            debug_dir=debug_dir,
            debug_summary=debug_summary,
            discretizer_fingerprint=fingerprint,
            tau_frames=int(tau_frames) if tau_frames is not None else None,
            effective_tau_frames=(
                int(effective_tau_frames)
                if effective_tau_frames is not None
                else None
            ),
            effective_stride_max=(
                int(effective_stride_max) if effective_stride_max is not None else None
            ),
            analysis_msm_n_states=(
                int(msm_n_states) if msm_n_states is not None else None
            ),
        )

    def delete_analysis_bundle(self, index: int) -> bool:
        """Delete an analysis bundle and its associated files."""
        entry = self.state.remove_build(index)
        if entry is None:
            return False

        try:
            # Delete bundle file
            bundle_path = self._path_from_value(entry.get("bundle"))
            if bundle_path is not None and bundle_path.exists():
                bundle_path.unlink()
            debug_dir = self._path_from_value(entry.get("debug_dir"))
            if debug_dir is not None and debug_dir.exists() and debug_dir.is_dir():
                shutil.rmtree(debug_dir)

            return True
        except Exception:
            return False

    def load_analysis_bundle(self, index: int) -> Optional[BuildArtifact]:
        if index < 0 or index >= len(self.state.builds):
            return None
        entry = dict(self.state.builds[index])
        return self._load_analysis_from_entry(entry)

    def list_builds(self) -> List[Dict[str, Any]]:
        return [dict(entry) for entry in self.state.builds]

    def build_config_from_entry(self, entry: Dict[str, Any]) -> BuildConfig:
        """Reconstruct a BuildConfig from a state entry."""
        return BuildConfig(
            lag=int(entry.get("lag", 10)),
            bins=dict(entry.get("bins", {})),
            seed=int(entry.get("seed", 0)),
            temperature=float(entry.get("temperature", 300.0)),
            learn_cv=bool(entry.get("learn_cv", False)),
            apply_cv_whitening=bool(entry.get("apply_cv_whitening", True)),
            cluster_mode=str(entry.get("cluster_mode", "kmeans")),
            n_microstates=int(entry.get("n_microstates", 100)),
            reweight_mode=str(entry.get("reweight_mode", "MBAR")),
            fes_method=str(entry.get("fes_method", "kde")),
            fes_bandwidth=entry.get("fes_bandwidth", "scott"),
            fes_min_count_per_bin=int(entry.get("fes_min_count_per_bin", 1)),
        )

    def _extract_debug_data_from_build_result(
            self,
            br: Any,
            dataset: Mapping[str, Any],
            lag: int,
            stride_values: list[int],
            stride_map: dict,
            preview_truncated: list,
    ) -> tuple[Any, _AnalysisMSMStats]:
        """Extract debug data from the build result after discretization."""

        # Extract MSM data from build result
        msm_obj = getattr(br, "msm", None)

        # Debug logging to see what we have
        logger.info("[DEBUG] msm_obj type: %s", type(msm_obj))
        if msm_obj is not None:
            if isinstance(msm_obj, Mapping):
                logger.info("[DEBUG] msm_obj keys: %s", list(msm_obj.keys()))
                logger.info("[DEBUG] msm_obj counts shape: %s",
                           np.asarray(msm_obj.get("counts", [])).shape if msm_obj.get("counts") is not None else "None")
                logger.info("[DEBUG] msm_obj state_counts shape: %s",
                           np.asarray(msm_obj.get("state_counts", [])).shape if msm_obj.get("state_counts") is not None else "None")
                logger.info("[DEBUG] msm_obj dtrajs type: %s, length: %s",
                           type(msm_obj.get("dtrajs")),
                           len(msm_obj.get("dtrajs", [])) if msm_obj.get("dtrajs") is not None else "None")

        if msm_obj is None or not isinstance(msm_obj, Mapping):
            # No MSM data available
            shards_meta = dataset.get("__shards__", [])
            n_frames = sum(int(s.get("length", 0)) for s in shards_meta if isinstance(s, Mapping))
            debug_data = AnalysisDebugData(
                summary={
                    "tau_frames": int(lag),
                    "count_mode": "sliding",
                    "total_frames_declared": n_frames,
                    "total_frames_with_states": 0,
                    "total_pairs": 0,
                    "counts_shape": [0, 0],
                    "zero_rows": 0,
                    "states_observed": 0,
                    "effective_stride_max": max(stride_values) if stride_values else 1,
                    "warnings": [],
                },
                counts=np.zeros((0, 0), dtype=float),
                state_counts=np.zeros((0,), dtype=float),
                component_labels=np.zeros((0,), dtype=int),
            )
            stats = _AnalysisMSMStats(
                total_pairs=0,
                zero_rows=0,
                largest_scc_fraction=None,
                diag_mass=float("nan"),
            )
            return debug_data, stats

        # Extract counts and state_counts from MSM dict
        counts = np.asarray(msm_obj.get("counts", np.zeros((0, 0), dtype=float)), dtype=float)
        state_counts = np.asarray(msm_obj.get("state_counts", np.zeros((0,), dtype=float)), dtype=float)
        total_pairs = analysis_total_pairs(msm_obj, counts)

        # Extract shard metadata
        shards_meta = dataset.get("__shards__", [])
        n_frames = sum(int(s.get("length", 0)) for s in shards_meta if isinstance(s, Mapping))
        n_frames_with_states = int(state_counts.sum())

        # Compute zero rows
        zero_rows = int(np.count_nonzero(counts.sum(axis=1) == 0)) if counts.size > 0 else 0

        # Compute connected components
        components, component_labels = _strongly_connected_components(counts)
        largest_size = max((len(comp) for comp in components), default=0)
        largest_indices = max(components, key=len) if components else []
        largest_cover = _coverage_fraction(state_counts, largest_indices)

        # Compute diagonal mass
        diag_mass_val = _transition_diag_mass(counts)

        # Identify isolated states (not in the largest SCC)
        n_components = len(components)
        is_fully_connected = (n_components == 1) if n_components > 0 else False
        isolated_states: list[int] = []
        if len(components) > 1:
            # Find the largest component index
            largest_comp_idx = max(range(len(components)), key=lambda i: len(components[i]))
            # Find all states not in the largest component
            for state_id in range(len(component_labels)):
                if component_labels[state_id] != largest_comp_idx:
                    isolated_states.append(int(state_id))

        # Compute dwell time statistics from dtrajs if available
        dwell_stats = {}
        occupancy_tail = {}

        # First try to get dtrajs from MSM payload (where they're stored after clustering)
        dtrajs_raw = msm_obj.get("dtrajs")
        if dtrajs_raw is None:
            # Fall back to dataset dtrajs (pre-clustering)
            dtrajs_raw = dataset.get("dtrajs")

        if dtrajs_raw is not None:
            try:
                # Coerce dtrajs to list of numpy arrays
                from pmarlo.analysis.debug_export import _coerce_dtrajs, _infer_n_states
                dtrajs = _coerce_dtrajs(dtrajs_raw)
                if dtrajs and any(d.size > 0 for d in dtrajs):
                    n_states = _infer_n_states(dtrajs)
                    # Compute dwell time statistics
                    dwell_stats = _compute_dwell_times(dtrajs, n_states)
                    # Compute occupancy tail (lowest-occupancy states)
                    occupancy_tail = _compute_occupancy_tail(state_counts, top_k=10)
            except Exception as e:
                logger.warning("Failed to compute dwell/occupancy statistics: %s", e)

        # Build summary
        stride_max = max(stride_values) if stride_values else 1
        effective_tau_frames = int(lag * stride_max) if lag > 0 else 0

        warnings: list[dict[str, Any]] = []
        if total_pairs < 5000:
            warnings.append({
                "code": "TOTAL_PAIRS_LT_5000",
                "message": f"Too few (t, t+tau) pairs for reliable MSM (observed {total_pairs}, requires >=5000)."
            })
        if zero_rows > 0:
            warnings.append({
                "code": "ZERO_ROW_STATES_PRESENT",
                "message": "States with zero outgoing counts detected before regularisation."
            })

        summary = {
            "tau_frames": int(lag),
            "count_mode": "sliding",
            "total_frames_declared": int(n_frames),
            "total_frames_with_states": int(n_frames_with_states),
            "total_pairs": int(total_pairs),
            "counts_shape": [int(counts.shape[0]), int(counts.shape[1])],
            "zero_rows": int(zero_rows),
            "states_observed": int(np.count_nonzero(state_counts)),
            "largest_scc_size": int(largest_size),
            "largest_scc_frame_fraction": float(largest_cover) if largest_cover is not None else None,
            "component_sizes": [int(len(comp)) for comp in components],
            "n_components": int(n_components),
            "is_fully_connected": bool(is_fully_connected),
            "isolated_states": list(isolated_states),
            "expected_pairs": 0,  # Not available in this context
            "counted_pairs": int(total_pairs),
            "effective_stride_max": int(stride_max),
            "effective_strides": stride_values,
            "effective_stride_map": stride_map,
            "preview_truncated": preview_truncated,
            "effective_tau_frames": effective_tau_frames,
            "diag_mass": float(diag_mass_val),
            "warnings": warnings,
        }

        stats = _AnalysisMSMStats(
            total_pairs=int(total_pairs),
            zero_rows=int(zero_rows),
            largest_scc_fraction=(
                float(largest_cover) if largest_cover is not None else None
            ),
            diag_mass=float(diag_mass_val),
        )

        return (
            AnalysisDebugData(
                summary=summary,
                counts=counts,
                state_counts=state_counts,
                component_labels=component_labels,
            ),
            stats,
        )

    def build_analysis(
            self,
            shard_jsons: Sequence[Path],
            config: BuildConfig,
    ) -> BuildArtifact:
        print(
            f"--- DEBUG: backend.build_analysis called with {len(shard_jsons)} shards ---"
        )
        try:
            shards = [Path(p).resolve() for p in shard_jsons]
            if not shards:
                raise ValueError("No shards selected for analysis")
            stamp = _timestamp()
            bundle_path = self.layout.bundles_dir / f"bundle-{stamp}.pbz"
            dataset = load_shards_as_dataset(shards)

            # Log basic dataset info before building
            dataset_frames: int | None = None
            dataset_shard_count: int | None = None
            if isinstance(dataset, Mapping):
                dataset_shard_count = len(dataset.get("__shards__", []))
                if "X" in dataset:
                    try:
                        dataset_frames = int(len(dataset["X"]))
                    except TypeError:
                        dataset_frames = None

            logger.info(
                "[ANALYSIS_DEBUG] Pre-build config: lag=%d shard_count=%d dataset_frames=%s dataset_shards=%s",
                int(config.lag),
                len(shards),
                dataset_frames if dataset_frames is not None else "unknown",
                dataset_shard_count if dataset_shard_count is not None else "unknown",
            )

            config_payload = asdict(config)
            analysis_notes = dict(config.notes or {})
            if config.learn_cv and "model_dir" not in analysis_notes:
                analysis_notes["model_dir"] = str(self.layout.models_dir)
            analysis_notes["apply_cv_whitening_requested"] = bool(config.apply_cv_whitening)
            analysis_notes["apply_cv_whitening_enforced"] = True
            analysis_notes["kmeans_kwargs"] = dict(config.kmeans_kwargs)
            analysis_notes["analysis_overrides"] = {
                "cluster_mode": str(config.cluster_mode),
                "n_microstates": int(config.n_microstates),
                "reweight_mode": str(config.reweight_mode),
                "fes_method": str(config.fes_method),
                "fes_bandwidth": config.fes_bandwidth,
                "fes_min_count_per_bin": int(config.fes_min_count_per_bin),
            }
            requested_fingerprint = {
                "mode": str(config.cluster_mode),
                "n_states": int(config.n_microstates),
                "seed": int(config.seed),
            }
            previous_fingerprint = analysis_notes.get("discretizer_fingerprint")
            if previous_fingerprint and previous_fingerprint != requested_fingerprint:
                analysis_notes.setdefault(
                    "discretizer_fingerprint_previous", previous_fingerprint
                )
                logger.info(
                    "Discretizer fingerprint override changed from %s to %s; "
                    "forcing refit of clusterer.",
                    previous_fingerprint,
                    requested_fingerprint,
                )
            analysis_notes["discretizer_fingerprint_requested"] = requested_fingerprint
            analysis_notes["analysis_tau_requested"] = int(config.lag)

            br, ds_hash = build_from_shards(
                shard_jsons=shards,
                out_bundle=bundle_path,
                bins=dict(config.bins),
                lag=int(config.lag),
                seed=int(config.seed),
                temperature=float(config.temperature),
                learn_cv=bool(config.learn_cv),
                deeptica_params=config.deeptica_params,
                notes=analysis_notes,
                kmeans_kwargs=dict(config.kmeans_kwargs),
                n_microstates=int(config.n_microstates),
            )

            def _safe_int(value: Any, default: int) -> int:
                if value is None:
                    raise ValueError(
                        "Missing integer value; received None and fallback to"
                        f" {default!r} is not permitted."
                    )
                try:
                    return int(value)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Invalid integer value {value!r}; cannot fall back to {default!r}."
                    ) from exc

            def _safe_float(value: Any, default: float = float("nan")) -> float:
                if value is None:
                    raise ValueError(
                        "Missing float value; received None and fallback to"
                        f" {default!r} is not permitted."
                    )
                try:
                    return float(value)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Invalid float value {value!r}; cannot fall back to {default!r}."
                    ) from exc

            # Extract metadata from the dataset shards (not from discretization yet)
            shards_meta = dataset.get("__shards__", []) if isinstance(dataset, Mapping) else []
            stride_values = []
            stride_map = {}
            preview_truncated = []
            for idx, shard_entry in enumerate(shards_meta):
                if not isinstance(shard_entry, Mapping):
                    continue
                eff_stride = shard_entry.get("effective_frame_stride")
                if eff_stride is not None and eff_stride > 0:
                    stride_values.append(int(eff_stride))
                    shard_id = shard_entry.get("id", str(idx))
                    stride_map[str(shard_id)] = int(eff_stride)
                if shard_entry.get("preview_truncated"):
                    preview_truncated.append(str(shard_entry.get("id", idx)))

            stride_max = max(stride_values) if stride_values else 1
            tau_frames = int(config.lag)
            effective_tau_frames = tau_frames * stride_max if tau_frames > 0 else 0
            expected_effective_tau = effective_tau_frames  # Expected based on stride

            # Log stride information for diagnostics
            if stride_max > 1:
                logger.info(
                    "Stride-adjusted tau: tau_frames=%d (frame indices), "
                    "stride_max=%d, effective_tau_frames=%d (physical frames)",
                    tau_frames,
                    stride_max,
                    effective_tau_frames,
                )

            # Now that discretization is complete, compute debug data from the build result
            debug_data, msm_stats = self._extract_debug_data_from_build_result(
                br, dataset, config.lag, stride_values, stride_map, preview_truncated
            )

            # Extract actual statistics from the MSM build result (post-clustering)
            total_pairs_val = int(msm_stats.total_pairs)
            zero_rows_val = int(msm_stats.zero_rows)
            largest_cover = msm_stats.largest_scc_fraction
            diag_mass_val = float(msm_stats.diag_mass)

            transition_matrix_attr = getattr(br, "transition_matrix", None)
            diag_mass_val, transition_matrix_array, diag_guardrail = (
                compute_analysis_diag_mass(transition_matrix_attr)
            )
            msm_n_states_actual: Optional[int] = None

            actual_seed = int(config.seed)
            if getattr(br.metadata, "seed", None) is not None:
                try:
                    actual_seed = int(br.metadata.seed)  # type: ignore[arg-type]
                except Exception:
                    actual_seed = int(config.seed)
            fingerprint = {
                "mode": str(config.cluster_mode),
                "n_states": int(config.n_microstates),
                "seed": actual_seed,
            }
            msm_obj = getattr(br, "msm", None)
            feature_schema_payload: Dict[str, Any] | None = None
            if isinstance(msm_obj, Mapping):
                schema_candidate = msm_obj.get("feature_schema")
            else:
                schema_candidate = getattr(msm_obj, "feature_schema", None)
            if isinstance(schema_candidate, Mapping):
                feature_schema_payload = {
                    "names": list(schema_candidate.get("names", [])),
                    "n_features": int(schema_candidate.get("n_features", 0)),
                }
                fingerprint["feature_schema"] = feature_schema_payload

            fingerprint_compare = {
                "mode": fingerprint.get("mode"),
                "n_states": fingerprint.get("n_states"),
                "seed": fingerprint.get("seed"),
            }
            fingerprint_changed = fingerprint_compare != requested_fingerprint

            # Guardrail checks based on post-clustering statistics
            # Note: total_pairs and zero_rows checks are removed because they require
            # post-clustering data which we'll validate from the build result instead
            guardrail_violations: List[Dict[str, Any]] = []

            # Extract CK test results from build result
            ck_max_error = float('inf')
            ck_pass = False
            ck_threshold = 0.05  # 5% RMS error threshold

            msm_obj = getattr(br, "msm", None)
            if msm_obj is not None:
                # Try to extract CK metrics from MSM object
                if isinstance(msm_obj, Mapping):
                    ck_max_error = float(msm_obj.get("ck_max_error", float('inf')))
                    ck_pass = bool(msm_obj.get("ck_pass", False))
                    ck_threshold = float(msm_obj.get("ck_threshold", 0.05))
                else:
                    ck_max_error = float(getattr(msm_obj, "ck_max_error", float('inf')))
                    ck_pass = bool(getattr(msm_obj, "ck_pass", False))
                    ck_threshold = float(getattr(msm_obj, "ck_threshold", 0.05))

                # Add CK failure to guardrail violations
                if not ck_pass and ck_max_error != float('inf'):
                    guardrail_violations.append(
                        {
                            "code": "ck_test_failed",
                            "message": f"Chapman-Kolmogorov test failed: max_error={ck_max_error:.4f} exceeds threshold={ck_threshold:.4f}",
                            "max_error": float(ck_max_error),
                            "threshold": float(ck_threshold),
                        }
                    )
                    logger.warning(
                        "CK test failed: max_error=%.4f > threshold=%.4f",
                        ck_max_error,
                        ck_threshold,
                    )
                elif ck_pass:
                    logger.info(
                        "CK test passed: max_error=%.4f < threshold=%.4f",
                        ck_max_error,
                        ck_threshold,
                    )

            if total_pairs_val == 0:
                guardrail_violations.append(
                    {
                        "code": "no_transition_pairs",
                        "message": "no transition pairs after filtering",
                    }
                )

            if diag_guardrail is not None:
                guardrail_violations.append(diag_guardrail)

            if transition_matrix_array is None:
                guardrail_violations.append(
                    {"code": "msm_build_failed", "actual": "no_transition_matrix"}
                )
            else:
                msm_n_states_actual = int(transition_matrix_array.shape[0])
                if msm_n_states_actual == 0:
                    guardrail_violations.append(
                        {"code": "no_states_in_msm", "actual": 0}
                    )
            declared_states_raw = fingerprint.get("n_states")
            try:
                declared_states = (
                    int(declared_states_raw)
                    if declared_states_raw is not None
                    else None
                )
            except (TypeError, ValueError):
                declared_states = None
                logger.error(
                    "Invalid discretizer fingerprint n_states value %r", declared_states_raw
                )
            if (
                    msm_n_states_actual is not None
                    and declared_states is not None
                    and msm_n_states_actual != declared_states
            ):
                guardrail_violations.append(
                    {
                        "code": "state_count_mismatch",
                        "message": (
                            f"state_count_mismatch: declared={declared_states}, "
                            f"actual={msm_n_states_actual}"
                        ),
                        "declared": int(declared_states),
                        "actual": int(msm_n_states_actual),
                    }
                )

            if effective_tau_frames != expected_effective_tau:
                logger.warning(
                    "Effective tau mismatch: expected=%d, actual=%d",
                    expected_effective_tau,
                    effective_tau_frames,
                )
                # Don't treat tau mismatch as a hard failure

            # SCC Guardrail: MSM must be fully connected (require fraction = 1.0)
            # Check if the MSM is fully connected by looking at component information
            is_fully_connected = debug_data.summary.get("is_fully_connected", False)
            isolated_states = debug_data.summary.get("isolated_states", [])
            n_components = debug_data.summary.get("n_components", 0)

            # Allow override via config, but default to requiring full connectivity
            require_full_connectivity = config_payload.get(
                "require_fully_connected_msm", True
            )

            if require_full_connectivity and not is_fully_connected:
                # Provide clearer error messages depending on the situation
                if n_components == 0:
                    # This shouldn't happen with the new checks, but keep as fallback
                    logger.error(
                        "MSM construction failed: no valid states detected in discrete trajectories. "
                        "Check clustering configuration and data quality."
                    )
                    error_msg = (
                        "No valid states detected in discrete trajectories. "
                        "This indicates clustering failed or produced no valid state assignments. "
                        "Check your data quality and clustering configuration."
                    )
                elif isolated_states:
                    logger.error(
                        "MSM is not fully connected: %d isolated state(s) detected: %s",
                        len(isolated_states),
                        isolated_states,
                    )
                    error_msg = (
                        f"MSM has {n_components} strongly connected components "
                        f"(must be 1 for fully connected). "
                        f"Isolated states: {isolated_states}"
                    )
                else:
                    logger.error(
                        "MSM is not fully connected: %d components detected",
                        n_components,
                    )
                    error_msg = (
                        f"MSM has {n_components} strongly connected components "
                        f"(must be 1 for fully connected). "
                        f"Isolated states: {isolated_states}"
                    )

                guardrail_violations.append(
                    {
                        "code": "msm_not_fully_connected",
                        "message": error_msg,
                        "n_components": int(n_components),
                        "isolated_states": isolated_states,
                        "is_fully_connected": False,
                    }
                )

            analysis_healthy = not guardrail_violations

            summary_overrides = {
                "fingerprint": fingerprint,
                "analysis_guardrail_violations": guardrail_violations,
                "analysis_expected_effective_tau_frames": expected_effective_tau,
                "analysis_healthy": analysis_healthy,
                "discretizer_fingerprint_changed": bool(fingerprint_changed),
            }
            if msm_n_states_actual is not None:
                summary_overrides["analysis_msm_n_states"] = int(msm_n_states_actual)

            debug_dir = (self.layout.analysis_debug_dir / f"analysis-{stamp}").resolve()
            export_info = export_analysis_debug(
                output_dir=debug_dir,
                build_result=br,
                debug_data=debug_data,
                config=config_payload,
                dataset_hash=ds_hash,
                summary_overrides=summary_overrides,
                fingerprint=fingerprint,
            )

            for idx, shard in enumerate(debug_data.summary.get("shards", [])):
                shard_id = shard.get("id", f"shard-{idx}")
                frames_loaded = shard.get("frames_loaded", shard.get("length"))
                frames_declared = shard.get("frames_declared", shard.get("length"))
                stride_val = shard.get("effective_frame_stride")
                logger.info(
                    "Shard %s: loaded=%s declared=%s stride=%s",
                    shard_id,
                    frames_loaded,
                    frames_declared,
                    stride_val,
                )
                if (
                        shard.get("first_timestamp") is not None
                        or shard.get("last_timestamp") is not None
                ):
                    logger.info(
                        "Shard %s timestamps: first=%s last=%s",
                        shard_id,
                        shard.get("first_timestamp"),
                        shard.get("last_timestamp"),
                    )

            if not analysis_healthy:
                summary_path = Path(export_info["summary"]).resolve()
                raise ValueError(
                    "Analysis guardrails failed: "
                    f"{guardrail_violations}. "
                    f"See {summary_path} for details."
                )

            logger.info(
                "Analysis lag requested=%d, applied=%d, effective_tau=%d (max stride=%d, stride values=%s)",
                int(config.lag),
                tau_frames,
                effective_tau_frames,
                stride_max,
                stride_values,
            )
            if fingerprint_changed:
                logger.info(
                    "Effective discretizer fingerprint differs from request: %s (requested %s)",
                    fingerprint,
                    requested_fingerprint,
                )
            if tau_frames != int(config.lag):
                logger.warning(
                    "Analysis lag mismatch: requested %d frames, applied %d frames",
                    int(config.lag),
                    tau_frames,
                )
            analysis_notes["discretizer_fingerprint"] = fingerprint
            analysis_notes["analysis_total_pairs"] = int(total_pairs_val)
            analysis_notes["analysis_zero_rows"] = int(zero_rows_val)
            analysis_notes["analysis_largest_scc_fraction"] = (
                float(largest_cover) if largest_cover is not None else None
            )
            analysis_notes["analysis_diag_mass"] = float(diag_mass_val)
            if msm_n_states_actual is not None:
                analysis_notes["analysis_msm_n_states"] = int(msm_n_states_actual)
            analysis_notes["analysis_tau_frames"] = tau_frames
            analysis_notes["analysis_effective_tau_frames"] = effective_tau_frames
            analysis_notes["analysis_effective_stride_max"] = stride_max
            analysis_notes["analysis_effective_stride_values"] = stride_values
            analysis_notes["analysis_effective_stride_map"] = stride_map
            analysis_notes["analysis_expected_effective_tau_frames"] = expected_effective_tau
            analysis_notes["analysis_healthy"] = analysis_healthy
            # Persist CK test metrics
            analysis_notes["analysis_ck_max_error"] = float(ck_max_error) if ck_max_error != float('inf') else None
            analysis_notes["analysis_ck_pass"] = bool(ck_pass)
            analysis_notes["analysis_ck_threshold"] = float(ck_threshold)
            if preview_truncated:
                analysis_notes["analysis_preview_truncated"] = preview_truncated
            analysis_notes["analysis_guardrail_violations"] = guardrail_violations
            analysis_notes["discretizer_fingerprint_changed"] = bool(
                fingerprint_changed
            )
            analysis_notes["analysis_kmeans_kwargs"] = dict(config.kmeans_kwargs)
            analysis_notes.pop("discretizer_fingerprint_requested", None)

            try:
                flags = dict(br.flags or {})
            except Exception:
                flags = {}
            flags["discretizer_fingerprint"] = fingerprint
            flags["discretizer_fingerprint_changed"] = bool(fingerprint_changed)
            flags["analysis_requested_tau_frames"] = int(config.lag)
            flags["analysis_total_pairs"] = int(total_pairs_val)
            flags["analysis_zero_rows"] = int(zero_rows_val)
            flags["analysis_largest_scc_fraction"] = (
                float(largest_cover) if largest_cover is not None else None
            )
            flags["analysis_diag_mass"] = float(diag_mass_val)
            if msm_n_states_actual is not None:
                flags["analysis_msm_n_states"] = int(msm_n_states_actual)
            flags["analysis_tau_frames"] = int(tau_frames)
            flags["analysis_effective_tau_frames"] = int(effective_tau_frames)
            flags["analysis_expected_effective_tau_frames"] = int(
                expected_effective_tau
            )
            flags["analysis_effective_stride_max"] = int(stride_max)
            flags["analysis_healthy"] = analysis_healthy
            flags["analysis_guardrail_violations"] = guardrail_violations
            flags["analysis_kmeans_kwargs"] = dict(config.kmeans_kwargs)
            if stride_values:
                flags["analysis_effective_stride_values"] = list(stride_values)
            if stride_map:
                flags["analysis_effective_stride_map"] = stride_map
            if preview_truncated:
                flags["analysis_preview_truncated"] = list(preview_truncated)
            if tau_frames != int(config.lag):
                flags["analysis_tau_mismatch"] = {
                    "requested": int(config.lag),
                    "actual": int(tau_frames),
                }
            overrides = {
                "cluster_mode": str(config.cluster_mode),
                "n_microstates": int(config.n_microstates),
                "reweight_mode": str(config.reweight_mode),
                "fes_method": str(config.fes_method),
                "fes_bandwidth": config.fes_bandwidth,
                "fes_min_count_per_bin": int(config.fes_min_count_per_bin),
                "apply_whitening": bool(config.apply_cv_whitening),
                "kmeans_kwargs": dict(config.kmeans_kwargs),
            }
            flags.setdefault("analysis_overrides", overrides)
            flags.setdefault("analysis_reweight_mode", str(config.reweight_mode))
            flags.setdefault("analysis_apply_whitening", bool(config.apply_cv_whitening))
            warning_count = len(debug_data.summary.get("warnings", []))
            flags.setdefault("analysis_debug_warning_count", warning_count)
            if warning_count:
                flags.setdefault(
                    "analysis_debug_warnings",
                    _sanitize_artifacts(debug_data.summary.get("warnings")),
                )
            br.flags = flags  # type: ignore[assignment]
            try:
                artifacts = dict(br.artifacts or {})
                artifacts["analysis_debug"] = {
                    "directory": str(debug_dir),
                    "summary": str(Path(export_info["summary"]).name),
                    "arrays": export_info.get("arrays", {}),
                }
                artifacts["analysis_discretizer_fingerprint"] = fingerprint
                artifacts["analysis_tau_frames"] = int(tau_frames)
                artifacts["analysis_effective_tau_frames"] = int(effective_tau_frames)
                artifacts["analysis_effective_stride_max"] = int(stride_max)
                if stride_map:
                    artifacts["analysis_effective_stride_map"] = stride_map
                if preview_truncated:
                    artifacts["analysis_preview_truncated"] = list(
                        preview_truncated
                    )
                br.artifacts = artifacts  # type: ignore[assignment]
            except Exception:
                logger.debug(
                    "Failed to attach analysis debug artifacts", exc_info=True
                )

            artifact = BuildArtifact(
                bundle_path=bundle_path.resolve(),
                dataset_hash=ds_hash,
                build_result=br,
                created_at=stamp,
                debug_dir=debug_dir,
                debug_summary=debug_data.summary,
                discretizer_fingerprint=fingerprint,
                tau_frames=int(tau_frames),
                effective_tau_frames=int(effective_tau_frames),
                effective_stride_max=int(stride_max),
                analysis_msm_n_states=(
                    int(msm_n_states_actual)
                    if msm_n_states_actual is not None
                    else None
                ),
                analysis_healthy=analysis_healthy,
                guardrail_violations=guardrail_violations or None,
            )
            self.state.append_build(
                {
                    "bundle": str(bundle_path.resolve()),
                    "dataset_hash": ds_hash,
                    "lag": int(config.lag),
                    "bins": dict(config.bins),
                    "seed": int(config.seed),
                    "temperature": float(config.temperature),
                    "learn_cv": bool(config.learn_cv),
                    "deeptica_params": (
                        _sanitize_artifacts(config.deeptica_params)
                        if config.deeptica_params
                        else None
                    ),
                    "created_at": stamp,
                    "flags": _sanitize_artifacts(br.flags),
                    "mlcv": _sanitize_artifacts(
                        br.artifacts.get("mlcv_deeptica", {})
                    ),
                    "apply_cv_whitening": bool(config.apply_cv_whitening),
                    "cluster_mode": str(config.cluster_mode),
                    "n_microstates": int(config.n_microstates),
                    "kmeans_kwargs": _sanitize_artifacts(config.kmeans_kwargs),
                    "reweight_mode": str(config.reweight_mode),
                    "fes_method": str(config.fes_method),
                    "fes_bandwidth": config.fes_bandwidth,
                    "fes_min_count_per_bin": int(
                        config.fes_min_count_per_bin
                    ),
                    "debug_dir": str(debug_dir),
                    "debug_summary": _sanitize_artifacts(debug_data.summary),
                    "analysis_msm_n_states": (
                        int(msm_n_states_actual)
                        if msm_n_states_actual is not None
                        else None
                    ),
                    "debug_summary_file": str(Path(export_info["summary"]).name),
                    "discretizer_fingerprint": _sanitize_artifacts(
                        fingerprint
                    ),
                    "discretizer_fingerprint_changed": bool(
                        fingerprint_changed
                    ),
                    "tau_frames": int(tau_frames),
                    "effective_tau_frames": int(effective_tau_frames),
                    "effective_stride_max": int(stride_max),
                    "effective_stride_values": list(stride_values),
                    "effective_stride_map": _sanitize_artifacts(stride_map),
                    "preview_truncated": list(preview_truncated),
                    "analysis_healthy": bool(analysis_healthy),
                    "guardrail_violations": _sanitize_artifacts(
                        guardrail_violations
                    ),
                    "total_pairs": int(total_pairs_val),
                    "zero_rows": int(zero_rows_val),
                    "largest_scc_fraction": (
                        float(largest_cover) if largest_cover is not None else None
                    ),
                    "diag_mass": float(diag_mass_val),
                }
            )
            print("--- DEBUG: backend.build_analysis finished successfully ---")

            # Log metrics to structured text files
            try:
                metrics_logger = MetricsLogger(self.layout.logs_dir)
                metrics_dir = metrics_logger.log_msm_fes_build(artifact)
                logger.info(f"Metrics logged to: {metrics_dir}")
            except Exception as e:
                logger.warning(f"Failed to log metrics: {e}", exc_info=True)

            return artifact

        except Exception as e:
            import traceback

            print("--- DEBUG: ERROR INSIDE backend.build_analysis ---")
            print(f"Error Type: {type(e)}")
            print(f"Error Details: {e}")
            print("Traceback:")
            traceback.print_exc()
            print("--- END DEBUG ERROR ---")
            raise
