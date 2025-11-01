from __future__ import annotations

"""Utilities for emitting detailed debugging artifacts for MSM analysis builds."""

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Sequence, Tuple, cast

import numpy as np

from pmarlo.markov_state_model.free_energy import FESResult
from pmarlo.utils.coercion import coerce_finite_float
from pmarlo.utils.path_utils import ensure_directory

from .counting import expected_pairs
from .errors import CountingLogicError

__all__ = [
    "AnalysisDebugData",
    "CountingLogicError",
    "compute_analysis_debug",
    "export_analysis_debug",
    "total_pairs_from_shards",
]


@dataclass
class AnalysisDebugData:
    """Container for dataset-level debug information gathered prior to MSM build."""

    summary: Dict[str, Any]
    counts: np.ndarray
    state_counts: np.ndarray
    component_labels: np.ndarray

    def to_summary_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable summary dictionary."""
        payload = dict(self.summary)
        payload["component_labels"] = self.component_labels.astype(int).tolist()
        payload["state_counts"] = self.state_counts.astype(float).tolist()
        payload["counts_nonzero"] = int(np.count_nonzero(self.counts))
        payload["counts_density"] = (
            float(payload["counts_nonzero"] / float(self.counts.size))
            if self.counts.size
            else 0.0
        )
        return payload


def total_pairs_from_shards(
    lengths: Sequence[int],
    tau: int,
    *,
    count_mode: str = "sliding",
) -> int:
    """Predict total (t, t+tau) pairs from shard lengths.

    Parameters
    ----------
    lengths
        Effective lengths (number of frames considered for MSM counting)
        for each shard.
    tau
        Lag time in frames.
    count_mode
        Counting strategy; ``"sliding"`` matches the default sliding window.

    Returns
    -------
    int
        Predicted number of transition pairs.
    """

    if tau < 0:
        raise ValueError("tau must be non-negative")
    if count_mode not in {"sliding", "strided"}:
        raise ValueError(f"Unsupported count_mode: {count_mode}")

    stride = 1 if count_mode == "sliding" else max(1, tau)
    return expected_pairs(lengths, tau, stride)


def compute_analysis_debug(
    dataset: Mapping[str, Any],
    *,
    lag: int,
    count_mode: str = "sliding",
) -> AnalysisDebugData:
    """Compute dataset diagnostics ahead of MSM construction.

    Requires the dataset to have discrete trajectories (dtrajs) already computed.
    Raises an error if dtrajs are missing or empty.
    """

    shards_raw = _normalise_shard_info(dataset.get("__shards__", ()))
    shard_lengths = [int(entry["length"]) for entry in shards_raw]
    # total_frames from metadata - keep for reference but don't use for accounting
    total_frames_metadata = sum(shard_lengths)

    dtrajs = _coerce_dtrajs(dataset.get("dtrajs", ()))

    # Enforce that dtrajs must exist - no silent fallbacks
    if not dtrajs or all(d.size == 0 for d in dtrajs):
        raise ValueError(
            "Cannot compute analysis debug statistics: dataset has no discrete trajectories (dtrajs). "
            "The dataset must be discretized (clustered) before transition counts can be computed. "
            f"Dataset contains {total_frames_metadata} frames across {len(shard_lengths)} shards, "
            "but no state assignments are present. Run discretization first."
        )

    n_states = _infer_n_states(dtrajs)

    # Check if no states were detected at all - this is a critical error
    if n_states == 0:
        raise ValueError(
            "Cannot compute MSM statistics: no valid states detected in discrete trajectories. "
            f"Dataset contains {total_frames_metadata} frames across {len(shard_lengths)} shards, "
            f"but all discrete trajectory values are negative or empty. "
            "This may indicate clustering failed or produced invalid state assignments. "
            "Check your clustering configuration and ensure valid state labels are generated."
        )

    counts, total_pairs = _build_transition_counts(dtrajs, n_states, lag, count_mode)
    state_counts = _count_state_visits(dtrajs, n_states)
    total_frames_state = int(state_counts.sum())

    # FIX: Compute total_frames_declared from actual dtraj lengths, not shard metadata
    dtraj_lengths = [len(dtraj) for dtraj in dtrajs]
    total_frames = sum(dtraj_lengths)

    frames_declared_all = [int(entry.get("frames_declared", 0)) for entry in shards_raw]
    frames_loaded_all = [
        int(entry.get("frames_loaded", entry.get("length", 0))) for entry in shards_raw
    ]
    effective_stride_raw = [entry.get("effective_frame_stride") for entry in shards_raw]
    stride_values = [int(s) for s in effective_stride_raw if s and s > 0]
    max_stride = max(stride_values) if stride_values else 1
    preview_truncated_ids = [
        str(entry.get("id", idx))
        for idx, entry in enumerate(shards_raw)
        if entry.get("preview_truncated")
    ]

    zero_rows = _count_zero_rows(counts)
    components, component_labels = _strongly_connected_components(counts)
    largest_size = max((len(comp) for comp in components), default=0)
    largest_indices = max(components, key=len) if components else []
    largest_cover = _coverage_fraction(state_counts, largest_indices)
    diag_mass_val = _transition_diag_mass(counts)

    # Compute dwell time statistics
    dwell_stats = _compute_dwell_times(dtrajs, n_states)

    # Compute occupancy tail (lowest-occupancy states)
    occupancy_tail = _compute_occupancy_tail(state_counts, top_k=10)

    # Identify isolated states (not in the largest SCC)
    isolated_states: List[int] = []
    if len(components) > 1:
        # Find the largest component index
        largest_comp_idx = max(range(len(components)), key=lambda i: len(components[i]))
        # Find all states not in the largest component
        for state_id in range(len(component_labels)):
            if component_labels[state_id] != largest_comp_idx:
                isolated_states.append(int(state_id))

    # Calculate SCC fraction (fraction of components that are size 1 = fully connected)
    n_components = len(components)
    is_fully_connected = (n_components == 1) if n_components > 0 else False

    warnings: List[Dict[str, Any]] = []
    if total_pairs < 5000:
        warnings.append(
            {
                "code": "TOTAL_PAIRS_LT_5000",
                "message": (
                    "Too few (t, t+tau) pairs for reliable MSM "
                    f"(observed {total_pairs}, requires >=5000)."
                ),
            }
        )
    if zero_rows > 0:
        warnings.append(
            {
                "code": "ZERO_ROW_STATES_PRESENT",
                "message": (
                    "States with zero outgoing counts detected before regularisation; "
                    "prune states or lower lag to avoid singular rows."
                ),
            }
        )
    if largest_cover is not None and largest_cover < 0.9:
        warnings.append(
            {
                "code": "SCC_COVERAGE_LT_0.90",
                "message": (
                    f"Largest strongly connected component covers only "
                    f"{largest_cover:.2%} of visited frames."
                ),
            }
        )
    if stride_values and max_stride > 1:
        warnings.append(
            {
                "code": "EFFECTIVE_STRIDE_GT_1",
                "message": (
                    "Loaded shard data appears subsampled; "
                    f"effective stride values detected: {stride_values}"
                ),
            }
        )
    if preview_truncated_ids:
        warnings.append(
            {
                "code": "SHARD_PREVIEW_TRUNCATION",
                "message": (
                    "Shards truncated relative to declared frame count: "
                    f"{preview_truncated_ids}"
                ),
            }
        )

    temperatures = sorted(
        {float(entry["temperature"]) for entry in shards_raw if entry["temperature"]}
    )

    # Build complete stride map covering ALL unique shards
    # Include shards even if effective_frame_stride is None (default to 1)
    stride_map = {}
    for idx, entry in enumerate(shards_raw):
        shard_id = str(entry.get("id", str(idx)))
        stride_value = entry.get("effective_frame_stride")
        # Always include the shard in the map, even if stride is None
        stride_map[shard_id] = stride_value if stride_value is not None else 1

    first_timestamps = [
        entry.get("first_timestamp")
        for entry in shards_raw
        if entry.get("first_timestamp") is not None
    ]
    last_timestamps = [
        entry.get("last_timestamp")
        for entry in shards_raw
        if entry.get("last_timestamp") is not None
    ]
    effective_tau_frames = int(lag * max_stride) if lag > 0 else 0
    stride_for_pairs = 1 if count_mode == "sliding" else max(1, lag)
    # FIX: Compute expected_pairs from actual dtraj lengths, not shard metadata
    expected_pair_count = expected_pairs(dtraj_lengths, lag, stride_for_pairs)
    total_pairs_predicted = expected_pair_count

    # Assert that counted_pairs matches expected_pairs within one hop per segment
    # Allow tolerance of one pair per segment to account for edge cases
    tolerance = len(dtraj_lengths)
    if abs(expected_pair_count - total_pairs) > tolerance:
        raise CountingLogicError(
            f"Pair counting mismatch: counted {total_pairs} pairs but expected "
            f"{expected_pair_count} pairs based on actual dtraj lengths (tolerance: {tolerance}). "
            f"Dtraj lengths: {dtraj_lengths}, lag: {lag}, stride: {stride_for_pairs}. "
            f"This indicates a bug in the transition counting logic."
        )

    summary: Dict[str, Any] = {
        "tau_frames": int(lag),
        "count_mode": str(count_mode),
        "total_frames_declared": int(total_frames),
        "total_frames_with_states": int(total_frames_state),
        "total_pairs": int(total_pairs),
        "counts_shape": [int(counts.shape[0]), int(counts.shape[1])],
        "zero_rows": int(zero_rows),
        "states_observed": int(np.count_nonzero(state_counts)),
        "largest_scc_size": int(largest_size),
        "largest_scc_frame_fraction": (
            float(largest_cover) if largest_cover is not None else None
        ),
        "component_sizes": [int(len(comp)) for comp in components],
        "n_components": int(n_components),
        "is_fully_connected": bool(is_fully_connected),
        "isolated_states": isolated_states,
        "stride": int(stride_for_pairs),
        "expected_pairs": int(expected_pair_count),
        "counted_pairs": int(total_pairs),
        "per_shard_lengths": shard_lengths,
        "shards": shards_raw,
        "frames_declared": frames_declared_all,
        "frames_loaded": frames_loaded_all,
        "effective_strides": stride_values,
        "per_shard_effective_strides": effective_stride_raw,
        "effective_stride_map": stride_map,
        "effective_stride_max": int(max_stride),
        "preview_truncated": preview_truncated_ids,
        "effective_tau_frames": effective_tau_frames,
        "total_pairs_predicted": int(total_pairs_predicted),
        "first_timestamps": first_timestamps,
        "last_timestamps": last_timestamps,
        "temperatures": temperatures,
        "diag_mass": float(diag_mass_val),
        "warnings": warnings,
        "dwell_time_stats": dwell_stats,
        "occupancy_tail": occupancy_tail,
    }

    return AnalysisDebugData(
        summary=summary,
        counts=counts,
        state_counts=state_counts,
        component_labels=component_labels,
    )


def export_analysis_debug(
    *,
    output_dir: Path,
    build_result: Any,
    debug_data: AnalysisDebugData,
    config: Mapping[str, Any] | None,
    dataset_hash: str,
    summary_overrides: Mapping[str, Any] | None = None,
    fingerprint: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Persist debug artefacts (counts, MSM arrays, diagnostics) to disk."""

    output_dir = Path(output_dir)
    ensure_directory(output_dir)

    summary_payload = _prepare_summary_payload(
        debug_data,
        dataset_hash=dataset_hash,
        config=config,
        summary_overrides=summary_overrides,
        fingerprint=fingerprint,
    )

    if _requires_transition_artifacts(build_result):
        _ensure_nonempty_transition_statistics(
            debug_data.counts, debug_data.state_counts
        )

    arrays_written = _export_core_arrays(debug_data, output_dir)
    arrays_written.update(_export_result_arrays(build_result, output_dir))

    fes_payload = _maybe_export_fes(build_result, output_dir)
    if fes_payload:
        arrays_written.update(fes_payload)

    summary_payload["result"] = _collect_result_summary(build_result)
    summary_payload["arrays"] = arrays_written

    _write_additional_metadata(build_result, output_dir, summary_payload)

    msm_obj = getattr(build_result, "msm", None)
    assignment_arrays, assignment_splits = _maybe_export_assignments(
        msm_obj, output_dir
    )
    if assignment_arrays:
        arrays_written.update(assignment_arrays)
        summary_payload["state_assignment_splits"] = assignment_splits

    # Generate annotated plots with MSM metadata
    plots_written = _generate_annotated_plots(
        build_result=build_result,
        output_dir=output_dir,
        summary_payload=summary_payload,
        config=config,
    )
    if plots_written:
        summary_payload["plots"] = plots_written

    summary_path = _write_summary(output_dir, summary_payload)
    return {"summary": summary_path, "arrays": arrays_written, "plots": plots_written}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _prepare_summary_payload(
    debug_data: AnalysisDebugData,
    *,
    dataset_hash: str,
    config: Mapping[str, Any] | None,
    summary_overrides: Mapping[str, Any] | None,
    fingerprint: Mapping[str, Any] | None,
) -> Dict[str, Any]:
    summary_payload = debug_data.to_summary_dict()
    if summary_overrides:
        summary_payload.update({str(k): v for k, v in summary_overrides.items()})
    if fingerprint is not None:
        summary_payload.setdefault("fingerprint", fingerprint)
    summary_payload["dataset_hash"] = str(dataset_hash)
    if config is not None:
        summary_payload["config"] = config
    return summary_payload


def _export_core_arrays(
    debug_data: AnalysisDebugData, output_dir: Path
) -> Dict[str, str]:
    arrays_written: Dict[str, str] = {}
    counts_path = output_dir / "transition_counts.npy"
    np.save(counts_path, debug_data.counts)
    arrays_written["transition_counts"] = counts_path.name

    state_counts_path = output_dir / "state_counts.npy"
    np.save(state_counts_path, debug_data.state_counts)
    arrays_written["state_counts"] = state_counts_path.name

    component_labels_path = output_dir / "component_labels.npy"
    np.save(component_labels_path, debug_data.component_labels.astype(int))
    arrays_written["component_labels"] = component_labels_path.name
    return arrays_written


def _export_result_arrays(build_result: Any, output_dir: Path) -> Dict[str, str]:
    arrays_written: Dict[str, str] = {}
    transition_matrix = getattr(build_result, "transition_matrix", None)
    if transition_matrix is not None:
        tm_path = output_dir / "transition_matrix.npy"
        np.save(tm_path, np.asarray(transition_matrix, dtype=float))
        arrays_written["transition_matrix"] = tm_path.name

    stationary = getattr(build_result, "stationary_distribution", None)
    if stationary is not None:
        pi_path = output_dir / "stationary_distribution.npy"
        np.save(pi_path, np.asarray(stationary, dtype=float))
        arrays_written["stationary_distribution"] = pi_path.name

    cluster_pop = getattr(build_result, "cluster_populations", None)
    if cluster_pop is not None:
        cp_path = output_dir / "cluster_populations.npy"
        np.save(cp_path, np.asarray(cluster_pop, dtype=float))
        arrays_written["cluster_populations"] = cp_path.name
    return arrays_written


def _write_additional_metadata(
    build_result: Any, output_dir: Path, summary_payload: Dict[str, Any]
) -> None:
    diagnostics = getattr(build_result, "diagnostics", None)
    if isinstance(diagnostics, Mapping):
        diag_path = output_dir / "diagnostics.json"
        diag_path.write_text(
            json.dumps(diagnostics, cls=_AnalysisJSONEncoder, indent=2)
        )
        summary_payload["diagnostics_file"] = diag_path.name

    flags = getattr(build_result, "flags", None)
    if isinstance(flags, Mapping):
        flags_path = output_dir / "flags.json"
        flags_path.write_text(json.dumps(flags, cls=_AnalysisJSONEncoder, indent=2))
        summary_payload["flags_file"] = flags_path.name

    feature_stats = _extract_feature_stats(build_result)
    if isinstance(feature_stats, Mapping) and feature_stats:
        stats_path = output_dir / "feature_stats.json"
        stats_path.write_text(
            json.dumps(feature_stats, cls=_AnalysisJSONEncoder, indent=2)
        )
        summary_payload["feature_stats_file"] = stats_path.name


def _write_summary(output_dir: Path, summary_payload: Mapping[str, Any]) -> Path:
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary_payload, cls=_AnalysisJSONEncoder, indent=2)
    )
    return summary_path


def _normalise_shard_info(shards: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    normalised: List[Dict[str, Any]] = []
    for raw in shards:
        if not isinstance(raw, Mapping):
            continue
        start = int(raw.get("start", 0))
        stop = int(raw.get("stop", start))
        length = max(0, stop - start)
        if "id" not in raw:
            raise ValueError("Shard metadata is missing required 'id' field")
        entry = {
            "id": str(raw["id"]),
            "start": start,
            "stop": stop,
            "length": length,
            "temperature": coerce_finite_float(raw.get("temperature")),
        }
        for key in (
            "frames_declared",
            "frames_loaded",
            "effective_frame_stride",
            "preview_truncated",
            "time_metadata",
            "notes",
            "first_timestamp",
            "last_timestamp",
        ):
            if key in raw:
                entry[key] = raw[key]
        if "source" in raw:
            entry["source"] = raw["source"]
        source_path = raw.get("source_path")
        if source_path:
            try:
                entry["source_path"] = str(Path(source_path))
            except Exception:
                entry["source_path"] = str(source_path)
        run_uid = raw.get("run_uid")
        if run_uid:
            entry["run_uid"] = str(run_uid)
        normalised.append(entry)
    return normalised


def _coerce_dtrajs(
    dtrajs: (
        Iterable[Iterable[int]] | Mapping[Any, Iterable[int]] | Iterable[np.ndarray]
    ),
) -> List[np.ndarray]:
    coerced: List[np.ndarray] = []
    iterable: Iterable[Iterable[int] | np.ndarray]
    if isinstance(dtrajs, Mapping):
        iterable = cast(Iterable[Iterable[int] | np.ndarray], dtrajs.values())
    else:
        iterable = dtrajs
    for traj in iterable:
        try:
            arr = np.asarray(traj, dtype=int).reshape(-1)
        except Exception:
            continue
        coerced.append(arr)
    return coerced


def _infer_n_states(dtrajs: Sequence[np.ndarray]) -> int:
    max_state = -1
    for dtraj in dtrajs:
        if dtraj.size == 0:
            continue
        local_max = int(np.max(dtraj))
        if local_max >= 0:
            max_state = max(max_state, local_max)
    return int(max_state + 1) if max_state >= 0 else 0


def _build_transition_counts(
    dtrajs: Sequence[np.ndarray],
    n_states: int,
    lag: int,
    count_mode: str,
) -> Tuple[np.ndarray, int]:
    counts = np.zeros((n_states, n_states), dtype=float)
    if n_states == 0 or lag <= 0:
        return counts, 0

    total_pairs = 0
    step = lag if str(count_mode).lower() == "strided" else 1
    for traj in dtrajs:
        if traj.size <= lag:
            continue
        for idx in range(0, traj.size - lag, step):
            a = int(traj[idx])
            b = int(traj[idx + lag])
            if a < 0 or b < 0:
                continue
            if a >= n_states or b >= n_states:
                continue
            counts[a, b] += 1.0
            total_pairs += 1
    return counts, total_pairs


def _count_state_visits(dtrajs: Sequence[np.ndarray], n_states: int) -> np.ndarray:
    visits = np.zeros((n_states,), dtype=int)
    if n_states == 0:
        return visits
    for traj in dtrajs:
        if traj.size == 0:
            continue
        mask = traj >= 0
        if not np.any(mask):
            continue
        selected = traj[mask]
        bins = np.bincount(selected, minlength=n_states)
        visits[: bins.shape[0]] += bins
    return visits


def _compute_dwell_times(
    dtrajs: Sequence[np.ndarray], n_states: int
) -> Dict[str, Any]:
    """Compute dwell time statistics for each state.

    Dwell time is the number of consecutive frames spent in a state before
    transitioning to another state.

    Returns a dictionary with per-state dwell statistics including min, max,
    mean, median, and counts of transitions.
    """
    if n_states == 0:
        return {
            "per_state_dwell_min": [],
            "per_state_dwell_max": [],
            "per_state_dwell_mean": [],
            "per_state_dwell_median": [],
            "per_state_transition_counts": [],
        }

    # Collect all dwell times for each state
    state_dwells: List[List[int]] = [[] for _ in range(n_states)]

    for traj in dtrajs:
        if traj.size == 0:
            continue

        # Filter out invalid states
        valid_mask = traj >= 0
        if not np.any(valid_mask):
            continue

        valid_traj = traj[valid_mask]
        if valid_traj.size == 0:
            continue

        # Compute run lengths (consecutive visits to same state)
        current_state = int(valid_traj[0])
        current_dwell = 1

        for i in range(1, valid_traj.size):
            state = int(valid_traj[i])
            if state == current_state:
                current_dwell += 1
            else:
                # Record the dwell time for the previous state
                if current_state < n_states:
                    state_dwells[current_state].append(current_dwell)
                current_state = state
                current_dwell = 1

        # Don't forget the last dwell
        if current_state < n_states:
            state_dwells[current_state].append(current_dwell)

    # Compute statistics for each state
    per_state_dwell_min = []
    per_state_dwell_max = []
    per_state_dwell_mean = []
    per_state_dwell_median = []
    per_state_transition_counts = []

    for state_id in range(n_states):
        dwells = state_dwells[state_id]
        if dwells:
            per_state_dwell_min.append(int(np.min(dwells)))
            per_state_dwell_max.append(int(np.max(dwells)))
            per_state_dwell_mean.append(float(np.mean(dwells)))
            per_state_dwell_median.append(float(np.median(dwells)))
            per_state_transition_counts.append(len(dwells))
        else:
            # State was never visited
            per_state_dwell_min.append(0)
            per_state_dwell_max.append(0)
            per_state_dwell_mean.append(0.0)
            per_state_dwell_median.append(0.0)
            per_state_transition_counts.append(0)

    return {
        "per_state_dwell_min": per_state_dwell_min,
        "per_state_dwell_max": per_state_dwell_max,
        "per_state_dwell_mean": per_state_dwell_mean,
        "per_state_dwell_median": per_state_dwell_median,
        "per_state_transition_counts": per_state_transition_counts,
    }


def _compute_occupancy_tail(
    state_counts: np.ndarray, top_k: int = 10
) -> Dict[str, Any]:
    """Identify states with the lowest occupancy (bottom-k states).

    Parameters
    ----------
    state_counts
        Array of visit counts per state.
    top_k
        Number of lowest-occupancy states to report.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - "lowest_occupancy_states": list of state indices
        - "lowest_occupancy_counts": list of corresponding counts
    """
    n_states = len(state_counts)
    if n_states == 0:
        return {
            "lowest_occupancy_states": [],
            "lowest_occupancy_counts": [],
        }

    # Sort states by count (ascending)
    sorted_indices = np.argsort(state_counts)

    # Take the bottom k states
    k = min(top_k, n_states)
    lowest_states = sorted_indices[:k].tolist()
    lowest_counts = state_counts[lowest_states].tolist()

    return {
        "lowest_occupancy_states": lowest_states,
        "lowest_occupancy_counts": lowest_counts,
    }


def _count_zero_rows(counts: np.ndarray) -> int:
    if counts.size == 0:
        return 0
    row_sums = counts.sum(axis=1)
    return int(np.sum(np.isclose(row_sums, 0.0)))


def _transition_diag_mass(counts: np.ndarray) -> float:
    if counts.size == 0 or counts.shape[0] == 0:
        return float("nan")
    row_sum = counts.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = np.where(row_sum == 0.0, 1.0, row_sum)
        T = counts / denom
    return float(np.trace(T) / T.shape[0]) if T.size else float("nan")


class _TarjanSCC:
    """Tarjan strongly connected components solver with minimal state."""

    def __init__(self, adjacency: Sequence[Sequence[int]]) -> None:
        self.adjacency = adjacency
        self.n = len(adjacency)
        self.index = 0
        self.indices = np.full(self.n, -1, dtype=int)
        self.lowlinks = np.zeros(self.n, dtype=int)
        self.on_stack = np.zeros(self.n, dtype=bool)
        self.stack: List[int] = []
        self.components: List[List[int]] = []
        self.labels = np.full(self.n, -1, dtype=int)

    def run(self) -> Tuple[List[List[int]], np.ndarray]:
        for v in range(self.n):
            if self.indices[v] == -1:
                self._visit(v)
        return self.components, self.labels

    def _visit(self, v: int) -> None:
        self.indices[v] = self.index
        self.lowlinks[v] = self.index
        self.index += 1
        self.stack.append(v)
        self.on_stack[v] = True

        for w in self.adjacency[v]:
            if self.indices[w] == -1:
                self._visit(w)
                self.lowlinks[v] = min(self.lowlinks[v], self.lowlinks[w])
            elif self.on_stack[w]:
                self.lowlinks[v] = min(self.lowlinks[v], self.indices[w])

        if self.lowlinks[v] == self.indices[v]:
            component: List[int] = []
            while True:
                w = self.stack.pop()
                self.on_stack[w] = False
                component.append(w)
                if w == v:
                    break
            comp_idx = len(self.components)
            for node in component:
                self.labels[node] = comp_idx
            self.components.append(component)


def _strongly_connected_components(
    counts: np.ndarray,
) -> Tuple[List[List[int]], np.ndarray]:
    n = int(counts.shape[0])
    if n == 0:
        return [], np.empty((0,), dtype=int)

    # Build adjacency list, checking if there are any transitions at all
    adjacency = [np.where(counts[i] > 0.0)[0].astype(int).tolist() for i in range(n)]

    # Check if all states are isolated (no transitions)
    total_transitions = sum(len(neighbors) for neighbors in adjacency)
    if total_transitions == 0:
        # Every state is its own component (all isolated)
        components = [[i] for i in range(n)]
        labels = np.arange(n, dtype=int)
        return components, labels

    solver = _TarjanSCC(adjacency)
    return solver.run()


def _coverage_fraction(
    state_counts: np.ndarray,
    indices: Sequence[int],
) -> float | None:
    if state_counts.size == 0:
        return None
    total = float(state_counts.sum())
    if total <= 0.0:
        return None
    in_component = float(state_counts[indices].sum()) if indices else 0.0
    return in_component / total


def _maybe_export_fes(
    build_result: Any,
    output_dir: Path,
) -> Dict[str, str]:
    fes = getattr(build_result, "fes", None)
    if fes is None:
        return {}
    payload: Dict[str, Any]
    if isinstance(fes, FESResult):
        payload = {
            "F": np.asarray(fes.F, dtype=float),
            "xedges": np.asarray(fes.xedges, dtype=float),
            "yedges": np.asarray(fes.yedges, dtype=float),
        }
        if getattr(fes, "levels_kJmol", None) is not None:
            payload["levels_kJmol"] = np.asarray(fes.levels_kJmol, dtype=float)
        meta = getattr(fes, "metadata", None)
        if meta:
            payload["metadata"] = _make_json_ready(meta)
    elif isinstance(fes, Mapping):
        payload = {key: value for key, value in fes.items() if key != "result"}
        if "result" in fes and isinstance(fes["result"], Mapping):
            payload["result"] = _make_json_ready(fes["result"])
    else:
        payload = {"raw": _make_json_ready(fes)}

    fes_path = output_dir / "fes.json"
    fes_path.write_text(json.dumps(_make_json_ready(payload), indent=2))
    return {"fes": fes_path.name}


def _collect_result_summary(build_result: Any) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "n_frames": int(getattr(build_result, "n_frames", 0)),
        "n_shards": int(getattr(build_result, "n_shards", 0)),
        "feature_names": list(getattr(build_result, "feature_names", []) or []),
        "messages": list(getattr(build_result, "messages", []) or []),
    }
    artifacts = getattr(build_result, "artifacts", None)
    if isinstance(artifacts, Mapping):
        summary["artifact_keys"] = sorted(str(k) for k in artifacts.keys())
    return summary


def _requires_transition_artifacts(build_result: Any) -> bool:
    if build_result is None:
        return False
    if _has_payload(getattr(build_result, "transition_matrix", None)):
        return True
    if _has_payload(getattr(build_result, "stationary_distribution", None)):
        return True
    if _has_payload(getattr(build_result, "cluster_populations", None)):
        return True
    return _has_payload(getattr(build_result, "fes", None))


def _has_payload(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, np.ndarray):
        return int(value.size) > 0
    if isinstance(value, Mapping):
        return bool(value)
    if isinstance(value, (list, tuple, set)):
        return bool(value)
    return True


def _ensure_nonempty_transition_statistics(
    counts: np.ndarray, state_counts: np.ndarray
) -> None:
    counts_arr = np.asarray(counts, dtype=float)
    if counts_arr.ndim != 2 or counts_arr.size == 0:
        raise ValueError(
            "Insufficient data to export MSM artefacts: transition counts array is empty."
        )
    if not np.any(counts_arr > 0.0):
        raise ValueError(
            "Insufficient data to export MSM artefacts: transition counts contain no observed transitions."
        )

    state_arr = np.asarray(state_counts, dtype=float)
    if state_arr.size == 0:
        raise ValueError(
            "Insufficient data to export MSM artefacts: state visit counts array is empty."
        )
    if not np.any(state_arr > 0.0):
        raise ValueError(
            "Insufficient data to export MSM artefacts: state visit counts contain no observations."
        )


def _extract_feature_stats(build_result: Any) -> Mapping[str, Any] | None:
    if build_result is None:
        return None

    direct = getattr(build_result, "feature_stats", None)
    if isinstance(direct, Mapping):
        return direct

    msm_obj = getattr(build_result, "msm", None)
    stats = getattr(msm_obj, "feature_stats", None)
    if isinstance(stats, Mapping):
        return stats

    artifacts = getattr(build_result, "artifacts", None)
    if isinstance(artifacts, Mapping):
        stats = artifacts.get("feature_stats")
        if isinstance(stats, Mapping):
            return stats
    return None


def _maybe_export_assignments(
    msm_obj: Any,
    output_dir: Path,
) -> Tuple[Dict[str, str], List[str]]:
    if msm_obj is None:
        return {}, []

    assignments = getattr(msm_obj, "assignments", None)
    if not isinstance(assignments, Mapping):
        return {}, []

    masks = getattr(msm_obj, "assignment_masks", None)
    if not isinstance(masks, Mapping):
        masks = {}

    written: Dict[str, str] = {}
    splits: List[str] = []

    for split_name, labels in assignments.items():
        safe_name = _sanitise_name(split_name)
        state_path = output_dir / f"state_ids_{safe_name}.npy"
        np.save(state_path, np.asarray(labels, dtype=np.int32))
        written[f"state_ids[{split_name}]"] = state_path.name
        splits.append(str(split_name))

        mask = masks.get(split_name)
        if mask is not None:
            mask_path = output_dir / f"valid_mask_{safe_name}.npy"
            np.save(mask_path, np.asarray(mask, dtype=bool))
            written[f"valid_mask[{split_name}]"] = mask_path.name

    return written, sorted(splits)


def _sanitise_name(name: str) -> str:
    safe_chars = []
    for ch in str(name):
        if ch.isalnum() or ch in ("-", "_", "."):
            safe_chars.append(ch)
        else:
            safe_chars.append("_")
    return "".join(safe_chars)


def _make_json_ready(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        return {str(k): _make_json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_make_json_ready(v) for v in obj]
    if isinstance(obj, (np.generic,)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _generate_annotated_plots(
    *,
    build_result: Any,
    output_dir: Path,
    summary_payload: Mapping[str, Any],
    config: Mapping[str, Any] | None,
) -> Dict[str, str]:
    """Generate annotated plots with MSM metadata."""
    # Placeholder implementation - extend as needed
    return {}


class _AnalysisJSONEncoder(json.JSONEncoder):
    """JSON encoder for analysis artifacts."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, Mapping):
            return dict(obj)
        return super().default(obj)

