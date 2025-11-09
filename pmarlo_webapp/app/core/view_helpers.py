from pathlib import Path
from typing import List, Dict, Sequence, Optional, Mapping, Any, TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
import streamlit as st

from pmarlo.data.shard_io import ShardRunSummary, summarize_shard_runs
from plots import plot_msm, plot_fes
from backend.utils import _sanitize_artifacts
from core.tables import _metrics_table

if TYPE_CHECKING:
    from backend.workspace import WorkflowBackend
    from backend.layout import WorkspaceLayout
    from backend.types import BuildArtifact, TrainingResult, ConformationsResult

SHARD_SELECTOR_HELP = "M CV-BIASED = DeepTICA/metabias, U UNBIASED = Regular MD"


def _aggregate_shard_selector_stats(
    records: Sequence[Mapping[str, Any]],
    selected_run_ids: Sequence[str],
) -> Dict[str, int]:
    """Summarize selected vs total runs, shards, and frames for the selector UI."""

    selected = {str(run_id) for run_id in selected_run_ids}
    stats = {
        "runs_total": len(records),
        "runs_selected": 0,
        "shards_total": 0,
        "shards_selected": 0,
        "frames_total": 0,
        "frames_selected": 0,
    }
    for record in records:
        run_id = str(record.get("run_id", ""))
        n_shards = int(record.get("n_shards", 0) or 0)
        frames_total = int(record.get("frames_total", 0) or 0)
        stats["shards_total"] += n_shards
        stats["frames_total"] += frames_total
        if run_id in selected:
            stats["runs_selected"] += 1
            stats["shards_selected"] += n_shards
            stats["frames_selected"] += frames_total
    return stats


def _format_run_selection_summary(runs: Sequence[ShardRunSummary]) -> str:
    if not runs:
        return ""
    parts: List[str] = []
    for run in runs:
        count = int(run.shard_count)
        shard_label = "shard" if count == 1 else "shards"
        parts.append(
            f"{run.run_id}: {count} {shard_label} @ {run.temperature_K:.1f} K"
        )
    return "; ".join(parts)


def _summarize_selected_shards(
    shard_paths: Sequence[Path],
) -> tuple[List[ShardRunSummary], str]:
    if not shard_paths:
        return [], ""
    summaries = summarize_shard_runs(shard_paths)
    return summaries, _format_run_selection_summary(summaries)


def build_shard_selector_options(
    entries: Sequence[Mapping[str, Any]],
) -> tuple[List[str], Dict[str, str]]:
    """Construct display labels for shard selectors with bias markers and frame counts."""

    options: List[str] = []
    run_id_map: Dict[str, str] = {}
    for entry in entries:
        run_id = str(entry.get("run_id"))
        is_cv_informed = bool(entry.get("cv_informed", False))
        bias_tag = "[CV-BIASED]" if is_cv_informed else "[UNBIASED]"
        bias_marker = "M" if is_cv_informed else "U"
        frames_raw = entry.get("n_frames")
        if frames_raw is None:
            raise ValueError(
                f"Shard group for run '{run_id}' is missing 'n_frames' metadata."
            )
        try:
            frames_total = int(frames_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Shard group for run '{run_id}' has invalid 'n_frames': {frames_raw!r}"
            ) from exc
        frames_text = f"{frames_total:,}"
        display_label = f"{bias_marker} {run_id} {bias_tag}  FRAMES: {frames_text}"
        options.append(display_label)
        run_id_map[display_label] = run_id

    return options, run_id_map


def summarize_selected_feature_profiles(
    shard_groups: Sequence[Mapping[str, Any]],
    selected_run_ids: Sequence[Any],
) -> Dict[str, Any]:
    """Collect feature profile metadata for the selected shard groups."""

    selected_lookup = {str(run_id).strip() for run_id in selected_run_ids if str(run_id).strip()}
    selected_entries: List[Mapping[str, Any]] = []
    for entry in shard_groups:
        run_id = str(entry.get("run_id", "")).strip()
        if run_id and run_id in selected_lookup:
            selected_entries.append(entry)

    def _profile_name(entry: Mapping[str, Any]) -> str:
        raw = str(entry.get("feature_profile") or "").strip()
        if raw:
            return raw
        feature_type = str(entry.get("feature_type") or "").strip().lower()
        return "molecular_cv_biasing" if feature_type == "molecular" else "cv_analysis"

    profiles = { _profile_name(entry) for entry in selected_entries }
    feature_types = {
        (str(entry.get("feature_type") or "cv")).strip() or "cv" for entry in selected_entries
    }
    cv_flags = {bool(entry.get("cv_biasing_compatible")) for entry in selected_entries}

    return {
        "entries": selected_entries,
        "profiles": profiles,
        "feature_types": feature_types,
        "cv_flags": cv_flags,
        "primary_profile": next(iter(profiles)) if len(profiles) == 1 else None,
    }


def render_shard_selection_table(
    label: str,
    shard_groups: Sequence[Mapping[str, Any]],
    *,
    state_key: str,
    default_behavior: Literal["latest", "all", "none"] = "latest",
    help_text: Optional[str] = None,
) -> List[str]:
    """Render a fast checkbox-based shard selector that keeps its state visible."""

    if not shard_groups:
        return []

    if default_behavior not in {"latest", "all", "none"}:
        raise ValueError(
            f"Unsupported default_behavior '{default_behavior}'. "
            "Use one of: 'latest', 'all', 'none'."
        )

    ordered_run_ids: List[str] = []
    for entry in shard_groups:
        run_id_raw = entry.get("run_id")
        if run_id_raw is None:
            raise ValueError("Shard group entry is missing 'run_id'.")
        run_id = str(run_id_raw).strip()
        if not run_id:
            raise ValueError("Shard group entry has an empty 'run_id'.")
        ordered_run_ids.append(run_id)

    stored_selection = [
        run_id
        for run_id in st.session_state.get(state_key, [])
        if run_id in ordered_run_ids
    ]
    if not stored_selection:
        if default_behavior == "all":
            stored_selection = ordered_run_ids[:]
        elif default_behavior == "latest" and ordered_run_ids:
            stored_selection = [ordered_run_ids[-1]]
        else:
            stored_selection = []

    prefix = f"{state_key}__"
    active_checkbox_keys = {f"{prefix}{run_id}" for run_id in ordered_run_ids}
    stale_keys = [
        key
        for key in list(st.session_state.keys())
        if key.startswith(prefix) and key not in active_checkbox_keys
    ]
    for key in stale_keys:
        st.session_state.pop(key, None)

    def _set_selection(new_selection: Sequence[str]) -> None:
        nonlocal stored_selection
        normalized: List[str] = []
        seen: set[str] = set()
        for run_id in new_selection:
            if run_id not in ordered_run_ids or run_id in seen:
                continue
            normalized.append(run_id)
            seen.add(run_id)
        stored_selection = normalized
        st.session_state[state_key] = normalized
        for run_id in ordered_run_ids:
            st.session_state[f"{prefix}{run_id}"] = run_id in seen

    _set_selection(stored_selection)

    def _normalize_frames(entry: Mapping[str, Any], run_id: str) -> int:
        frames_raw = entry.get("n_frames")
        if frames_raw is None:
            raise ValueError(
                f"Shard group for run '{run_id}' is missing 'n_frames' metadata."
            )
        try:
            return int(frames_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Shard group for run '{run_id}' has invalid 'n_frames': {frames_raw!r}"
            ) from exc

    def _normalize_shards(entry: Mapping[str, Any]) -> int:
        n_shards_raw = entry.get("n_shards")
        if n_shards_raw is None:
            n_shards_raw = len(entry.get("paths", []))
        try:
            return int(n_shards_raw)
        except (TypeError, ValueError):
            return len(entry.get("paths", []))

    def _normalize_temperature(entry: Mapping[str, Any], run_id: str) -> Optional[float]:
        temp_raw = entry.get("temperature_K")
        if temp_raw is None:
            temp_raw = entry.get("temperature")
        if temp_raw is None:
            return None
        try:
            return float(temp_raw)
        except (TypeError, ValueError) as exc:  # pragma: no cover - sanity guard
            raise ValueError(
                f"Shard group for run '{run_id}' has invalid temperature: {temp_raw!r}"
            ) from exc

    def _normalize_analysis_temps(entry: Mapping[str, Any]) -> List[str]:
        payload = entry.get("analysis_temperatures")
        if payload is None:
            return []
        if isinstance(payload, (list, tuple, set)):
            iterable = payload
        else:
            iterable = [payload]
        values: List[str] = []
        for item in iterable:
            if item is None:
                continue
            values.append(str(item))
        return values

    st.markdown(f"**{label}**")
    display_help = help_text or SHARD_SELECTOR_HELP
    if display_help:
        st.caption(display_help)

    records: List[Dict[str, Any]] = []
    original_positions: Dict[str, int] = {
        run_id: idx for idx, run_id in enumerate(ordered_run_ids)
    }
    for run_id, entry in zip(ordered_run_ids, shard_groups):
        frames_total = _normalize_frames(entry, run_id)
        n_shards = _normalize_shards(entry)
        temperature = _normalize_temperature(entry, run_id)
        created_at = entry.get("created_at") or entry.get("created") or ""
        analysis_temps = _normalize_analysis_temps(entry)
        frames_per_shard = int(frames_total / n_shards) if n_shards else None
        bias_flag = bool(entry.get("cv_informed"))
        bias_label = "CV-BIASED" if bias_flag else "UNBIASED"
        bias_marker = "M" if bias_flag else "U"
        search_blob = " ".join(
            part
            for part in [
                run_id.lower(),
                bias_label.lower(),
                bias_marker.lower(),
                str(frames_total),
                str(n_shards),
                str(temperature or ""),
                (created_at or "").lower(),
                " ".join(t.lower() for t in analysis_temps),
            ]
            if part
        )
        records.append(
            {
                "run_id": run_id,
                "frames_total": frames_total,
                "frames_per_shard": frames_per_shard,
                "n_shards": n_shards,
                "temperature": temperature,
                "created_at": created_at,
                "analysis_temperatures": analysis_temps,
                "bias_label": bias_label,
                "bias_marker": bias_marker,
                "cv_informed": bias_flag,
                "search_blob": search_blob,
                "position": original_positions[run_id],
            }
        )

    filter_col, sort_col = st.columns([3, 1])
    filter_query = filter_col.text_input(
        "Filter shard groups",
        key=f"{state_key}_filter",
        placeholder="Search by run id, bias, temperature, or date...",
    ).strip()
    sort_choice = sort_col.selectbox(
        "Sort order",
        options=(
            "Original order",
            "Newest",
            "Oldest",
            "Most frames",
            "Fewest frames",
            "Run ID",
        ),
        key=f"{state_key}_sort",
    )

    def _filter_records(
        items: Sequence[Mapping[str, Any]]
    ) -> List[Mapping[str, Any]]:
        if not filter_query:
            return list(items)
        needle = filter_query.lower()
        return [
            record for record in items if needle in record.get("search_blob", "")
        ]

    def _sort_records(items: Sequence[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
        sort_map = {
            "Original order": (lambda rec: rec["position"], False),
            "Newest": (lambda rec: rec.get("created_at") or "", True),
            "Oldest": (lambda rec: rec.get("created_at") or "", False),
            "Most frames": (lambda rec: rec.get("frames_total", 0), True),
            "Fewest frames": (lambda rec: rec.get("frames_total", 0), False),
            "Run ID": (lambda rec: rec.get("run_id", ""), False),
        }
        key_fn, reverse = sort_map.get(sort_choice, sort_map["Original order"])
        return sorted(items, key=key_fn, reverse=reverse)

    action_cols = st.columns(3)
    if action_cols[0].button(
        "Select all",
        key=f"{state_key}_select_all",
        use_container_width=True,
    ):
        _set_selection(ordered_run_ids)
    if action_cols[1].button(
        "Clear selection",
        key=f"{state_key}_clear",
        use_container_width=True,
    ):
        _set_selection([])
    latest_disabled = not ordered_run_ids
    if action_cols[2].button(
        "Latest only",
        key=f"{state_key}_latest",
        use_container_width=True,
        disabled=latest_disabled,
    ):
        _set_selection(ordered_run_ids[-1:] if ordered_run_ids else [])

    stats = _aggregate_shard_selector_stats(records, stored_selection)
    summary_cols = st.columns(3)
    summary_cols[0].markdown(
        f"**Runs selected**\n{stats['runs_selected']} / {stats['runs_total']}"
    )
    summary_cols[1].markdown(
        f"**Shards selected**\n{stats['shards_selected']} / {stats['shards_total']}"
    )
    summary_cols[2].markdown(
        f"**Frames selected**\n"
        f"{stats['frames_selected']:,} / {stats['frames_total']:,}"
    )

    display_records = _sort_records(_filter_records(records))
    st.divider()
    if not display_records:
        st.info("No shard groups match the current filter.")
    else:
        for idx, record in enumerate(display_records):
            run_id = str(record["run_id"])
            checkbox_key = f"{prefix}{run_id}"
            select_col, info_col = st.columns([0.12, 0.88])
            st.checkbox(
                "Select run",
                key=checkbox_key,
                label_visibility="collapsed",
            )
            info_col.markdown(
                f"**{run_id}**  "
                f"`{record['bias_marker']} {record['bias_label']}`"
            )
            summary_line = (
                f"{record['n_shards']} shard{'s' if record['n_shards'] != 1 else ''} | "
                f"{record['frames_total']:,} frame{'s' if record['frames_total'] != 1 else ''}"
            )
            if record["frames_per_shard"]:
                summary_line += (
                    f" | ~{record['frames_per_shard']:,} frames/shard"
                )
            info_col.caption(summary_line)

            detail_parts = []
            temperature = record.get("temperature")
            if temperature is not None:
                detail_parts.append(f"{float(temperature):.1f} K")
            created_at = record.get("created_at")
            if created_at:
                detail_parts.append(f"created {created_at}")
            analysis_temps = record.get("analysis_temperatures") or []
            if analysis_temps:
                detail_parts.append(
                    "analysis temps: " + ", ".join(str(t) for t in analysis_temps)
                )
            if detail_parts:
                info_col.caption(" | ".join(detail_parts))

            if idx < len(display_records) - 1:
                st.divider()

    selected_runs = [
        run_id
        for run_id in ordered_run_ids
        if st.session_state.get(f"{prefix}{run_id}", False)
    ]
    st.session_state[state_key] = selected_runs
    return selected_runs


def _default_feature_spec_path(layout: "WorkspaceLayout") -> Optional[Path]:
    """Return the packaged feature specification if it exists."""

    candidate = (layout.app_root / "app" / "feature_spec.yaml").resolve()
    if candidate.exists():
        return candidate
    return None

def _infer_default_topology(
    backend: "WorkflowBackend",
    layout: "WorkspaceLayout",
    run_ids: Sequence[str],
) -> Optional[Path]:
    """Heuristically locate a topology file for the selected shard runs."""

    target_ids = {str(run_id) for run_id in run_ids if run_id}
    for entry in reversed(list(backend.state.runs)):
        run_id = str(entry.get("run_id"))
        if target_ids and run_id not in target_ids:
            continue
        candidate = entry.get("pdb") or entry.get("topology")
        if not candidate:
            continue
        path = Path(candidate).expanduser().resolve()
        if path.exists():
            return path

    inputs = layout.available_inputs()
    if inputs:
        return inputs[0]
    return None

def _model_entry_label(entry: Mapping[str, Any], idx: int) -> str:
    bundle_raw = entry.get("bundle", "")
    bundle_name = Path(bundle_raw).name if bundle_raw else f"model-{idx}"
    created = entry.get("created_at", "unknown")
    return f"{bundle_name} (created {created})"

def _show_build_outputs(artifact: "BuildArtifact | TrainingResult") -> None:
    br = artifact.build_result
    col1, col2 = st.columns(2)
    with col1:
        T = br.transition_matrix
        pi = br.stationary_distribution
        msm_states = getattr(artifact, "analysis_msm_n_states", None)
        if msm_states is None and T is not None:
            try:
                msm_states = int(np.asarray(T).shape[0])
            except Exception:
                msm_states = None
        metadata_obj = getattr(br, "metadata", None)
        msm_run_id: Optional[str] = None
        if metadata_obj is not None:
            raw_run_id = getattr(metadata_obj, "run_id", None)
            if raw_run_id is not None:
                msm_run_id = str(raw_run_id)
        fig = plot_msm(T, pi, msm_n_states=msm_states, msm_run_id=msm_run_id)
        st.pyplot(fig, clear_figure=True, width="stretch")
    with col2:
        fig = plot_fes(br.fes)
        st.pyplot(fig, clear_figure=True, width="stretch")

    flags: Dict[str, Any] = br.flags or {}

    def _safe_int(value: Any) -> Optional[int]:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    tau_frames = _safe_int(getattr(artifact, "tau_frames", None))
    if tau_frames is None:
        tau_frames = _safe_int(flags.get("analysis_tau_frames"))
    effective_tau = _safe_int(getattr(artifact, "effective_tau_frames", None))
    if effective_tau is None:
        effective_tau = _safe_int(flags.get("analysis_effective_tau_frames"))
    stride_max = _safe_int(getattr(artifact, "effective_stride_max", None))
    if stride_max is None:
        stride_max = _safe_int(flags.get("analysis_effective_stride_max"))
    fingerprint = getattr(artifact, "discretizer_fingerprint", None) or flags.get(
        "discretizer_fingerprint"
    )
    fingerprint_changed = bool(flags.get("discretizer_fingerprint_changed"))
    tau_mismatch = flags.get("analysis_tau_mismatch")
    if not isinstance(tau_mismatch, Mapping):
        tau_mismatch = None
    preview_truncated = flags.get("analysis_preview_truncated") or []
    stride_map = flags.get("analysis_effective_stride_map") or {}
    guardrail_info = (
        flags.get("analysis_guardrail_violations")
        or getattr(artifact, "guardrail_violations", None)
        or []
    )
    analysis_healthy_flag = flags.get("analysis_healthy")
    if analysis_healthy_flag is None:
        analysis_healthy_flag = getattr(artifact, "analysis_healthy", None)

    meta_cols = st.columns(4)
    meta_cols[0].metric("Shards", int(br.n_shards))
    meta_cols[1].metric("Frames", int(br.n_frames))
    meta_cols[2].metric("Features", len(br.feature_names))
    if tau_frames is not None:
        delta_label = None
        if tau_mismatch is not None:
            delta_label = f"req {tau_mismatch.get('requested')}"
        meta_cols[3].metric("tau (frames)", tau_frames, delta=delta_label)
    else:
        meta_cols[3].metric("tau (frames)", "n/a")

    if flags:
        st.dataframe(_metrics_table(br.flags), width="stretch")

    if tau_mismatch is not None:
        st.warning(
            f"Requested tau={tau_mismatch.get('requested')} frames but build used {tau_mismatch.get('actual')}.",
        )

    if fingerprint_changed:
        st.info("Discretizer fingerprint differs from requested configuration; cached mappings were invalidated.")

    if stride_max is not None and stride_max > 1:
        info_msg = (
            f"Effective tau={effective_tau}" if effective_tau is not None else f"Max stride={stride_max}"
        )
        st.info(f"Detected subsampling in shards (max stride={stride_max}). {info_msg}.")

    if analysis_healthy_flag is not None:
        status_text = "healthy" if analysis_healthy_flag else "violations detected"
        st.caption(f"Analysis health: {status_text}")
    if guardrail_info:
        st.warning("Guardrail violations detected:")
        st.json(guardrail_info)

    if preview_truncated:
        formatted = ", ".join(str(item) for item in preview_truncated)
        st.warning(
            "Some shards appear truncated relative to their metadata: "
            f"{formatted}"
        )

    debug_summary = getattr(artifact, "debug_summary", None)
    if debug_summary:
        with st.expander("Analysis metadata", expanded=False):
            total_pairs = debug_summary.get("total_pairs", "n/a")
            zero_rows = debug_summary.get("zero_rows", "n/a")
            st.write(
                f"Total (t, t+tau) pairs: {total_pairs} | Zero rows: {zero_rows}"
            )
            stride_values = debug_summary.get("effective_strides") or []
            if stride_values:
                st.write(f"Effective strides detected: {stride_values}")
            effective_tau_summary = debug_summary.get("effective_tau_frames")
            if effective_tau_summary is not None:
                st.write(f"Effective tau (stride-adjusted): {effective_tau_summary}")
            if stride_map:
                st.write("Per-shard stride map:")
                st.json(stride_map)
            first_ts = debug_summary.get("first_timestamps") or []
            last_ts = debug_summary.get("last_timestamps") or []
            if first_ts or last_ts:
                first_repr = ", ".join(str(val) for val in first_ts) if first_ts else "n/a"
                last_repr = ", ".join(str(val) for val in last_ts) if last_ts else "n/a"
                st.write(f"First timestamps: {first_repr} | Last timestamps: {last_repr}")
            if fingerprint:
                st.write("Discretizer fingerprint:")
                st.json(fingerprint)
            else:
                st.write("Discretizer fingerprint: unavailable")
    elif fingerprint:
        st.caption("Discretizer fingerprint")
        st.json(fingerprint)

    if br.messages:
        st.write("Messages:")
        for msg in br.messages:
            st.write(f"- {msg}")

def _render_deeptica_summary(summary: Dict[str, object]) -> None:
    cleaned = _sanitize_artifacts(summary)
    st.caption("Deep-TICA summary")
    if not isinstance(cleaned, dict):
        st.json(cleaned)
        return
    st.json(cleaned)
    with st.expander("Training diagnostics", expanded=False):
        epochs = list(range(1, len(cleaned.get("val_score_curve", [])) + 1))
        if epochs:
            df_score = pd.DataFrame(
                {"val_score": cleaned.get("val_score_curve", [])}, index=epochs
            )
            st.line_chart(df_score, height=200)
        var_z0 = cleaned.get("var_z0_curve") or []
        if var_z0:
            df_var_z0 = pd.DataFrame(var_z0)
            df_var_z0.index = list(range(1, len(df_var_z0) + 1))
            df_var_z0.columns = [f"z0_{i+1}" for i in range(df_var_z0.shape[1])]
            st.line_chart(df_var_z0, height=200)
        var_zt = cleaned.get("var_zt_curve") or []
        if var_zt:
            df_var_zt = pd.DataFrame(var_zt)
            df_var_zt.index = list(range(1, len(df_var_zt) + 1))
            df_var_zt.columns = [f"zt_{i+1}" for i in range(df_var_zt.shape[1])]
            st.line_chart(df_var_zt, height=200)
        cond_data: Dict[str, object] = {}
        if cleaned.get("cond_c00_curve"):
            cond_data["cond_C00"] = cleaned.get("cond_c00_curve")
        if cleaned.get("cond_ctt_curve"):
            cond_data["cond_Ctt"] = cleaned.get("cond_ctt_curve")
        if cond_data:
            df_cond = pd.DataFrame(cond_data)
            st.line_chart(df_cond, height=200)
        if cleaned.get("grad_norm_curve"):
            df_grad = pd.DataFrame({"grad_norm": cleaned.get("grad_norm_curve")})
            st.line_chart(df_grad, height=200)


def _render_conformations_result(conf_result: "ConformationsResult") -> None:
    """Visualise stored or freshly computed conformations analysis results."""

    if conf_result.error:
        st.error(f"Conformations analysis failed: {conf_result.error}")
        st.info(f"Output directory: {conf_result.output_dir}")
        return

    if not conf_result.tpt_converged:
        iteration_count = (
            conf_result.tpt_pathway_iterations
            or conf_result.tpt_summary.get("pathway_iterations")
            or conf_result.tpt_pathway_max_iterations
            or conf_result.tpt_summary.get("pathway_max_iterations")
        )
        if iteration_count is not None:
            st.error(
                "Warning: TPT calculation failed to converge after "
                f"{iteration_count} iterations. Pathway results are unreliable. "
                "Try increasing the number of clusters or adjusting the MSM lag time."
            )
        else:
            st.error(
                "Warning: TPT calculation failed to converge. Pathway results are unreliable. "
                "Try increasing the number of clusters or adjusting the MSM lag time."
            )

    if conf_result.tpt_summary:
        st.subheader("TPT Results")
        cols = st.columns(4)
        cols[0].metric(
            "Rate", f"{conf_result.tpt_summary.get('rate', float('nan')):.3e}"
        )
        cols[1].metric(
            "MFPT", f"{conf_result.tpt_summary.get('mfpt', float('nan')):.1f}"
        )
        cols[2].metric(
            "Total Flux",
            f"{conf_result.tpt_summary.get('total_flux', float('nan')):.3e}",
        )
        cols[3].metric(
            "N Pathways", conf_result.tpt_summary.get("n_pathways", "n/a")
        )

    if conf_result.metastable_states:
        st.subheader("Metastable States")
        meta_df_data = []
        for state_id, state_data in conf_result.metastable_states.items():
            meta_df_data.append(
                {
                    "State": state_id,
                    "Population": f"{float(state_data.get('population', 0.0)):.4f}",
                    "N States": state_data.get("n_states", 0),
                    "PDB": (
                        Path(state_data.get("representative_pdb", "")).name
                        if state_data.get("representative_pdb")
                        else "N/A"
                    ),
                }
            )
        st.dataframe(pd.DataFrame(meta_df_data), width="stretch")

    if conf_result.transition_states:
        st.subheader("Transition States")
        ts_df_data = []
        for ts_data in conf_result.transition_states:
            ts_df_data.append(
                {
                    "State Index": ts_data.get("state_index", ""),
                    "Committor": f"{float(ts_data.get('committor', 0.0)):.3f}",
                    "PDB": (
                        Path(ts_data.get("representative_pdb", "")).name
                        if ts_data.get("representative_pdb")
                        else "N/A"
                    ),
                }
            )
        st.dataframe(pd.DataFrame(ts_df_data), width="stretch")

    if conf_result.pathways:
        st.subheader("Dominant Pathways")
        st.write(conf_result.pathways)

    if conf_result.plots:
        st.subheader("Visualizations")
        plot_cols = st.columns(2)
        plot_idx = 0
        for plot_name, plot_path in conf_result.plots.items():
            if Path(plot_path).exists():
                with plot_cols[plot_idx % 2]:
                    st.image(
                        str(plot_path),
                        caption=plot_name.replace("_", " ").title(),
                    )
                plot_idx += 1

    st.info(f"All conformations saved to: {conf_result.output_dir}")
    if conf_result.representative_pdbs:
        st.write(
            f"{len(conf_result.representative_pdbs)} representative PDB files saved"
        )


def plot_ck_errors_with_threshold(
    ck_errors: Dict[int, float],
    selected_lag: int,
    threshold: float = 0.15,
):
    """Plot CK errors vs lag time with threshold line and selected lag highlighted.

    Parameters
    ----------
    ck_errors : Dict[int, float]
        CK error for each candidate lag.
    selected_lag : int
        The selected optimal lag time.
    threshold : float
        CK error threshold to display.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """
    import matplotlib.pyplot as plt

    lags = sorted(ck_errors.keys())
    errors = [ck_errors[lag] for lag in lags]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot CK errors
    ax.plot(lags, errors, marker="o", linestyle="-", linewidth=2, label="CK Error")

    # Highlight selected lag
    if selected_lag in ck_errors:
        selected_error = ck_errors[selected_lag]
        ax.scatter(
            [selected_lag],
            [selected_error],
            color="red",
            s=200,
            zorder=5,
            label=f"Selected: τ={selected_lag}",
            marker="*",
        )

    # Threshold line
    ax.axhline(
        y=threshold,
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Threshold ({threshold:.0%})",
    )

    ax.set_xlabel("Lag time τ (steps)", fontsize=12)
    ax.set_ylabel("CK Error", fontsize=12)
    ax.set_title("Chapman-Kolmogorov Test Error vs Lag Time", fontsize=14, fontweight="bold")
    ax.legend(loc="best", frameon=True)
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.tight_layout()

    return fig


def plot_its_with_selection(
    lag_times: np.ndarray,
    timescales: np.ndarray,
    selected_lag: int,
    max_timescales: int = 10,
):
    """Plot implied timescales with selected lag highlighted.

    Parameters
    ----------
    lag_times : np.ndarray
        Array of lag times.
    timescales : np.ndarray
        Array of timescales (shape: [n_lags, n_timescales]).
    selected_lag : int
        The selected optimal lag time.
    max_timescales : int
        Maximum number of timescale curves to plot.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """
    import matplotlib.pyplot as plt

    lags = np.asarray(lag_times, dtype=int)
    ts = np.asarray(timescales, dtype=float)

    if ts.ndim == 1:
        ts = ts.reshape(-1, 1)

    n_curves = min(ts.shape[1], max_timescales)

    fig, ax = plt.subplots(figsize=(10, 6))

    for idx in range(n_curves):
        y_vals = ts[:, idx]
        valid_mask = np.isfinite(y_vals) & (y_vals > 0)
        ax.plot(
            lags[valid_mask],
            y_vals[valid_mask],
            marker="o",
            linestyle="-",
            label=f"Timescale {idx + 1}",
        )

    # Highlight selected lag with vertical line
    if selected_lag in lags:
        ax.axvline(
            x=selected_lag,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Selected: τ={selected_lag}",
        )

    ax.set_xlabel("Lag time τ (steps)", fontsize=12)
    ax.set_ylabel("Implied timescale (steps)", fontsize=12)
    ax.set_title("Implied Timescales with Selected Lag", fontsize=14, fontweight="bold")
    ax.set_yscale("log")
    ax.legend(loc="best", frameon=True, fontsize=9)
    ax.grid(True, which="both", linestyle=":", alpha=0.6, linewidth=0.5)
    fig.tight_layout()

    return fig
