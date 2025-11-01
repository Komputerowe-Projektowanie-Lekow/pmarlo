from pathlib import Path
from typing import List, Dict, Sequence, Optional, Mapping, Any, TYPE_CHECKING

import numpy as np
import pandas as pd
import streamlit as st

from pmarlo.data.shard_io import ShardRunSummary, summarize_shard_runs
from app.plots import plot_msm, plot_fes
from app.backend.utils import _sanitize_artifacts
from app.core.tables import _metrics_table

if TYPE_CHECKING:
    from app.backend.workspace import WorkflowBackend
    from app.backend.layout import WorkspaceLayout
    from app.backend.types import BuildArtifact, TrainingResult, ConformationsResult



def _select_shard_paths(
    groups: Sequence[Dict[str, object]], run_ids: Sequence[str]
) -> List[Path]:
    lookup: Dict[str, Sequence[str]] = {
        str(entry.get("run_id")): entry.get("paths", [])  # type: ignore[dict-item]
        for entry in groups
    }
    selected: List[Path] = []
    for run_id in run_ids:
        paths = lookup.get(run_id, [])
        for p in paths:
            selected.append(Path(p))
    return selected


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
