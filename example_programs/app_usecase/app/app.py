from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence
import json
import traceback

import pandas as pd
import streamlit as st

from pmarlo.data.shard_io import ShardRunSummary, summarize_shard_runs
try:  # Prefer package-relative imports when launched via `streamlit run -m`
    from pmarlo.transform.build import _sanitize_artifacts

    from .backend import (
        BuildArtifact,
        BuildConfig,
        ConformationsConfig,
        ConformationsResult,
        ShardRequest,
        SimulationConfig,
        TrainingConfig,
        TrainingResult,
        WorkflowBackend,
        WorkspaceLayout,
    )
    from .plots import plot_fes, plot_msm
    from .plots.diagnostics import (
        format_warnings,
        plot_autocorrelation_curves,
        plot_canonical_correlations,
    )
except ImportError:  # Fallback for `streamlit run app.py`
    import sys

    _APP_DIR = Path(__file__).resolve().parent
    if str(_APP_DIR) not in sys.path:
        sys.path.insert(0, str(_APP_DIR))
    from backend import (  # type: ignore
        BuildArtifact,
        BuildConfig,
        ConformationsConfig,
        ConformationsResult,
        ShardRequest,
        SimulationConfig,
        TrainingConfig,
        TrainingResult,
        WorkflowBackend,
        WorkspaceLayout,
    )
    from plots import plot_fes, plot_msm  # type: ignore
    from plots.diagnostics import (  # type: ignore
        format_warnings,
        plot_autocorrelation_curves,
        plot_canonical_correlations,
    )

    from pmarlo.transform.build import _sanitize_artifacts


# Banner shown when Deep-TICA training is gated off by missing extras
DEEPTICA_SKIP_MESSAGE = (
    "Deep-TICA CV learning was skipped because optional dependencies are not installed."
)

# Keys used inside st.session_state
_LAST_SIM = "__pmarlo_last_simulation"
_LAST_SHARDS = "__pmarlo_last_shards"
_LAST_TRAIN = "__pmarlo_last_training"
_LAST_TRAIN_CONFIG = "__pmarlo_last_train_cfg"
_LAST_BUILD = "__pmarlo_last_build"
_RUN_PENDING = "__pmarlo_run_pending"
_TRAIN_CONFIG_PENDING = "__pmarlo_pending_train_cfg"
_TRAIN_FEEDBACK = "__pmarlo_train_feedback"


def _parse_temperature_ladder(raw: str) -> List[float]:
    cleaned = raw.replace(";", ",")
    temps: List[float] = []
    for token in cleaned.split(","):
        token = token.strip()
        if not token:
            continue
        temps.append(float(token))
    if not temps:
        raise ValueError("Provide at least one temperature in Kelvin.")
    return temps


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


def _metrics_table(flags: Dict[str, object]) -> pd.DataFrame:
    """Render build flags in a tabular, Arrow-friendly representation.

    Streamlit converts the returned DataFrame into an Arrow table. Mixed dtypes
    within a column (for example booleans alongside strings) lead Arrow to infer
    an incompatible schema which subsequently raises an ``ArrowInvalid``. The
    build flags frequently contain boolean toggles together with nested
    structures such as diagnostic warning lists, so we normalise them into a
    flat table that stores display strings alongside their original type.
    """

    from collections.abc import Mapping
    from collections.abc import Sequence as _SequenceABC

    try:  # Local import to avoid an unconditional dependency at module import
        import numpy as np  # type: ignore
    except Exception:  # pragma: no cover - NumPy is always available in practice
        np = None  # type: ignore

    def _coerce_scalar(val: object) -> object:
        if np is not None and isinstance(val, np.generic):
            return val.item()
        return val

    def _is_sequence(val: object) -> bool:
        return isinstance(val, _SequenceABC) and not isinstance(
            val, (str, bytes, bytearray)
        )

    def _iter_items(prefix: str, value: object):
        value = _coerce_scalar(value)

        if isinstance(value, Mapping):
            for sub_key, sub_val in value.items():
                next_prefix = f"{prefix}.{sub_key}" if prefix else str(sub_key)
                yield from _iter_items(next_prefix, sub_val)
            return

        if (
            np is not None
            and hasattr(value, "tolist")
            and not isinstance(value, (str, bytes, bytearray))
        ):
            try:
                value = value.tolist()
            except Exception:
                pass

        if _is_sequence(value):
            seq = list(value)
            if not seq:
                yield (prefix, "[]")
                return
            for idx, item in enumerate(seq, start=1):
                suffix = f"[{idx}]"
                next_prefix = f"{prefix}{suffix}" if prefix else suffix
                yield from _iter_items(next_prefix, item)
            return

        yield (prefix, value)

    def _format_value(val: object) -> tuple[str, str]:
        val = _coerce_scalar(val)
        if val is None:
            return "", "NoneType"
        if isinstance(val, bool):
            return ("True" if val else "False"), "bool"
        if isinstance(val, (int, float)):
            return f"{val}", type(val).__name__
        if isinstance(val, bytes):
            try:
                decoded = val.decode("utf-8")
            except Exception:
                decoded = val.decode("utf-8", errors="replace")
            return decoded, "bytes"
        return str(val), type(val).__name__

    rows: List[Dict[str, object]] = []
    for key, raw_value in flags.items():
        for metric_key, metric_value in _iter_items(key, raw_value):
            display, dtype_name = _format_value(metric_value)
            rows.append(
                {
                    "metric": metric_key,
                    "value": display,
                    "value_type": dtype_name,
                }
            )

    if not rows:
        return pd.DataFrame({"metric": [], "value": [], "value_type": []})

    df = pd.DataFrame(rows)
    df["value"] = pd.Series(df["value"], dtype="string")
    df["value_type"] = pd.Series(df["value_type"], dtype="string")
    return df


def _format_tau_schedule(values: Sequence[int]) -> str:
    if not values:
        return ""
    return ", ".join(str(int(v)) for v in values)


def _parse_tau_schedule(raw: str) -> List[int]:
    cleaned = raw.replace(";", ",")
    values: List[int] = []
    for token in cleaned.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            val = int(token)
        except ValueError as exc:  # pragma: no cover - defensive parsing
            raise ValueError(f"Invalid tau value '{token}'") from exc
        if val <= 0:
            raise ValueError("Tau values must be positive integers.")
        values.append(val)
    if not values:
        raise ValueError("Provide at least one tau value.")
    return sorted(set(values))


def _update_state(**kwargs: Any) -> None:
    for key, value in kwargs.items():
        st.session_state[key] = value


def _apply_training_config_to_state(cfg: TrainingConfig) -> None:
    bins = dict(cfg.bins or {})
    hidden_str = ", ".join(str(int(h)) for h in cfg.hidden)
    _update_state(
        train_lag=int(cfg.lag),
        train_bins_rg=int(bins.get("Rg", 64)),
        train_bins_rmsd=int(bins.get("RMSD_ref", 64)),
        train_seed=int(cfg.seed),
        train_max_epochs=int(cfg.max_epochs),
        train_patience=int(cfg.early_stopping),
        train_temperature=float(cfg.temperature),
        train_hidden_layers=hidden_str or "128,128",
        train_tau_schedule=_format_tau_schedule(cfg.tau_schedule),
        train_val_tau=int(cfg.val_tau),
        train_epochs_per_tau=int(cfg.epochs_per_tau),
    )


def _apply_analysis_config_to_state(cfg: BuildConfig) -> None:
    bins = dict(cfg.bins or {})
    if isinstance(cfg.fes_bandwidth, (int, float)):
        bw_value = f"{float(cfg.fes_bandwidth):g}"
    else:
        bw_value = str(cfg.fes_bandwidth)
    mode = str(cfg.reweight_mode)
    mode_norm = mode.upper() if mode.upper() in {"MBAR", "TRAM"} else mode.lower()
    _update_state(
        analysis_lag=int(cfg.lag),
        analysis_bins_rg=int(bins.get("Rg", 72)),
        analysis_bins_rmsd=int(bins.get("RMSD_ref", 72)),
        analysis_seed=int(cfg.seed),
        analysis_temperature=float(cfg.temperature),
        analysis_learn_cv=bool(cfg.learn_cv),
        analysis_apply_whitening=bool(cfg.apply_cv_whitening),
        analysis_cluster_mode=str(cfg.cluster_mode),
        analysis_n_microstates=int(cfg.n_microstates),
        analysis_reweight_mode=mode_norm,
        analysis_fes_method=str(cfg.fes_method),
        analysis_fes_bandwidth=bw_value,
        analysis_min_count_per_bin=int(cfg.fes_min_count_per_bin),
    )


def _show_build_outputs(artifact: BuildArtifact | TrainingResult) -> None:
    br = artifact.build_result
    col1, col2 = st.columns(2)
    with col1:
        T = br.transition_matrix
        pi = br.stationary_distribution
        fig = plot_msm(T, pi)
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


def _ensure_session_defaults() -> None:
    for key in (
        _LAST_SIM,
        _LAST_SHARDS,
        _LAST_TRAIN,
        _LAST_TRAIN_CONFIG,
        _LAST_BUILD,
    ):
        st.session_state.setdefault(key, None)
    st.session_state.setdefault(_RUN_PENDING, False)
    st.session_state.setdefault(_TRAIN_CONFIG_PENDING, None)
    st.session_state.setdefault(_TRAIN_FEEDBACK, None)
    st.session_state.setdefault("train_hidden_layers", "128,128")
    st.session_state.setdefault("train_tau_schedule", "2,5,10,20")
    st.session_state.setdefault("train_val_tau", 20)
    st.session_state.setdefault("train_epochs_per_tau", 15)
    st.session_state.setdefault("analysis_cluster_mode", "kmeans")
    st.session_state.setdefault("analysis_n_microstates", 20)
    st.session_state.setdefault("analysis_reweight_mode", "MBAR")
    st.session_state.setdefault("analysis_fes_method", "kde")
    st.session_state.setdefault("analysis_fes_bandwidth", "scott")
    st.session_state.setdefault("analysis_min_count_per_bin", 1)


def _consume_pending_training_config() -> None:
    pending = st.session_state.get(_TRAIN_CONFIG_PENDING)
    if isinstance(pending, TrainingConfig):
        _apply_training_config_to_state(pending)
    st.session_state[_TRAIN_CONFIG_PENDING] = None



def main() -> None:
    st.set_page_config(page_title="PMARLO Joint Learning", layout="wide")
    _ensure_session_defaults()
    _consume_pending_training_config()

    layout = WorkspaceLayout.from_app_package()
    backend = WorkflowBackend(layout)

    summary = backend.sidebar_summary()
    with st.sidebar:
        st.title("Workspace")
        st.caption(str(layout.workspace_dir))
        cols = st.columns(2)
        cols[0].metric("Sim runs", summary.get("runs", 0))
        cols[1].metric("Shard files", summary.get("shards", 0))
        cols = st.columns(2)
        cols[0].metric("Models", summary.get("models", 0))
        cols[1].metric("Bundles", summary.get("builds", 0))
        st.divider()
        inputs = layout.available_inputs()
        if inputs:
            st.write("Available inputs:")
            for pdb in inputs:
                st.caption(pdb.name)
        else:
            st.info("Drop prepared PDB files into app_intputs/ to get started.")

    tab_sampling, tab_training, tab_analysis, tab_model_preview, tab_assets = st.tabs(
        [
            "Sampling",
            "Model Training",
            "Analysis",
            "Model Preview",
            "Assets",
        ]
    )

    with tab_sampling:
        st.header("Sampling & Shard Production")
        inputs = layout.available_inputs()
        if not inputs:
            st.warning("No prepared proteins found. Place a PDB under app_intputs/.")
        else:
            default_index = 0
            input_choice = st.selectbox(
                "Protein input (PDB)",
                options=inputs,
                format_func=lambda p: p.name,
                index=default_index,
                key="sim_input_choice",
            )
            temps_raw = st.text_input(
                "Temperature ladder (K)",
                "300, 320, 340",
                key="sim_temperature_ladder",
            )
            steps = st.number_input(
                "Total MD steps",
                min_value=1000,
                max_value=5_000_000,
                value=50_000,
                step=5_000,
                key="sim_total_steps",
            )
            quick = st.checkbox(
                "Quick preset (short equilibration)",
                value=True,
                key="sim_quick_preset",
            )
            save_restart = st.checkbox(
                "Save last frame as restart input",
                value=True,
                help=(
                    "When enabled, the final MD frame is stored in the run directory and "
                    "copied into app_intputs/ so it becomes available as a protein input."
                ),
                key="sim_save_restart_snapshot",
            )
            random_seed_str = st.text_input(
                "Random seed (blank = auto)",
                "",
                key="sim_random_seed",
            )
            run_label = st.text_input(
                "Run label (optional)",
                "",
                key="sim_run_label",
            )

            # CV Model Selection
            st.subheader("CV-Informed Sampling (Optional)")
            models = backend.list_models()
            cv_model_path = None
            if models:
                use_cv_model = st.checkbox(
                    "Use trained CV model to inform sampling",
                    value=False,
                    help="Select a trained Deep-TICA model. Model parameters will be saved with simulation metadata for future CV-guided analysis.",
                    key="sim_use_cv_model",
                )
                if use_cv_model:
                    model_indices = list(range(len(models)))

                    def _cv_model_label(idx: int) -> str:
                        entry = models[idx]
                        bundle_raw = entry.get("bundle", "")
                        bundle_name = Path(bundle_raw).name if bundle_raw else f"model-{idx}"
                        return bundle_name

                    selected_cv_idx = st.selectbox(
                        "Select CV model",
                        options=model_indices,
                        format_func=_cv_model_label,
                        key="sim_cv_model_select",
                    )
                    # Get checkpoint_dir where exported .pt files are, not the .pbz bundle
                    model_entry = models[selected_cv_idx]
                    checkpoint_dir_str = model_entry.get("checkpoint_dir")
                    cv_model_path = Path(checkpoint_dir_str) if checkpoint_dir_str else None

                    if cv_model_path and cv_model_path.exists():
                        st.success(f"✓ Selected CV model from: {cv_model_path.name}")
                        st.info(
                            "**CV-Informed Sampling ENABLED** 🚀\n\n"
                            "The trained Deep-TICA model will be used to bias the simulation:\n"
                            "- **Bias type**: Harmonic expansion (E = k * Σ(cv²))\n"
                            "- **Effect**: Repulsive forces in CV space → explore diverse conformations\n"
                            "- **Implementation**: OpenMM computes forces via F = -∇E\n\n"
                            "⚠️ **Requirements**:\n"
                            "- `openmm-torch` must be installed (`conda install -c conda-forge openmm-torch`)\n"
                            "- CUDA-enabled PyTorch recommended (CPU is ~10-20x slower)\n\n"
                            "⚠️ **Note**: The model expects **molecular features** (distances, angles) as input. "
                            "Feature extraction is automatically configured in the OpenMM system."
                        )
                    elif cv_model_path:
                        st.error(f"⚠️ Model checkpoint directory not found: {cv_model_path}")
            else:
                st.info("No trained CV models available. Train a model in the 'Model Training' tab to enable CV-informed sampling.")

            col_extra = st.expander("Advanced options", expanded=False)
            with col_extra:
                jitter = st.checkbox(
                    "Jitter starting structure",
                    value=False,
                    key="sim_jitter_toggle",
                )
                jitter_sigma = st.number_input(
                    "Jitter sigma (Angstrom)",
                    min_value=0.0,
                    value=0.05,
                    step=0.01,
                    key="sim_jitter_sigma",
                )
                exchange_override = st.number_input(
                    "Exchange frequency override (steps)",
                    min_value=0,
                    value=0,
                    step=50,
                    help="0 keeps the automatic heuristic.",
                    key="sim_exchange_override",
                )
                temp_schedule = st.selectbox(
                    "Temperature schedule",
                    options=["auto", "exponential", "geometric", "linear"],
                    index=0,
                    key="sim_temperature_schedule",
                )
                schedule_mode = None if temp_schedule == "auto" else temp_schedule

            run_in_progress = bool(st.session_state.get(_RUN_PENDING, False))

            # CRITICAL: Only trigger simulation on button click, not on every rerun
            if st.button(
                "Run replica exchange",
                type="primary",
                disabled=run_in_progress,
                key="sim_run_button",
            ):
                # Button was just clicked - prepare config and run IMMEDIATELY
                try:
                    temps = _parse_temperature_ladder(temps_raw)
                    seed_val = int(random_seed_str) if random_seed_str.strip() else None
                    config = SimulationConfig(
                        pdb_path=input_choice,
                        temperatures=temps,
                        steps=int(steps),
                        quick=quick,
                         save_restart_pdb=bool(save_restart),
                        random_seed=seed_val,
                        label=run_label or None,
                        jitter_start=bool(jitter),
                        jitter_sigma_A=float(jitter_sigma),
                        exchange_frequency_steps=(
                            int(exchange_override) if exchange_override > 0 else None
                        ),
                        temperature_schedule_mode=schedule_mode,
                        # DISABLED: CV biasing is not production-ready (causes 10-20x slowdown on CPU)
                        # cv_model_bundle=cv_model_path if cv_model_path and cv_model_path.exists() else None,
                        cv_model_bundle=None,
                    )

                    # Mark as in progress BEFORE running to prevent double-clicks
                    st.session_state[_RUN_PENDING] = True

                    with st.spinner("Running replica exchange..."):
                        sim_result = backend.run_sampling(config)

                    st.session_state[_LAST_SIM] = sim_result
                    st.session_state[_LAST_SHARDS] = None
                    st.success(f"Simulation complete: {sim_result.run_id}")

                except Exception as exc:
                    st.error(f"Simulation failed: {exc}")
                finally:
                    st.session_state[_RUN_PENDING] = False
                    st.rerun()  # Force rerun to update UI

            # Show status if simulation is still running (shouldn't happen with sync execution)
            elif run_in_progress:
                st.info("⏳ Simulation in progress... (This shouldn't persist - if it does, refresh the page)")

        sim = st.session_state.get(_LAST_SIM)

        if backend.state.runs:
            with st.expander("Load recorded run", expanded=sim is None):
                run_entries = backend.state.runs
                run_ids = [str(entry.get("run_id")) for entry in run_entries]
                if run_ids:
                    current_idx = 0
                    if sim is not None and sim.run_id in run_ids:
                        current_idx = run_ids.index(sim.run_id)
                    selected_run_id = st.selectbox(
                        "Select run",
                        options=run_ids,
                        index=current_idx,
                        key="load_run_select",
                    )
                    if st.button("Use this run", key="load_run_button"):
                        loaded = backend.load_run(selected_run_id)
                        if loaded is not None:
                            st.session_state[_LAST_SIM] = loaded
                            st.session_state[_LAST_SHARDS] = None
                            sim = loaded
                            st.success(f"Loaded run {loaded.run_id}.")
                        else:
                            st.error("Could not load the selected run from disk.")
                else:
                    st.info("No recorded runs available yet.")

        if sim is not None:
            st.success(
                f"Latest run {sim.run_id} produced {len(sim.traj_files)} "
                f"trajectories across {len(sim.analysis_temperatures)} temperatures."
            )
            st.caption(f"Workspace: {sim.run_dir}")
            with st.expander("Run outputs", expanded=False):
                payload = {
                    "run_id": sim.run_id,
                    "trajectories": [p.name for p in sim.traj_files],
                    "analysis_temperatures": sim.analysis_temperatures,
                }
                if sim.restart_pdb_path:
                    payload["restart_pdb"] = sim.restart_pdb_path.name
                if sim.restart_inputs_entry:
                    payload["restart_input_entry"] = sim.restart_inputs_entry.name
                st.json(payload)
                if sim.restart_inputs_entry:
                    st.caption(
                        f"Restart snapshot copied to inputs: {sim.restart_inputs_entry.name}"
                    )
            st.subheader("Emit shards from the latest run")
            with st.form("emit_shards_form"):
                stride = st.number_input(
                    "Stride (frames)",
                    min_value=1,
                    value=5,
                    step=1,
                    key="emit_stride",
                )
                frames_per_shard = st.number_input(
                    "Frames per shard",
                    min_value=500,
                    value=5000,
                    step=500,
                    key="frames_per_shard",
                )
                hop_frames = st.number_input(
                    "Hop (overlap step)",
                    min_value=0,
                    value=5000,
                    step=500,
                    key="hop_frames",
                )
                temp_default = (
                    sim.analysis_temperatures[0] if sim.analysis_temperatures else 300.0
                )
                shard_temp = st.number_input(
                    "Shard metadata temperature (K)",
                    min_value=0.0,
                    value=float(temp_default),
                    step=5.0,
                    key="emit_temperature",
                )
                seed_start = st.number_input(
                    "Shard ID seed start",
                    min_value=0,
                    value=0,
                    step=1,
                    key="emit_seed_start",
                )
                reference_path = st.text_input(
                    "Reference PDB for RMSD (optional)",
                    value="",
                    key="emit_reference",
                )
                emit = st.form_submit_button("Emit shard files")
                if emit:
                    try:
                        request = ShardRequest(
                            stride=int(stride),
                            temperature=float(shard_temp),
                            seed_start=int(seed_start),
                            frames_per_shard=int(frames_per_shard),
                            hop_frames=int(hop_frames) if hop_frames > 0 else None,
                            reference=(
                                Path(reference_path).expanduser().resolve()
                                if reference_path.strip()
                                else None
                            ),
                        )
                        shard_result = backend.emit_shards(
                            sim,
                            request,
                            provenance={"source": "app_usecase"},
                        )
                        st.session_state[_LAST_SHARDS] = shard_result
                        st.success(
                            f"Emitted {shard_result.n_shards} shards "
                            f"({shard_result.n_frames} frames)."
                        )
                        st.json(
                            {
                                "directory": str(shard_result.shard_dir),
                                "files": [p.name for p in shard_result.shard_paths],
                            }
                        )
                    except Exception as exc:
                        st.error(f"Shard emission failed: {exc}")

    with tab_training:
        st.header("Train collective-variable model")
        feedback = st.session_state.get(_TRAIN_FEEDBACK)
        if feedback:
            if isinstance(feedback, tuple) and len(feedback) == 2:
                level, message = feedback
            else:
                level, message = ("info", str(feedback))
            display_fn = getattr(st, str(level), st.info)
            display_fn(str(message))
            st.session_state[_TRAIN_FEEDBACK] = None
        shard_groups = backend.shard_summaries()
        if not shard_groups:
            st.info("Emit shards before training a CV model.")
        else:
            run_ids = [str(entry.get("run_id")) for entry in shard_groups]
            selected_runs = st.multiselect(
                "Shard groups",
                options=run_ids,
                default=run_ids[-1:] if run_ids else [],
            )
            selected_paths = _select_shard_paths(shard_groups, selected_runs)
            try:
                _selection_runs, selection_text = _summarize_selected_shards(
                    selected_paths
                )
            except ValueError as exc:
                st.error(f"Shard selection invalid: {exc}")
                st.stop()
            st.write(f"Using {len(selected_paths)} shard files for training.")
            if selection_text:
                st.caption(selection_text)
            col_a, col_b, col_c = st.columns(3)
            lag = col_a.number_input(
                "Lag (steps)", min_value=1, value=5, step=1, key="train_lag"
            )
            bins_rg = col_b.number_input(
                "Bins for Rg", min_value=8, value=64, step=4, key="train_bins_rg"
            )
            bins_rmsd = col_c.number_input(
                "Bins for RMSD", min_value=8, value=64, step=4, key="train_bins_rmsd"
            )
            col_d, col_e, col_f = st.columns(3)
            seed = col_d.number_input(
                "Training seed", min_value=0, value=1337, step=1, key="train_seed"
            )
            max_epochs = col_e.number_input(
                "Max epochs", min_value=20, value=200, step=10, key="train_max_epochs"
            )
            patience = col_f.number_input(
                "Early stopping patience",
                min_value=5,
                value=25,
                step=5,
                key="train_patience",
            )
            temperature = st.number_input(
                "Reference temperature (K)",
                min_value=0.0,
                value=300.0,
                step=5.0,
                key="train_temperature",
            )
            hidden = st.text_input(
                "Hidden layer widths",
                value=st.session_state.get("train_hidden_layers", "128,128"),
                help="Comma-separated integers for the Deep-TICA network.",
                key="train_hidden_layers",
            )
            col_tau, col_val, col_ep = st.columns(3)
            tau_raw = col_tau.text_input(
                "Tau schedule (steps)",
                value=st.session_state.get("train_tau_schedule", "2,5,10,20"),
                key="train_tau_schedule",
            )
            val_tau = col_val.number_input(
                "Validation tau (steps)",
                min_value=1,
                value=int(st.session_state.get("train_val_tau", 20)),
                step=1,
                key="train_val_tau",
            )
            epochs_per_tau = col_ep.number_input(
                "Epochs per tau",
                min_value=1,
                value=int(st.session_state.get("train_epochs_per_tau", 15)),
                step=1,
                key="train_epochs_per_tau",
            )
            hidden_layers = tuple(
                int(v.strip()) for v in hidden.split(",") if v.strip()
            ) or (128, 128)
            disabled = len(selected_paths) == 0
            if st.button(
                "Train Deep-TICA model",
                type="primary",
                disabled=disabled,
                key="train_button",
            ):
                try:
                    tau_values = _parse_tau_schedule(tau_raw)
                except ValueError as exc:
                    st.error(f"Tau schedule error: {exc}")
                else:
                    try:
                        train_cfg = TrainingConfig(
                            lag=int(lag),
                            bins={"Rg": int(bins_rg), "RMSD_ref": int(bins_rmsd)},
                            seed=int(seed),
                            temperature=float(temperature),
                            max_epochs=int(max_epochs),
                            early_stopping=int(patience),
                            hidden=hidden_layers,
                            tau_schedule=tuple(tau_values),
                            val_tau=int(val_tau),
                            epochs_per_tau=int(epochs_per_tau),
                        )

                        # Use st.status to show progress during training
                        with st.status("Training Deep-TICA model...", expanded=True) as status:
                            st.write("📊 Loading and preparing shard data...")
                            st.write(f"- Using {len(selected_paths)} shard files")
                            if selection_text:
                                st.write(f"  Runs: {selection_text}")
                            st.write(f"- Lag: {lag}, Bins: Rg={bins_rg}, RMSD={bins_rmsd}")
                            st.write(f"- Max epochs: {max_epochs}, Patience: {patience}")
                            st.write("")
                            st.write("⚙️ Starting training pipeline...")
                            st.caption("Note: Initial data loading may take several minutes for large datasets.")

                            result = backend.train_model(selected_paths, train_cfg)

                            status.update(label="Training completed!", state="complete", expanded=False)

                        st.session_state[_LAST_TRAIN] = result
                        st.session_state[_LAST_TRAIN_CONFIG] = train_cfg
                        st.session_state[_TRAIN_CONFIG_PENDING] = train_cfg

                        # Show training progress if available
                        if result.checkpoint_dir:
                            progress = backend.get_training_progress(result.checkpoint_dir)
                            if progress and progress.get("status") == "completed":
                                st.success(f"Training completed! Best val score: {progress.get('best_val_score', 0.0):.4f}")

                        st.session_state[_TRAIN_FEEDBACK] = (
                            "success",
                            f"Model stored at {result.bundle_path.name} (hash {result.dataset_hash}).",
                        )
                        st.rerun()
                    except RuntimeError as exc:
                        if "Deep-TICA optional dependencies missing" in str(exc):
                            st.warning(DEEPTICA_SKIP_MESSAGE)
                        else:
                            st.error(f"Training failed: {exc}")
                    except Exception as exc:
                        st.error(f"Training failed: {exc}")

        # Check for ongoing training and show log viewer
        models_dir = layout.models_dir
        training_dirs = [d for d in models_dir.glob("training-*") if d.is_dir()]
        if training_dirs:
            latest_training = max(training_dirs, key=lambda d: d.name)
            log_file = latest_training / "training.log"
            progress_file = latest_training / "training_progress.json"

            # Check if this is an ongoing training (progress file doesn't exist or status is "training")
            is_training_ongoing = False
            if log_file.exists():
                if not progress_file.exists():
                    is_training_ongoing = True
                else:
                    try:
                        with progress_file.open("r") as f:
                            progress_data = json.load(f)
                            if progress_data.get("status") == "training":
                                is_training_ongoing = True
                    except Exception:
                        pass

            if is_training_ongoing and log_file.exists():
                with st.expander("⚠️ Training in Progress - View Log", expanded=True):
                    st.warning(f"Training directory: `{latest_training.name}`")
                    st.caption("Training may take 10-30 minutes depending on data size. Check the log below for progress.")

                    if st.button("Refresh Log", key="refresh_train_log_button"):
                        st.rerun()

                    try:
                        with log_file.open("r") as f:
                            log_content = f.read()

                        # Show last 50 lines of log
                        log_lines = log_content.strip().split("\n")
                        if len(log_lines) > 50:
                            st.text_area(
                                "Recent Log Entries (last 50 lines)",
                                "\n".join(log_lines[-50:]),
                                height=300,
                                key="train_log_viewer"
                            )
                        else:
                            st.text_area(
                                "Training Log",
                                log_content,
                                height=300,
                                key="train_log_viewer_full"
                            )
                    except Exception as e:
                        st.error(f"Could not read log file: {e}")

        last_train: TrainingResult | None = st.session_state.get(_LAST_TRAIN)
        if last_train is not None:
            # Show real-time training progress if available
            if last_train.checkpoint_dir:
                progress = backend.get_training_progress(last_train.checkpoint_dir)
                if progress:
                    with st.expander("Training Progress", expanded=True):
                        status = progress.get("status", "unknown")
                        st.write(f"**Status**: {status}")

                        if status == "training":
                            current_epoch = progress.get("current_epoch", 0)
                            total_epochs = progress.get("total_epochs_planned", 0)
                            if total_epochs > 0:
                                st.progress(current_epoch / total_epochs)
                                st.write(f"Epoch {current_epoch} / {total_epochs}")

                        epochs_data = progress.get("epochs", [])
                        if epochs_data:
                            # Plot training curves
                            import pandas as pd
                            df = pd.DataFrame(epochs_data)

                            col1, col2 = st.columns(2)
                            with col1:
                                if "val_score" in df.columns:
                                    st.line_chart(df[["epoch", "val_score"]].set_index("epoch"))
                                    st.caption("Validation Score")
                            with col2:
                                if "train_loss" in df.columns:
                                    st.line_chart(df[["epoch", "train_loss"]].set_index("epoch"))
                                    st.caption("Training Loss")

                            # Show best epoch info
                            best_epoch = progress.get("best_epoch")
                            best_score = progress.get("best_val_score", 0.0)
                            if best_epoch:
                                st.info(f"Best validation score: {best_score:.4f} at epoch {best_epoch}")

            _show_build_outputs(last_train)
            summary = (
                last_train.build_result.artifacts.get("mlcv_deeptica")
                if last_train.build_result
                else None
            )
            if summary:
                _render_deeptica_summary(summary)

        models = backend.list_models()
        if models:
            with st.expander(
                "Load recorded model",
                expanded=st.session_state.get(_LAST_TRAIN) is None,
            ):
                indices = list(range(len(models)))

                def _model_label(idx: int) -> str:
                    entry = models[idx]
                    bundle_raw = entry.get("bundle", "")
                    bundle_name = (
                        Path(bundle_raw).name if bundle_raw else f"model-{idx}"
                    )
                    created = entry.get("created_at", "unknown")
                    return f"{bundle_name} (created {created})"

                selected_idx = st.selectbox(
                    "Stored models",
                    options=indices,
                    format_func=_model_label,
                    key="load_model_select",
                )
                if st.button("Show model", key="load_model_button"):
                    loaded = backend.load_model(int(selected_idx))
                    if loaded is not None:
                        st.session_state[_LAST_TRAIN] = loaded
                        cfg_loaded: TrainingConfig | None = None
                        try:
                            cfg_loaded = backend.training_config_from_entry(
                                models[int(selected_idx)]
                            )
                        except Exception:
                            cfg_loaded = None
                        if cfg_loaded is not None:
                            st.session_state[_LAST_TRAIN_CONFIG] = cfg_loaded
                            st.session_state[_TRAIN_CONFIG_PENDING] = cfg_loaded
                        st.session_state[_TRAIN_FEEDBACK] = (
                            "success",
                            f"Loaded model {loaded.bundle_path.name}.",
                        )
                        st.rerun()
                    else:
                        st.error("Could not load the selected model from disk.")

        last_model_path = backend.latest_model_path()
        if last_model_path is not None:
            st.caption(f"Latest model bundle: {last_model_path}")

    with tab_analysis:
        st.header("Build MSM and FES")
        shard_groups = backend.shard_summaries()

        builds = backend.list_builds()
        if builds:
            with st.expander(
                "Load recorded analysis bundle",
                expanded=st.session_state.get(_LAST_BUILD) is None,
            ):
                indices = list(range(len(builds)))

                def _build_label(idx: int) -> str:
                    entry = builds[idx]
                    bundle_raw = entry.get("bundle", "")
                    bundle_name = (
                        Path(bundle_raw).name if bundle_raw else f"bundle-{idx}"
                    )
                    created = entry.get("created_at", "unknown")
                    return f"{bundle_name} (created {created})"

                selected_idx = st.selectbox(
                    "Stored analysis bundles",
                    options=indices,
                    format_func=_build_label,
                    key="load_build_select",
                )
                if st.button("Show bundle", key="load_build_button"):
                    loaded = backend.load_analysis_bundle(int(selected_idx))
                    if loaded is not None:
                        st.session_state[_LAST_BUILD] = loaded
                        try:
                            cfg_loaded = backend.build_config_from_entry(
                                builds[int(selected_idx)]
                            )
                            _apply_analysis_config_to_state(cfg_loaded)
                        except Exception:
                            pass
                        st.success(f"Loaded bundle {loaded.bundle_path.name}.")
                        _show_build_outputs(loaded)
                        summary = (
                            loaded.build_result.artifacts.get("mlcv_deeptica")
                            if loaded.build_result
                            else None
                        )
                        if summary:
                            _render_deeptica_summary(summary)
                    else:
                        st.error(
                            "Could not load the selected analysis bundle from disk."
                        )

        if not shard_groups:
            st.info("Emit shards to build an MSM/FES bundle.")
        else:
            run_ids = [str(entry.get("run_id")) for entry in shard_groups]
            selected_runs = st.multiselect(
                "Shard groups for analysis",
                options=run_ids,
                default=run_ids,
            )
            selected_paths = _select_shard_paths(shard_groups, selected_runs)
            try:
                _analysis_runs, analysis_text = _summarize_selected_shards(
                    selected_paths
                )
            except ValueError as exc:
                st.error(f"Shard selection invalid: {exc}")
                st.stop()
            st.write(f"Using {len(selected_paths)} shard files for analysis.")
            if analysis_text:
                st.caption(analysis_text)
            col_a, col_b, col_c = st.columns(3)
            lag = col_a.number_input(
                "Lag (steps)", min_value=1, value=10, step=1, key="analysis_lag"
            )
            bins_rg = col_b.number_input(
                "Bins for Rg", min_value=8, value=72, step=4, key="analysis_bins_rg"
            )
            bins_rmsd = col_c.number_input(
                "Bins for RMSD", min_value=8, value=72, step=4, key="analysis_bins_rmsd"
            )
            col_d, col_e = st.columns(2)
            seed = col_d.number_input(
                "Build seed", min_value=0, value=2025, step=1, key="analysis_seed"
            )
            temperature = col_e.number_input(
                "Reference temperature (K)",
                min_value=0.0,
                value=300.0,
                step=5.0,
                key="analysis_temperature",
            )
            learn_cv = st.checkbox(
                "Re-learn Deep-TICA during build",
                value=False,
                key="analysis_learn_cv",
            )
            apply_whitening = st.checkbox(
                "Apply CV whitening",
                value=True,
                key="analysis_apply_whitening",
            )
            col_cluster, col_micro = st.columns(2)
            cluster_mode = col_cluster.selectbox(
                "Discretization mode",
                options=["kmeans", "grid"],
                index=(
                    0
                    if str(
                        st.session_state.get("analysis_cluster_mode", "kmeans")
                    ).lower()
                    != "grid"
                    else 1
                ),
                key="analysis_cluster_mode",
            )
            n_microstates = col_micro.number_input(
                "Number of microstates",
                min_value=2,
                value=int(st.session_state.get("analysis_n_microstates", 20)),
                step=1,
                key="analysis_n_microstates",
            )
            reweight_default = str(
                st.session_state.get("analysis_reweight_mode", "MBAR")
            )
            reweight_index = 0
            if reweight_default.upper() == "TRAM":
                reweight_index = 1
            elif reweight_default.lower() == "none":
                reweight_index = 2
            reweight_mode = st.selectbox(
                "Reweighting mode",
                options=["MBAR", "TRAM", "none"],
                index=reweight_index,
                key="analysis_reweight_mode",
            )
            col_fes_method, col_bw, col_min = st.columns(3)
            fes_method = col_fes_method.selectbox(
                "FES method",
                options=["kde", "grid"],
                index=(
                    0
                    if str(st.session_state.get("analysis_fes_method", "kde")).lower()
                    != "grid"
                    else 1
                ),
                key="analysis_fes_method",
            )
            fes_bandwidth = col_bw.text_input(
                "FES bandwidth",
                value=str(st.session_state.get("analysis_fes_bandwidth", "scott")),
                help="Use 'scott', 'silverman', or a positive float (only for KDE).",
                key="analysis_fes_bandwidth",
            )
            min_count_per_bin = col_min.number_input(
                "Min count per bin",
                min_value=0,
                value=int(st.session_state.get("analysis_min_count_per_bin", 1)),
                step=1,
                key="analysis_min_count_per_bin",
            )
            deeptica_params = None
            if learn_cv:
                reuse = st.checkbox(
                    "Reuse last training hyperparameters",
                    value=st.session_state.get(_LAST_TRAIN_CONFIG) is not None,
                    key="analysis_reuse_train_cfg",
                )
                if reuse and st.session_state.get(_LAST_TRAIN_CONFIG) is not None:
                    cfg: TrainingConfig = st.session_state[_LAST_TRAIN_CONFIG]
                    deeptica_params = cfg.deeptica_params()
                else:
                    lag_ml = st.number_input(
                        "Deep-TICA lag",
                        min_value=1,
                        value=int(lag),
                        key="analysis_lag_ml",
                    )
                    hidden_ml = st.text_input(
                        "Deep-TICA hidden layers",
                        value="128,128",
                        key="analysis_hidden_layers",
                    )
                    hidden_layers = tuple(
                        int(v.strip()) for v in hidden_ml.split(",") if v.strip()
                    ) or (128, 128)
                    max_epochs = st.number_input(
                        "Deep-TICA max epochs",
                        min_value=20,
                        value=200,
                        step=10,
                        key="analysis_max_epochs",
                    )
                    patience = st.number_input(
                        "Deep-TICA patience",
                        min_value=5,
                        value=25,
                        step=5,
                        key="analysis_patience",
                    )
                    deeptica_params = {
                        "lag": int(lag_ml),
                        "n_out": 2,
                        "hidden": hidden_layers,
                        "max_epochs": int(max_epochs),
                        "early_stopping": int(patience),
                        "reweight_mode": "scaled_time",
                    }
            disabled = len(selected_paths) == 0
            if st.button(
                "Build MSM/FES bundle",
                type="primary",
                disabled=disabled,
                key="analysis_build_button",
            ):
                print("--- DEBUG: Build MSM/FES button clicked! ---")
                try:
                    bw_clean = fes_bandwidth.strip()
                    try:
                        bandwidth_val: str | float = (
                            float(bw_clean) if bw_clean else "scott"
                        )
                    except ValueError:
                        bandwidth_val = bw_clean or "scott"
                    reweight_norm = str(reweight_mode)
                    if reweight_norm.upper() in {"MBAR", "TRAM"}:
                        reweight_final = reweight_norm.upper()
                    else:
                        reweight_final = "none"
                    build_cfg = BuildConfig(
                        lag=int(lag),
                        bins={"Rg": int(bins_rg), "RMSD_ref": int(bins_rmsd)},
                        seed=int(seed),
                        temperature=float(temperature),
                        learn_cv=bool(learn_cv),
                        deeptica_params=deeptica_params,
                        notes={"source": "app_usecase"},
                        apply_cv_whitening=bool(apply_whitening),
                        cluster_mode=str(cluster_mode),
                        n_microstates=int(n_microstates),
                        reweight_mode=reweight_final,
                        fes_method=str(fes_method),
                        fes_bandwidth=bandwidth_val,
                        fes_min_count_per_bin=int(min_count_per_bin),
                    )
                    artifact: BuildArtifact | None = None
                    try:
                        artifact = backend.build_analysis(selected_paths, build_cfg)
                    except ValueError as err:
                        error_message = str(err)
                        if "No transition pairs found" in error_message:
                            st.warning(
                                "Analysis halted: "
                                "no transition pairs were detected for the current lag. "
                                f"{error_message}"
                            )
                        else:
                            raise
                    if artifact is not None:
                        st.session_state[_LAST_BUILD] = artifact
                        st.success(
                            f"Bundle {artifact.bundle_path.name} written (hash {artifact.dataset_hash})."
                        )
                        _show_build_outputs(artifact)
                        summary = (
                            artifact.build_result.artifacts.get("mlcv_deeptica")
                            if artifact.build_result
                            else None
                        )
                        if summary:
                            _render_deeptica_summary(summary)
                except Exception as exc:
                    print(f"--- DEBUG: Analysis failed with exception: {exc}")
                    traceback.print_exc()
                    st.error(f"Analysis failed: {exc}")
        
        # Conformations Analysis Section
        st.divider()
        st.subheader("TPT Conformations Analysis")
        st.write("Find metastable states, transition states, and pathways using Transition Path Theory.")
        
        if not shard_groups:
            st.info("Emit shards to run conformations analysis.")
        else:
            with st.expander("Configure Conformations Analysis", expanded=False):
                available_topologies = layout.available_inputs()
                topology_select_col, topology_manual_col = st.columns(2)
                selected_topology: Optional[Path] = None
                if available_topologies:
                    selected_topology = topology_select_col.selectbox(
                        "Topology PDB (from app_intputs/)",
                        options=available_topologies,
                        format_func=lambda p: p.name,
                        key="conf_topology_select",
                    )
                else:
                    topology_select_col.warning(
                        "No PDB files detected in app_intputs/. Provide the topology used during sampling."
                    )

                manual_topology_entry = topology_manual_col.text_input(
                    "Custom topology PDB path",
                    value="",
                    help=(
                        "Optional manual override. Provide an absolute path or a path relative to the workspace "
                        "for the topology used when running the simulations."
                    ),
                    key="conf_topology_manual",
                ).strip()

                topology_path_str = (
                    manual_topology_entry
                    if manual_topology_entry
                    else (str(selected_topology) if selected_topology is not None else "")
                )

                if topology_path_str:
                    topology_candidate = Path(topology_path_str)
                    if not topology_candidate.is_absolute():
                        topology_candidate = (layout.workspace_dir / topology_candidate).resolve()
                    if not topology_candidate.exists():
                        st.warning(
                            f"Topology PDB {topology_path_str!s} does not exist. The analysis run will fail until a valid file is provided."
                        )

                conf_col1, conf_col2, conf_col3 = st.columns(3)
                conf_lag = conf_col1.number_input(
                    "Lag (steps)", min_value=1, value=10, step=1, key="conf_lag"
                )
                conf_n_clusters = conf_col2.number_input(
                    "N microstates", min_value=10, value=30, step=5, key="conf_n_clusters"
                )
                conf_n_components = conf_col3.number_input(
                    "TICA components", min_value=2, value=3, step=1, key="conf_n_components"
                )
                
                conf_col4, conf_col5, conf_col6 = st.columns(3)
                conf_n_metastable = conf_col4.number_input(
                    "N metastable states", min_value=2, value=4, step=1, key="conf_n_metastable"
                )
                conf_temperature = conf_col5.number_input(
                    "Temperature (K)", min_value=0.0, value=300.0, step=5.0, key="conf_temperature"
                )
                conf_n_paths = conf_col6.number_input(
                    "Max pathways", min_value=1, value=10, step=1, key="conf_n_paths"
                )
                
                conf_col7, conf_col8 = st.columns(2)
                conf_auto_detect = conf_col7.checkbox(
                    "Auto-detect source/sink states", value=True, key="conf_auto_detect"
                )
                conf_compute_kis = conf_col8.checkbox(
                    "Compute Kinetic Importance Score", value=True, key="conf_compute_kis"
                )
            
            if st.button(
                "Run Conformations Analysis",
                type="primary",
                disabled=len(selected_paths) == 0 or not topology_path_str,
                key="conformations_button",
            ):
                try:
                    conf_config = ConformationsConfig(
                        lag=int(conf_lag),
                        n_clusters=int(conf_n_clusters),
                        n_components=int(conf_n_components),
                        n_metastable=int(conf_n_metastable),
                        temperature=float(conf_temperature),
                        auto_detect_states=bool(conf_auto_detect),
                        n_paths=int(conf_n_paths),
                        compute_kis=bool(conf_compute_kis),
                        topology_pdb=Path(topology_path_str),
                    )

                    with st.spinner("Running conformations analysis..."):
                        conf_result = backend.run_conformations_analysis(
                            selected_paths, conf_config
                        )
                    
                    if conf_result.error:
                        st.error(f"Conformations analysis failed: {conf_result.error}")
                    else:
                        st.success(f"Conformations analysis complete! Output saved to {conf_result.output_dir.name}")
                        
                        # Display TPT summary
                        if conf_result.tpt_summary:
                            st.subheader("TPT Results")
                            cols = st.columns(4)
                            cols[0].metric("Rate", f"{conf_result.tpt_summary['rate']:.3e}")
                            cols[1].metric("MFPT", f"{conf_result.tpt_summary['mfpt']:.1f}")
                            cols[2].metric("Total Flux", f"{conf_result.tpt_summary['total_flux']:.3e}")
                            cols[3].metric("N Pathways", conf_result.tpt_summary['n_pathways'])
                        
                        # Display metastable states
                        if conf_result.metastable_states:
                            st.subheader("Metastable States")
                            meta_df_data = []
                            for state_id, state_data in conf_result.metastable_states.items():
                                meta_df_data.append({
                                    "State": state_id,
                                    "Population": f"{state_data['population']:.4f}",
                                    "N States": state_data['n_states'],
                                    "PDB": Path(state_data['representative_pdb']).name if state_data['representative_pdb'] else "N/A",
                                })
                            st.dataframe(pd.DataFrame(meta_df_data), use_container_width=True)
                        
                        # Display transition states
                        if conf_result.transition_states:
                            st.subheader("Transition States")
                            ts_df_data = []
                            for ts_data in conf_result.transition_states:
                                ts_df_data.append({
                                    "State Index": ts_data['state_index'],
                                    "Committor": f"{ts_data['committor']:.3f}",
                                    "PDB": Path(ts_data['representative_pdb']).name if ts_data['representative_pdb'] else "N/A",
                                })
                            st.dataframe(pd.DataFrame(ts_df_data), use_container_width=True)
                        
                        # Display plots
                        if conf_result.plots:
                            st.subheader("Visualizations")
                            plot_cols = st.columns(2)
                            plot_idx = 0
                            for plot_name, plot_path in conf_result.plots.items():
                                if plot_path.exists():
                                    with plot_cols[plot_idx % 2]:
                                        st.image(str(plot_path), caption=plot_name.replace("_", " ").title())
                                    plot_idx += 1
                        
                        # Show output directory
                        st.info(f"All conformations saved to: {conf_result.output_dir}")
                        if conf_result.representative_pdbs:
                            st.write(f"📁 {len(conf_result.representative_pdbs)} representative PDB files saved")
                
                except Exception as exc:
                    traceback.print_exc()
                    st.error(f"Conformations analysis failed: {exc}")

    with tab_model_preview:
        st.header("Model Preview & Inspection")

        # Allow user to select a trained model to inspect
        models = backend.list_models()
        if not models:
            st.info("No trained models available. Train a model in the 'Model Training' tab first.")
        else:
            indices = list(range(len(models)))

            def _model_preview_label(idx: int) -> str:
                entry = models[idx]
                bundle_raw = entry.get("bundle", "")
                bundle_name = Path(bundle_raw).name if bundle_raw else f"model-{idx}"
                created = entry.get("created_at", "unknown")
                return f"{bundle_name} (created {created})"

            selected_model_idx = st.selectbox(
                "Select model to inspect",
                options=indices,
                format_func=_model_preview_label,
                key="model_preview_select",
            )

            if st.button("Load Model Details", key="model_preview_load_button"):
                loaded = backend.load_model(int(selected_model_idx))
                if loaded is not None:
                    st.session_state["_model_preview_data"] = loaded
                    st.rerun()

            # Display loaded model
            preview_data = st.session_state.get("_model_preview_data")
            if preview_data is not None:
                st.success(f"Model: {preview_data.bundle_path.name}")

                # Model Configuration
                with st.expander("Model Configuration", expanded=True):
                    model_entry = models[selected_model_idx]
                    config_data = {
                        "Dataset Hash": model_entry.get("dataset_hash", "N/A"),
                        "Lag": model_entry.get("lag", "N/A"),
                        "Temperature (K)": model_entry.get("temperature", "N/A"),
                        "Seed": model_entry.get("seed", "N/A"),
                        "Max Epochs": model_entry.get("max_epochs", "N/A"),
                        "Early Stopping Patience": model_entry.get("early_stopping", "N/A"),
                        "Created At": model_entry.get("created_at", "N/A"),
                    }

                    bins = model_entry.get("bins", {})
                    if bins:
                        config_data["Bins (Rg)"] = bins.get("Rg", "N/A")
                        config_data["Bins (RMSD)"] = bins.get("RMSD_ref", "N/A")

                    hidden = model_entry.get("hidden", [])
                    if hidden:
                        config_data["Hidden Layers"] = " → ".join(str(h) for h in hidden)

                    tau_schedule = model_entry.get("tau_schedule", [])
                    if tau_schedule:
                        config_data["Tau Schedule"] = ", ".join(str(t) for t in tau_schedule)

                    config_data["Val Tau"] = model_entry.get("val_tau", "N/A")
                    config_data["Epochs per Tau"] = model_entry.get("epochs_per_tau", "N/A")

                    for key, value in config_data.items():
                        st.write(f"**{key}**: {value}")

                # Model Architecture
                with st.expander("Model Architecture", expanded=True):
                    st.write("**Network Structure:**")
                    hidden_layers = model_entry.get("hidden", [])
                    if hidden_layers:
                        # Visualize network architecture
                        layers = ["Input"] + [f"Hidden {i+1} ({h})" for i, h in enumerate(hidden_layers)] + ["Output (2)"]
                        st.write(" → ".join(layers))

                        # Calculate approximate parameter count
                        # Assuming input dimension from bins
                        bins_dict = model_entry.get("bins", {})
                        input_dim = 2  # Default: Rg + RMSD

                        total_params = 0
                        prev_dim = input_dim
                        for h in hidden_layers:
                            total_params += prev_dim * h + h  # weights + biases
                            prev_dim = h
                        total_params += prev_dim * 2 + 2  # output layer

                        st.metric("Approximate Total Parameters", f"{total_params:,}")
                    else:
                        st.info("Hidden layer configuration not available")

                # Training Metrics
                with st.expander("Training Metrics", expanded=True):
                    metrics = model_entry.get("metrics", {})
                    if metrics:
                        # Display key metrics
                        key_metrics = {
                            "Best Val Score": metrics.get("best_val_score", "N/A"),
                            "Best Epoch": metrics.get("best_epoch", "N/A"),
                            "Best Tau": metrics.get("best_tau", "N/A"),
                            "Wall Time (s)": metrics.get("wall_time_s", "N/A"),
                        }

                        cols = st.columns(len(key_metrics))
                        for col, (key, value) in zip(cols, key_metrics.items()):
                            col.metric(key, value if value != "N/A" else "N/A")

                        # Plot training curves
                        val_score = metrics.get("val_score_curve", [])
                        if val_score:
                            st.write("**Validation Score Curve:**")
                            import pandas as pd
                            epochs = list(range(1, len(val_score) + 1))
                            df = pd.DataFrame({"Epoch": epochs, "Val Score": val_score})
                            st.line_chart(df.set_index("Epoch"))
                    else:
                        st.info("Training metrics not available for this model")

                # Model Files
                with st.expander("Model Files & Checkpoints"):
                    st.write(f"**Bundle Path**: `{preview_data.bundle_path}`")

                    if preview_data.checkpoint_dir and preview_data.checkpoint_dir.exists():
                        st.write(f"**Checkpoint Directory**: `{preview_data.checkpoint_dir}`")

                        # List checkpoint files
                        checkpoint_files = list(preview_data.checkpoint_dir.glob("*"))
                        if checkpoint_files:
                            st.write("**Available Files:**")
                            for f in sorted(checkpoint_files):
                                st.write(f"- `{f.name}`")
                    else:
                        st.info("No checkpoint directory available")

    with tab_assets:
        st.header("Recorded assets")

        # Simulations section
        st.subheader("Simulations")
        runs = backend.state.runs
        if runs:
            for i, run in enumerate(runs):
                col1, col2 = st.columns([8, 1])
                with col1:
                    st.write(
                        f"**{run.get('run_id', 'Unknown')}** - {run.get('steps', 0)} steps - {run.get('created_at', 'Unknown date')}"
                    )
                    st.caption(f"Temperatures: {run.get('temperatures', [])} K")
                with col2:
                    if st.button(
                        "❌", key=f"delete_run_{i}", help="Delete this simulation"
                    ):
                        if backend.delete_simulation(i):
                            st.success(
                                f"Deleted simulation {run.get('run_id', 'Unknown')}"
                            )
                            st.rerun()
                        else:
                            st.error("Failed to delete simulation")
                st.divider()
        else:
            st.info("No simulations recorded yet.")

        # Shard batches section
        st.subheader("Shard batches")
        shards = backend.state.shards
        if shards:
            for i, shard in enumerate(shards):
                col1, col2 = st.columns([8, 1])
                with col1:
                    st.write(
                        f"**{shard.get('run_id', 'Unknown')}** - {shard.get('n_shards', 0)} shards ({shard.get('n_frames', 0)} frames)"
                    )
                    st.caption(
                        f"Temperature: {shard.get('temperature', 0)} K, Stride: {shard.get('stride', 0)} - {shard.get('created_at', 'Unknown date')}"
                    )
                with col2:
                    if st.button(
                        "❌", key=f"delete_shard_{i}", help="Delete this shard batch"
                    ):
                        if backend.delete_shard_batch(i):
                            st.success(
                                f"Deleted shard batch from {shard.get('run_id', 'Unknown')}"
                            )
                            st.rerun()
                        else:
                            st.error("Failed to delete shard batch")
                st.divider()
        else:
            st.info("No shard batches recorded yet.")

        # Models section
        st.subheader("Models")
        last_train: TrainingResult | None = st.session_state.get(_LAST_TRAIN)
        if last_train is not None:
            _show_build_outputs(last_train)
            summary = (
                last_train.build_result.artifacts.get("mlcv_deeptica")
                if last_train.build_result
                else None
            )
            if summary:
                _render_deeptica_summary(summary)

        models = backend.list_models()
        if models:
            for i, model in enumerate(models):
                col1, col2 = st.columns([8, 1])
                with col1:
                    bundle_name = Path(model.get("bundle", "")).name
                    st.write(
                        f"**{bundle_name}** - Lag: {model.get('lag', 0)}, Temperature: {model.get('temperature', 0)} K"
                    )
                    st.caption(
                        f"Bins: Rg={model.get('bins', {}).get('Rg', 0)}, RMSD={model.get('bins', {}).get('RMSD_ref', 0)} - {model.get('created_at', 'Unknown date')}"
                    )
                with col2:
                    if st.button(
                        "❌", key=f"delete_model_{i}", help="Delete this model"
                    ):
                        if backend.delete_model(i):
                            st.success(f"Deleted model {bundle_name}")
                            st.rerun()
                        else:
                            st.error("Failed to delete model")
                st.divider()
        else:
            st.info("No models recorded yet.")

        # Analysis bundles section
        st.subheader("Analysis bundles")
        builds = backend.list_builds()
        if builds:
            for i, build in enumerate(builds):
                col1, col2 = st.columns([8, 1])
                with col1:
                    bundle_name = Path(build.get("bundle", "")).name
                    st.write(
                        f"**{bundle_name}** - Lag: {build.get('lag', 0)}, Temperature: {build.get('temperature', 0)} K"
                    )
                    st.caption(
                        f"Learn CV: {build.get('learn_cv', False)} - {build.get('created_at', 'Unknown date')}"
                    )
                with col2:
                    if st.button(
                        "❌",
                        key=f"delete_build_{i}",
                        help="Delete this analysis bundle",
                    ):
                        if backend.delete_analysis_bundle(i):
                            st.success(f"Deleted analysis bundle {bundle_name}")
                            st.rerun()
                        else:
                            st.error("Failed to delete analysis bundle")
                st.divider()
        else:
            st.info("No analysis bundles recorded yet.")

    st.caption(
        "Run this app with: poetry run streamlit run example_programs/app_usecase/app/app.py"
    )


if __name__ == "__main__":
    main()
