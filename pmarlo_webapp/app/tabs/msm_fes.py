import streamlit as st
import traceback
from pathlib import Path
from typing import Dict, Any, List

from app.core.context import AppContext
from app.core.session import (
    _LAST_BUILD,
    _LAST_TRAIN_CONFIG,
    _apply_analysis_config_to_state,
)
from app.backend.types import BuildConfig, BuildArtifact, TrainingConfig

def render_msm_fes_tab(ctx: AppContext) -> None:
    """Render the MSM/FES analysis tab."""
    backend = ctx.backend
    layout = ctx.layout
    
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

        # General Settings
        with st.expander(" General Settings", expanded=True):
            col_seed, col_temp = st.columns(2)
            seed = col_seed.number_input(
                "Build seed",
                min_value=0,
                value=2025,
                step=1,
                key="analysis_seed",
                help="Random seed for reproducibility"
            )
            temperature = col_temp.number_input(
                "Reference temperature (K)",
                min_value=0.0,
                value=300.0,
                step=5.0,
                key="analysis_temperature",
                help="Temperature for free energy calculations"
            )

        # MSM Configuration
        with st.expander(" MSM Configuration", expanded=True):
            lag = st.number_input(
                "Lag time (steps)",
                min_value=1,
                value=10,
                step=1,
                key="analysis_lag",
                help="Time delay for building the Markov State Model"
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
                help="Method for partitioning CV space into microstates"
            )
            n_microstates = col_micro.number_input(
                "Number of microstates",
                min_value=2,
                value=int(st.session_state.get("analysis_n_microstates", 20)),
                step=1,
                key="analysis_n_microstates",
                help="Number of discrete states for MSM construction"
            )
            reweight_default = str(
                st.session_state.get("analysis_reweight_mode", "TRAM")
            )
            reweight_index = 0
            if reweight_default.upper() == "MBAR":
                reweight_index = 1
            reweight_mode = st.selectbox(
                "Reweighting mode",
                options=["TRAM", "MBAR"],
                index=reweight_index,
                key="analysis_reweight_mode",
                help="Statistical reweighting method for enhanced sampling"
            )

        # Collective Variable Settings
        with st.expander(" Collective Variable (CV) Settings", expanded=True):
            col_rg, col_rmsd = st.columns(2)
            bins_rg = col_rg.number_input(
                "Bins for Rg",
                min_value=8,
                value=72,
                step=4,
                key="analysis_bins_rg",
                help="Number of bins for radius of gyration"
            )
            bins_rmsd = col_rmsd.number_input(
                "Bins for RMSD",
                min_value=8,
                value=72,
                step=4,
                key="analysis_bins_rmsd",
                help="Number of bins for RMSD from reference"
            )
            apply_whitening = st.checkbox(
                "Apply CV whitening",
                value=True,
                key="analysis_apply_whitening",
                help="Standardize CVs to have zero mean and unit variance"
            )
            learn_cv = st.checkbox(
                "Re-learn Deep-TICA during build",
                value=False,
                key="analysis_learn_cv",
                help="Train a new Deep-TICA model for dimensionality reduction"
            )
            deeptica_params = None
            if learn_cv:
                st.markdown("**Deep-TICA Parameters**")
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

        # FES Configuration
        with st.expander(" Free Energy Surface (FES) Configuration", expanded=True):
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
                help="Method for computing the free energy surface: KDE (kernel density) or grid-based"
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
                help="Minimum number of samples required per bin for FES computation"
            )

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
                    notes={"source": "pmarlo_webapp"},
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


# Helper functions for MSM/FES tab

def _select_shard_paths(shard_groups: List[Dict[str, Any]], selected_runs: List[str]) -> List[Path]:
    """Extract shard file paths from selected run IDs."""
    selected_paths = []
    for entry in shard_groups:
        run_id = str(entry.get("run_id", ""))
        if run_id in selected_runs:
            paths = entry.get("shard_paths", [])
            for p in paths:
                if isinstance(p, (str, Path)):
                    selected_paths.append(Path(p))
    return selected_paths


def _summarize_selected_shards(selected_paths: List[Path]) -> tuple[List[str], str]:
    """Summarize the selected shard files for display."""
    if not selected_paths:
        return [], ""

    # Extract run IDs from shard paths
    run_ids = set()
    for path in selected_paths:
        # Assume shard paths have format: .../run_<id>/shards/...
        parts = path.parts
        for i, part in enumerate(parts):
            if part.startswith("run_"):
                run_ids.add(part)
                break

    summary_text = f"{len(run_ids)} simulation run(s)"
    return list(run_ids), summary_text


def _show_build_outputs(artifact: BuildArtifact) -> None:
    """Display the outputs from a build artifact."""
    st.subheader("Analysis Bundle Outputs")

    # Display basic information
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Bundle Path", artifact.bundle_path.name)
        st.metric("Dataset Hash", artifact.dataset_hash[:12] + "...")
    with col2:
        st.metric("Created", artifact.created_at)
        if artifact.analysis_msm_n_states:
            st.metric("MSM States", artifact.analysis_msm_n_states)

    # Display build result artifacts if available
    if artifact.build_result and hasattr(artifact.build_result, 'artifacts'):
        artifacts = artifact.build_result.artifacts
        if artifacts:
            with st.expander("📦 Available Artifacts", expanded=False):
                for key, value in artifacts.items():
                    st.write(f"**{key}**")
                    if isinstance(value, dict):
                        st.json(value)
                    else:
                        st.write(value)

    # Display debug summary if available
    if artifact.debug_summary:
        with st.expander("🔍 Debug Summary", expanded=False):
            summary = artifact.debug_summary

            # MSM statistics
            if "total_pairs" in summary:
                st.write(f"**Total transition pairs:** {summary['total_pairs']}")
            if "states_observed" in summary:
                st.write(f"**States observed:** {summary['states_observed']}")
            if "largest_scc_size" in summary:
                st.write(f"**Largest connected component:** {summary['largest_scc_size']}")
            if "diag_mass" in summary:
                st.write(f"**Diagonal mass:** {summary['diag_mass']:.4f}")

            # Warnings
            if "warnings" in summary and summary["warnings"]:
                st.warning("**Analysis Warnings:**")
                for warning in summary["warnings"]:
                    if isinstance(warning, dict):
                        st.write(f"- {warning.get('message', warning)}")
                    else:
                        st.write(f"- {warning}")

    # Display guardrail violations if present
    if artifact.guardrail_violations:
        with st.expander("⚠️ Guardrail Violations", expanded=True):
            for violation in artifact.guardrail_violations:
                if isinstance(violation, dict):
                    code = violation.get("code", "unknown")
                    message = violation.get("message", str(violation))
                    st.error(f"**{code}:** {message}")
                else:
                    st.error(str(violation))


def _render_deeptica_summary(summary: Dict[str, Any]) -> None:
    """Render a summary of Deep-TICA training results."""
    st.subheader("Deep-TICA Model Summary")

    col1, col2, col3 = st.columns(3)

    with col1:
        if "n_components" in summary:
            st.metric("Output Dimensions", summary["n_components"])
        if "lag" in summary:
            st.metric("Lag Time", summary["lag"])

    with col2:
        if "epochs_trained" in summary:
            st.metric("Epochs Trained", summary["epochs_trained"])
        if "final_loss" in summary:
            st.metric("Final Loss", f"{summary['final_loss']:.6f}")

    with col3:
        if "validation_score" in summary:
            st.metric("Validation Score", f"{summary['validation_score']:.6f}")
        if "training_time" in summary:
            st.metric("Training Time", f"{summary['training_time']:.2f}s")

    # Display training history if available
    if "training_history" in summary:
        with st.expander("📈 Training History", expanded=False):
            history = summary["training_history"]
            st.line_chart(history)

    # Display model architecture if available
    if "architecture" in summary:
        with st.expander("🏗️ Model Architecture", expanded=False):
            arch = summary["architecture"]
            if isinstance(arch, dict):
                st.json(arch)
            else:
                st.write(arch)
