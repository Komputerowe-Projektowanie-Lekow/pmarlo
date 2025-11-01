import streamlit as st
import traceback
import numpy as np
from pathlib import Path
from typing import Dict, Any, List

from core.context import AppContext
from core.session import (
    _LAST_BUILD,
    _LAST_TRAIN_CONFIG,
    _apply_analysis_config_to_state,
)
from backend.types import BuildConfig, BuildArtifact, TrainingConfig
from plots.diagnostics import (
    plot_canonical_correlations,
    plot_autocorrelation_curves,
    format_warnings,
)

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
            require_connectivity = st.checkbox(
                "Require fully connected MSM",
                value=True,
                key="analysis_require_connectivity",
                help="If enabled, analysis will fail if the MSM has isolated states. Disable to allow disconnected components."
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
                    require_fully_connected_msm=bool(require_connectivity),
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
            # Fixed: backend.shard_summaries() returns "paths" not "shard_paths"
            paths = entry.get("paths", [])
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
    """Display the outputs from a build artifact in a compact grid layout."""
    st.subheader("Analysis Bundle Outputs")

    # Add debug logging
    import logging
    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info("RENDERING BUILD OUTPUTS")
    logger.info(f"Bundle: {artifact.bundle_path}")
    logger.info(f"Has build_result: {artifact.build_result is not None}")
    if artifact.build_result:
        logger.info(f"Has transition_matrix: {artifact.build_result.transition_matrix is not None}")
        logger.info(f"Has fes: {artifact.build_result.fes is not None}")
        logger.info(f"Has diagnostics: {artifact.build_result.diagnostics is not None}")
    logger.info("=" * 80)

    # Display basic information in a compact header
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Bundle", artifact.bundle_path.name[:20] + "...")
    with col2:
        st.metric("Hash", artifact.dataset_hash[:8] + "...")
    with col3:
        st.metric("Created", artifact.created_at.split(" ")[0] if " " in artifact.created_at else artifact.created_at[:10])
    with col4:
        if artifact.analysis_msm_n_states:
            st.metric("MSM States", artifact.analysis_msm_n_states)

    # === PLOTS SECTION (Single Expander) ===
    with st.expander("Plots", expanded=True):
        # Create grid layout for plots: 3 columns, 2 rows
        row1_col1, row1_col2, row1_col3 = st.columns(3)
        row2_col1, row2_col2, row2_col3 = st.columns(3)

        # PLOT 1: MSM Transition Matrix
        with row1_col1:
            if artifact.build_result and artifact.build_result.transition_matrix is not None:
                st.write("**MSM Transition Matrix**")
                logger.info("Rendering MSM transition matrix")
                T = artifact.build_result.transition_matrix
                from pmarlo.reporting.plots import plot_transition_matrix_heatmap
                try:
                    fig = plot_transition_matrix_heatmap(T)
                    # Make figure smaller for grid
                    fig.set_size_inches(5, 4)
                    st.pyplot(fig, use_container_width=True)
                    logger.info("Successfully rendered MSM transition matrix")
                except Exception as e:
                    logger.error(f"Failed to plot MSM: {e}", exc_info=True)
                    st.error(f"Error: {e}")
            else:
                st.write("**MSM Transition Matrix**")
                st.info("No MSM data")

        # PLOT 2: Free Energy Surface
        with row1_col2:
            if artifact.build_result and artifact.build_result.fes is not None:
                st.write("**Free Energy Surface**")
                logger.info(f"Rendering FES, type: {type(artifact.build_result.fes)}")
                fes_obj = artifact.build_result.fes
                from pmarlo.reporting.plots import plot_fes_contour
                try:
                    if hasattr(fes_obj, 'F') and hasattr(fes_obj, 'xedges') and hasattr(fes_obj, 'yedges'):
                        F = fes_obj.F
                        xedges = fes_obj.xedges
                        yedges = fes_obj.yedges
                        cv_names = artifact.build_result.feature_names if artifact.build_result.feature_names else ["CV1", "CV2"]
                        xlabel = cv_names[0] if len(cv_names) > 0 else "CV1"
                        ylabel = cv_names[1] if len(cv_names) > 1 else "CV2"

                        logger.info(f"FES shape: {F.shape}, labels: {xlabel} vs {ylabel}")
                        fig = plot_fes_contour(F, xedges, yedges, xlabel, ylabel)
                        fig.set_size_inches(5, 4)
                        st.pyplot(fig, use_container_width=True)
                        logger.info("Successfully rendered FES")
                    else:
                        st.warning("FES structure not recognized")
                except ValueError as e:
                    # Handle sparse FES gracefully
                    if "too sparse" in str(e):
                        logger.warning(f"FES too sparse: {e}")
                        st.warning(f"FES too sparse to display\n({str(e).split('(')[1].split(')')[0]})")
                    else:
                        raise
                except Exception as e:
                    logger.error(f"Failed to plot FES: {e}", exc_info=True)
                    st.error(f"Error: {e}")
            else:
                st.write("**Free Energy Surface**")
                st.info("No FES data")

        # PLOT 3: Autocorrelation
        with row1_col3:
            if artifact.build_result and hasattr(artifact.build_result, 'diagnostics') and artifact.build_result.diagnostics:
                st.write("**Autocorrelation**")
                logger.info("Rendering autocorrelation")
                try:
                    fig_autocorr = plot_autocorrelation_curves(artifact.build_result.diagnostics)
                    fig_autocorr.set_size_inches(5, 4)
                    st.pyplot(fig_autocorr, use_container_width=True)
                except Exception as e:
                    logger.error(f"Failed to plot autocorrelation: {e}", exc_info=True)
                    st.error(f"Error: {e}")
            else:
                st.write("**Autocorrelation**")
                st.info("No diagnostics")

        # PLOT 4: Canonical Correlation
        with row2_col1:
            if artifact.build_result and hasattr(artifact.build_result, 'diagnostics') and artifact.build_result.diagnostics:
                st.write("**Canonical Correlation**")
                logger.info("Rendering canonical correlation")
                try:
                    fig_canonical = plot_canonical_correlations(artifact.build_result.diagnostics)
                    fig_canonical.set_size_inches(5, 4)
                    st.pyplot(fig_canonical, use_container_width=True)
                except Exception as e:
                    logger.error(f"Failed to plot canonical correlation: {e}", exc_info=True)
                    st.error(f"Error: {e}")
            else:
                st.write("**Canonical Correlation**")
                st.info("No diagnostics")

        # PLOT 5: Stationary Distribution (if available)
        with row2_col2:
            if artifact.build_result and artifact.build_result.stationary_distribution is not None:
                st.write("**Stationary Distribution**")
                pi = artifact.build_result.stationary_distribution
                try:
                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots(figsize=(5, 4))
                    ax.bar(range(len(pi)), pi, color='steelblue', alpha=0.7)
                    ax.set_xlabel("State")
                    ax.set_ylabel("Probability")
                    ax.set_title("Stationary Distribution")
                    fig.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                except Exception as e:
                    logger.error(f"Failed to plot stationary dist: {e}", exc_info=True)
                    st.error(f"Error: {e}")
            else:
                st.write("**Stationary Distribution**")
                st.info("No data")

        # PLOT 6: FES Quality (if available)
        with row2_col3:
            if artifact.build_result and artifact.build_result.artifacts and "fes_quality" in artifact.build_result.artifacts:
                st.write("**FES Quality**")
                quality = artifact.build_result.artifacts["fes_quality"]
                try:
                    # Create a simple bar chart of quality metrics
                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots(figsize=(5, 4))

                    metrics = []
                    values = []
                    if "empty_bins_fraction" in quality:
                        metrics.append("Empty\nBins %")
                        values.append(quality["empty_bins_fraction"] * 100)
                    if "smoothing" in quality and "applied_fraction" in quality["smoothing"]:
                        metrics.append("Smoothed\nBins %")
                        values.append(quality["smoothing"]["applied_fraction"] * 100)

                    if metrics:
                        ax.bar(metrics, values, color=['orange', 'green'][:len(values)], alpha=0.7)
                        ax.set_ylabel("Percentage")
                        ax.set_title("FES Quality Metrics")
                        ax.set_ylim(0, 100)
                        fig.tight_layout()
                        st.pyplot(fig, use_container_width=True)
                    else:
                        st.info("No quality metrics")
                except Exception as e:
                    logger.error(f"Failed to plot FES quality: {e}", exc_info=True)
                    st.error(f"Error: {e}")
            else:
                st.write("**FES Quality**")
                st.info("No quality data")

    # === METRICS SECTION (Single Expander) ===
    with st.expander("All Metrics", expanded=False):
        # Create tabs for different metric categories
        tab_msm, tab_fes, tab_diag, tab_artifacts = st.tabs([
            "MSM Metrics",
            "FES Metrics",
            "Diagnostics",
            "Artifacts"
        ])

        # MSM Metrics Tab
        with tab_msm:
            if artifact.build_result and artifact.build_result.transition_matrix is not None:
                T = artifact.build_result.transition_matrix
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("States", T.shape[0])
                    st.metric("Diagonal Mass", f"{np.trace(T) / T.shape[0]:.4f}")
                with col2:
                    st.metric("Min Prob", f"{np.min(T):.6f}")
                    st.metric("Max Prob", f"{np.max(T):.6f}")
                with col3:
                    if artifact.build_result.stationary_distribution is not None:
                        pi = artifact.build_result.stationary_distribution
                        st.metric("Most Populated State", f"{np.argmax(pi)}")
                        st.metric("Max Population", f"{np.max(pi):.4f}")

                # Debug summary
                if artifact.debug_summary:
                    st.write("**Build Statistics:**")
                    summary_cols = st.columns(4)
                    if "total_pairs" in artifact.debug_summary:
                        summary_cols[0].metric("Transition Pairs", artifact.debug_summary["total_pairs"])
                    if "states_observed" in artifact.debug_summary:
                        summary_cols[1].metric("States Observed", artifact.debug_summary["states_observed"])
                    if "largest_scc_size" in artifact.debug_summary:
                        summary_cols[2].metric("Largest SCC", artifact.debug_summary["largest_scc_size"])
                    if "diag_mass" in artifact.debug_summary:
                        summary_cols[3].metric("Diag Mass", f"{artifact.debug_summary['diag_mass']:.4f}")
            else:
                st.info("No MSM metrics available")

        # FES Metrics Tab
        with tab_fes:
            if artifact.build_result and artifact.build_result.fes is not None:
                fes_obj = artifact.build_result.fes
                if hasattr(fes_obj, 'metadata'):
                    metadata = fes_obj.metadata
                    if metadata:
                        st.json(metadata)
                    else:
                        st.info("No FES metadata")

                if artifact.build_result.artifacts and "fes_quality" in artifact.build_result.artifacts:
                    st.write("**Quality Metrics:**")
                    st.json(artifact.build_result.artifacts["fes_quality"])
            else:
                st.info("No FES metrics available")

        # Diagnostics Tab
        with tab_diag:
            if artifact.build_result and hasattr(artifact.build_result, 'diagnostics') and artifact.build_result.diagnostics:
                diagnostics = artifact.build_result.diagnostics

                # Display warnings
                warnings = format_warnings(diagnostics)
                if warnings:
                    st.warning("**Warnings:**")
                    for warning in warnings:
                        st.write(f"- {warning}")

                # Display taus
                if "taus" in diagnostics:
                    st.write(f"**Lag times used:** {diagnostics['taus']}")

                # Display diag_mass
                if "diag_mass" in diagnostics and diagnostics["diag_mass"] is not None:
                    st.metric("Diagonal Mass", f"{diagnostics['diag_mass']:.4f}")

                # Raw diagnostics data
                with st.expander("Raw Diagnostics JSON", expanded=False):
                    st.json(diagnostics)
            else:
                st.info("No diagnostics available")

        # Artifacts Tab
        with tab_artifacts:
            if artifact.build_result and hasattr(artifact.build_result, 'artifacts'):
                artifacts = artifact.build_result.artifacts
                if artifacts:
                    for key, value in artifacts.items():
                        with st.expander(f"**{key}**", expanded=False):
                            if isinstance(value, dict):
                                st.json(value)
                            else:
                                st.write(value)
                else:
                    st.info("No artifacts available")
            else:
                st.info("No artifacts available")

    # Guardrail violations (always visible if present)
    if artifact.guardrail_violations:
        st.error("**Guardrail Violations**")
        for violation in artifact.guardrail_violations:
            if isinstance(violation, dict):
                code = violation.get("code", "unknown")
                message = violation.get("message", str(violation))
                st.error(f"**{code}:** {message}")
            else:
                st.error(str(violation))

    logger.info("Finished rendering build outputs")


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
        with st.expander("Training History", expanded=False):
            history = summary["training_history"]
            st.line_chart(history)

    # Display model architecture if available
    if "architecture" in summary:
        with st.expander("Model Architecture", expanded=False):
            arch = summary["architecture"]
            if isinstance(arch, dict):
                st.json(arch)
            else:
                st.write(arch)
