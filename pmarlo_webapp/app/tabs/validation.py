import traceback

import numpy as np
import streamlit as st

from core.context import AppContext
from core.view_helpers import (
    SHARD_SELECTOR_HELP,
    _summarize_selected_shards,
    render_shard_selection_table,
)
from pmarlo.visualization.diagnostics import (
    create_sampling_validation_plot,
    create_fes_validation_plot,
)
from pmarlo.api import select_shard_paths


def render_validation_tab(ctx: AppContext) -> None:
    """Render the free energy validation tab."""
    backend = ctx.backend
    layout = ctx.layout

    st.header("Free Energy Validation")
    shard_groups = backend.shard_summaries()
    if not shard_groups:
        st.info("Emit shards first to generate validation plots.")
    else:
        selected_runs = render_shard_selection_table(
            "Select shard groups for validation",
            shard_groups,
            state_key="validation_selected_runs",
            default_behavior="all",
            help_text=SHARD_SELECTOR_HELP,
        )
        if not selected_runs:
            st.warning("Choose at least one shard group to generate validation plots.")
            return
        try:
            selected_paths = select_shard_paths(shard_groups, selected_runs)
        except ValueError as exc:
            st.error(f"Invalid shard selection: {exc}")
            st.stop()

        if not selected_paths:
            st.warning("Select at least one shard group to generate validation plots.")
        else:
            try:
                _, shard_summary = _summarize_selected_shards(selected_paths)
                st.caption(f"Using {len(selected_paths)} shards: {shard_summary}")
            except ValueError as exc:
                st.error(f"Invalid shard selection: {exc}")
                st.stop()

            # Load a sample to determine the number of features available
            n_features_available = None
            try:
                from pmarlo.data.shard import read_shard

                with st.spinner("Inspecting shard data..."):
                    # Read just the first shard to check feature count
                    # read_shard returns (details, X, dtraj) where X is a 2D array
                    details, X_data, _ = read_shard(selected_paths[0])

                    # X_data should be a 2D numpy array (n_frames, n_features)
                    X_arr = np.asarray(X_data)

                    if X_arr.ndim == 1:
                        n_features_available = 1
                    elif X_arr.ndim == 2:
                        n_features_available = X_arr.shape[1]
                    else:
                        n_features_available = None

                    if n_features_available is not None:
                        st.info(f"Detected {n_features_available} feature(s) in shard data")
            except Exception as e:
                st.warning(f"Could not detect feature count from shards: {e}")
                n_features_available = None

            # TICA Parameters
            st.subheader("TICA Projection Parameters")
            col1, col2, col3 = st.columns(3)

            # Set dynamic max_value based on available features
            if n_features_available is not None and n_features_available >= 2:
                max_components = min(n_features_available, 20)
                default_components = min(3, n_features_available)
            else:
                # Fallback if we couldn't detect features
                max_components = 20
                default_components = 3

            val_n_components = col1.number_input(
                "TICA components",
                min_value=2,
                max_value=max_components,
                value=min(default_components, max_components),
                key="validation_n_components",
                help=f"Number of TICA components to compute (max: {max_components} based on available features)"
            )

            # Show warning if only 1 feature available
            if n_features_available == 1:
                st.warning("️ Only 1 feature detected in shards. TICA requires at least 2 features. Consider using shards with more features (e.g., Rg + RMSD).")

            val_lag = col2.number_input(
                "TICA lag",
                min_value=1,
                max_value=100,
                value=10,
                key="validation_lag",
            )
            val_temperature = col3.number_input(
                "Temperature (K)",
                min_value=0.0,
                value=300.0,
                step=10.0,
                key="validation_temperature",
            )

            # Clustering Parameters
            st.subheader("Clustering Parameters")
            col_cluster1, col_cluster2, col_cluster3 = st.columns(3)
            with col_cluster1:
                st.number_input(
                    "Number of clusters",
                    min_value=10,
                    max_value=500,
                    value=st.session_state.get("val_n_clusters", 100),
                    step=10,
                    key="val_n_clusters",
                    help="Number of microstates for discretization (used in discrete trajectory overlay)"
                )

            st.divider()

            # --- Sampling Plot Controls ---
            st.subheader("Sampling Plot Appearance Controls")
            st.markdown(
                """
                Adjust visualization parameters for the trajectory sampling validation plot (1D histogram on TICA 1).
                """
            )
            col_samp1, col_samp2, col_samp3 = st.columns(3)
            with col_samp1:
                st.number_input(
                    "Max Trajectory Length to Plot",
                    min_value=100,
                    max_value=20000,
                    value=st.session_state.get("val_plot_max_len", 1000),
                    step=100,
                    key="val_plot_max_len",
                    help="Maximum number of frames per shard to visualize."
                )
            with col_samp2:
                st.number_input(
                    "Histogram Bins",
                    min_value=10,
                    max_value=500,
                    value=st.session_state.get("val_plot_hist_bins", 150),
                    step=10,
                    key="val_plot_hist_bins",
                    help="Number of bins for the 1D histogram."
                )
            with col_samp3:
                st.number_input(
                    "Trajectory Point Stride",
                    min_value=1,
                    max_value=100,
                    value=st.session_state.get("val_plot_stride", 10),
                    step=1,
                    key="val_plot_stride",
                    help="Plot every N-th point of the trajectory path for clarity."
                )

            st.divider()

            # --- FES Plot Controls ---
            st.subheader("Free Energy Surface (FES) Plot Controls")
            st.markdown(
                """
                Customize the 2D Free Energy Surface visualization on TICA 1 vs TICA 2.
                """
            )
            col_fes1, col_fes2, col_fes3 = st.columns(3)
            with col_fes1:
                st.slider(
                    "Max Free Energy (kT)",
                    min_value=1.0,
                    max_value=20.0,
                    value=st.session_state.get("fes_plot_max_kt", 7.0),
                    step=0.5,
                    key="fes_plot_max_kt",
                    help="Cap the color scale at this energy value (in kT)."
                )
            with col_fes2:
                st.number_input(
                    "Contour Levels",
                    min_value=5,
                    max_value=100,
                    value=st.session_state.get("fes_plot_levels", 25),
                    step=5,
                    key="fes_plot_levels",
                    help="Number of levels for the contour plot."
                )
            with col_fes3:
                # Get list of available colormaps
                available_colormaps = sorted([
                    'viridis', 'plasma', 'inferno', 'magma', 'cividis',
                    'coolwarm', 'RdYlBu', 'seismic', 'twilight', 'jet'
                ])
                default_cmap = st.session_state.get("fes_plot_cmap", "viridis")
                if default_cmap not in available_colormaps:
                    default_cmap = "viridis"
                cmap_index = available_colormaps.index(default_cmap)

                st.selectbox(
                    "Colormap",
                    available_colormaps,
                    index=cmap_index,
                    key="fes_plot_cmap"
                )

            st.checkbox(
                "Show Contour Lines",
                value=st.session_state.get("fes_plot_lines", True),
                key="fes_plot_lines"
            )

            st.divider()

            if st.button("Generate Validation Plots", key="run_validation", type="primary"):
                try:
                    from pmarlo.data.shard import read_shard
                    from pmarlo.markov_state_model.reduction import reduce_features
                    from pmarlo.markov_state_model.free_energy import generate_2d_fes

                    with st.spinner("Loading shard data..."):
                        # Group shards by run_id to maintain temporal continuity
                        # Each run's shards should be concatenated, not treated separately
                        from collections import defaultdict
                        from pmarlo.data.shard import read_shard

                        run_shards = defaultdict(list)

                        # Read all shards and group by run_id
                        for shard_path in selected_paths:
                            details, X_data, _ = read_shard(shard_path)
                            X_arr = np.asarray(X_data)

                            # Extract run_id from shard path or details
                            # Shard paths typically follow pattern: .../run_id/shards/shard_name.json
                            run_id = None
                            for part in shard_path.parts:
                                if part.startswith("run-") or part.startswith("run_"):
                                    run_id = part
                                    break

                            # Fallback: use parent directory name
                            if run_id is None:
                                run_id = shard_path.parent.parent.name

                            run_shards[run_id].append(X_arr)

                        # Concatenate shards from the same run
                        X_list = []
                        run_labels = []
                        for run_id in sorted(run_shards.keys()):
                            shards = run_shards[run_id]
                            # Concatenate all shards from this run along time axis
                            run_trajectory = np.concatenate(shards, axis=0)
                            X_list.append(run_trajectory)
                            run_labels.append(run_id)

                        if not X_list:
                            st.error("No feature data found in shards.")
                            st.stop()

                        st.info(f"Grouped {len(selected_paths)} shards into {len(X_list)} simulation runs: {', '.join(run_labels)}")

                    with st.spinner("Computing TICA projection..."):
                        # Apply TICA per-trajectory to maintain trajectory structure
                        # and avoid frame count mismatches
                        from deeptime.decomposition import TICA
                        from sklearn.preprocessing import StandardScaler

                        # Ensure all trajectories are 2D arrays
                        X_list_2d = []
                        for traj in X_list:
                            traj_arr = np.asarray(traj)
                            if traj_arr.ndim == 1:
                                # Reshape 1D arrays to 2D (n_frames, 1)
                                traj_arr = traj_arr.reshape(-1, 1)
                            X_list_2d.append(traj_arr)

                        # Concatenate all trajectories for fitting the scaler
                        X_concat = np.concatenate(X_list_2d, axis=0)

                        # Check if we have enough features for the requested components
                        n_features = X_concat.shape[1]
                        if n_features < val_n_components:
                            st.error(f"Cannot compute {val_n_components} TICA components from {n_features} features. Reduce n_components.")
                            st.stop()

                        # Standardize features
                        scaler = StandardScaler()
                        scaler.fit(X_concat)

                        # Fit TICA model on all data (scaled)
                        X_list_scaled = [scaler.transform(traj) for traj in X_list_2d]
                        tica = TICA(lagtime=int(val_lag), dim=int(val_n_components))
                        tica_model = tica.fit(X_list_scaled).fetch_model()

                        # Transform each trajectory individually to maintain boundaries
                        projection_list = []
                        for traj in X_list_2d:
                            if len(traj) > val_lag:  # Only transform if trajectory is long enough
                                traj_scaled = scaler.transform(traj)
                                traj_projected = tica_model.transform(traj_scaled)
                                projection_list.append(traj_projected)
                            else:
                                # For very short trajectories, skip or use empty array
                                st.warning(f"Skipping trajectory with length {len(traj)} (shorter than lag={val_lag})")
                                projection_list.append(np.empty((0, int(val_n_components))))

                        # Also create concatenated version for FES
                        projection = np.concatenate([p for p in projection_list if len(p) > 0], axis=0)

                        if projection is None or len(projection) == 0:
                            st.error("TICA projection failed.")
                            st.stop()

                    st.success(f"Computed TICA projection with shape {projection.shape} from {len(projection_list)} trajectories")

                    col_left, col_right = st.columns(2)

                    with col_left:
                        st.subheader("Sampling Connectivity")
                        with st.spinner("Generating sampling plot..."):
                            try:
                                sampling_fig = create_sampling_validation_plot(
                                    projection_data=projection_list,
                                    run_labels=run_labels,
                                    max_length=int(st.session_state.get("val_plot_max_len", 1000)),
                                    hist_bins=int(st.session_state.get("val_plot_hist_bins", 150)),
                                    stride=int(st.session_state.get("val_plot_stride", 10)),
                                )

                                if sampling_fig and hasattr(sampling_fig, 'axes'):
                                    st.pyplot(
                                        sampling_fig,
                                        clear_figure=True,
                                        use_container_width=True,
                                    )
                                else:
                                    st.warning("Could not generate sampling validation plot.")
                            except Exception as sampling_err:
                                st.warning(f"Could not generate sampling plot: {sampling_err}")
                                with st.expander("Show error details"):
                                    st.code(traceback.format_exc())

                    with col_right:
                        st.subheader("Free Energy Surface")
                        with st.spinner("Computing FES..."):
                            try:
                                if projection.ndim == 2 and projection.shape[1] >= 2:
                                    fes_result = generate_2d_fes(
                                        cv1=projection[:, 0],
                                        cv2=projection[:, 1],
                                        temperature=float(val_temperature),
                                        bins=(50, 50),
                                    )

                                    # Create a mock app_state object with FES data
                                    x_centers = 0.5 * (fes_result.xedges[:-1] + fes_result.xedges[1:])
                                    y_centers = 0.5 * (fes_result.yedges[:-1] + fes_result.yedges[1:])
                                    xx, yy = np.meshgrid(x_centers, y_centers)

                                    fes_fig = create_fes_validation_plot(
                                        fes_grid=(xx, yy),
                                        fes_data=fes_result.F,
                                        max_kt=float(st.session_state.get("fes_plot_max_kt", 7.0)),
                                        levels=int(st.session_state.get("fes_plot_levels", 25)),
                                        cmap=st.session_state.get("fes_plot_cmap", "viridis"),
                                        show_lines=bool(st.session_state.get("fes_plot_lines", True)),
                                    )

                                    if fes_fig and hasattr(fes_fig, 'axes'):
                                        st.pyplot(
                                            fes_fig,
                                            clear_figure=True,
                                            use_container_width=True,
                                        )
                                    else:
                                        st.warning("Could not generate Free Energy Surface plot.")
                                else:
                                    st.warning("Need at least 2 TICA components for FES.")
                            except Exception as fes_err:
                                st.warning(f"Could not generate FES plot: {fes_err}")
                                with st.expander("Show error details"):
                                    st.code(traceback.format_exc())

                    st.divider()

                    # Discrete Trajectory Overlay Plot
                    st.subheader("Discrete Trajectory Overlay")


                    with st.spinner("Computing discrete trajectories..."):
                        try:
                            from pmarlo.markov_state_model.clustering import cluster_microstates

                            n_clusters = st.session_state.get("val_n_clusters", 100)
                            projection_concat = np.concatenate([p for p in projection_list if len(p) > 0], axis=0)

                            # Perform clustering with correct parameter name
                            clustering_result = cluster_microstates(
                                projection_concat,
                                n_states=n_clusters,
                                method="kmeans",  # Changed from mode="kmeans"
                                random_state=2025
                            )

                            cluster_centers = clustering_result.centers
                            labels_concat = clustering_result.labels

                            # Split labels back into per-trajectory
                            dtraj_list = []
                            start_idx = 0
                            for proj in projection_list:
                                length = len(proj)
                                if length > 0:
                                    dtraj_list.append(labels_concat[start_idx:start_idx + length])
                                    start_idx += length
                                else:
                                    dtraj_list.append(np.array([], dtype=np.int32))

                            discrete_fig = create_sampling_validation_plot(
                                projection_data=projection_list,
                                run_labels=run_labels,
                                dtraj_data=dtraj_list,
                                cluster_centers=cluster_centers,
                                max_length=int(st.session_state.get("val_plot_max_len", 1000)),
                                hist_bins=int(st.session_state.get("val_plot_hist_bins", 150)),
                                stride=int(st.session_state.get("val_plot_stride", 10)),
                            )

                            if discrete_fig and hasattr(discrete_fig, 'axes'):
                                st.pyplot(
                                    discrete_fig,
                                    clear_figure=True,
                                    use_container_width=True,
                                )
                            else:
                                st.warning("Could not generate discrete overlay plot.")

                            # Sampling quality metrics
                            st.markdown("**Sampling Quality Diagnostics**")

                            all_x = projection[:, 0]
                            x_min, x_max = np.min(all_x), np.max(all_x)
                            x_threshold = x_min + 0.9 * (x_max - x_min)

                            runs_reaching_right = 0
                            for proj in projection_list:
                                if len(proj) > 0:
                                    if np.max(proj[:, 0]) >= x_threshold:
                                        runs_reaching_right += 1

                            total_runs = len([p for p in projection_list if len(p) > 0])
                            fraction_reaching = runs_reaching_right / total_runs if total_runs > 0 else 0

                            col_metric1, col_metric2 = st.columns(2)
                            with col_metric1:
                                st.metric(
                                    "Runs reaching right barrier",
                                    f"{runs_reaching_right}/{total_runs}",
                                    help="Number of runs that reach the rightmost 10% of the TICA range"
                                )
                            with col_metric2:
                                reversibility_status = "Good" if fraction_reaching > 0.7 else "Partially reversible" if fraction_reaching > 0.3 else "Poor"
                                st.metric(
                                    "Reversibility status",
                                    reversibility_status,
                                    help="Based on fraction of runs exploring the full TICA range"
                                )

                            if fraction_reaching < 0.5:
                                st.warning(
                                    f" Only {fraction_reaching:.1%} of runs reach the full TICA range. "
                                    "MSM may only be valid for the largest connected component."
                                )

                        except Exception as discrete_err:
                            st.warning(f"Could not generate discrete overlay plot: {discrete_err}")
                            with st.expander("Show error details"):
                                st.code(traceback.format_exc())

                except Exception as e:
                    st.error(f"Validation failed: {e}")
                    with st.expander("Show error details"):
                        st.code(traceback.format_exc())
