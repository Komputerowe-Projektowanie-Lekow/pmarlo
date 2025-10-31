import streamlit as st
from app.core.context import AppContext

def render_validation_tab(ctx: AppContext) -> None:
    """Render the free energy validation tab."""
    st.header("Free Energy Validation")
    shard_groups = backend.shard_summaries()
    if not shard_groups:
        st.info("Emit shards first to generate validation plots.")
    else:
        run_ids = [str(entry.get("run_id")) for entry in shard_groups]
        selected_runs = st.multiselect(
            "Select shard groups for validation",
            options=run_ids,
            default=run_ids,
            key="validation_selected_runs",
        )
        selected_paths = _select_shard_paths(shard_groups, selected_runs)

        if not selected_paths:
            st.warning("Select at least one shard group to generate validation plots.")
        else:
            try:
                _, shard_summary = _summarize_selected_shards(selected_paths)
                st.caption(f"Using {len(selected_paths)} shards: {shard_summary}")
            except ValueError as exc:
                st.error(f"Invalid shard selection: {exc}")
                st.stop()

            # TICA Parameters
            st.subheader("TICA Projection Parameters")
            col1, col2, col3 = st.columns(3)
            val_n_components = col1.number_input(
                "TICA components",
                min_value=2,
                max_value=20,
                value=3,
                key="validation_n_components",
            )
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
                    from pmarlo.data.aggregate import load_shards_as_dataset
                    from pmarlo.markov_state_model.reduction import reduce_features
                    from pmarlo.markov_state_model.free_energy import generate_2d_fes

                    with st.spinner("Loading shard data..."):
                        dataset = load_shards_as_dataset(selected_paths)
                        X_list = dataset.get("X", [])
                        if X_list is None or (isinstance(X_list, list) and len(X_list) == 0):
                            st.error("No feature data found in shards.")
                            st.stop()

                    with st.spinner("Computing TICA projection..."):
                        projection = reduce_features(
                            X_list,
                            method="tica",
                            lag=int(val_lag),
                            n_components=int(val_n_components),
                        )
                        if projection is None:
                            st.error("TICA projection failed.")
                            st.stop()

                    st.success(f"Computed TICA projection with shape {projection.shape}")

                    col_left, col_right = st.columns(2)

                    with col_left:
                        st.subheader("Sampling Connectivity")
                        with st.spinner("Generating sampling plot..."):
                            try:
                                # Get trajectory lengths from the dataset
                                traj_lengths = [len(traj) for traj in X_list]
                                # Split projection back into per-trajectory arrays
                                projection_list = []
                                start_idx = 0
                                for length in traj_lengths:
                                    projection_list.append(projection[start_idx:start_idx + length])
                                    start_idx += length


                                # Create a mock app_state object with projection data
                                class MockAppState:
                                    def __init__(self, proj_list):
                                        self.projection_data = proj_list


                                mock_state = MockAppState(projection_list)
                                sampling_fig = create_sampling_validation_plot(mock_state)

                                if sampling_fig and hasattr(sampling_fig, 'axes'):
                                    st.pyplot(
                                        sampling_fig,
                                        clear_figure=True,
                                        width="stretch",
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
                                    class MockAppStateFES:
                                        def __init__(self, fes_result):
                                            # Extract bin centers from edges
                                            x_centers = 0.5 * (fes_result.xedges[:-1] + fes_result.xedges[1:])
                                            y_centers = 0.5 * (fes_result.yedges[:-1] + fes_result.yedges[1:])

                                            # Create meshgrid
                                            xx, yy = np.meshgrid(x_centers, y_centers)
                                            self.fes_grid = [xx, yy]

                                            # Extract free energy values
                                            self.fes_data = fes_result.F


                                    mock_state_fes = MockAppStateFES(fes_result)
                                    fes_fig = create_fes_validation_plot(mock_state_fes)

                                    if fes_fig and hasattr(fes_fig, 'axes'):
                                        st.pyplot(
                                            fes_fig,
                                            clear_figure=True,
                                            width="stretch",
                                        )
                                    else:
                                        st.warning("Could not generate Free Energy Surface plot.")
                                else:
                                    st.warning("Need at least 2 TICA components for FES.")
                            except Exception as fes_err:
                                st.warning(f"Could not generate FES plot: {fes_err}")
                                with st.expander("Show error details"):
                                    st.code(traceback.format_exc())

                except Exception as e:
                    st.error(f"Validation failed: {e}")
                    with st.expander("Show error details"):
                        st.code(traceback.format_exc())
