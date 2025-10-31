import streamlit as st
from app.core.context import AppContext

def render_conformations_tab(ctx: AppContext) -> None:
    """Render the conformations analysis tab."""
    st.header("TPT Conformations Analysis")
    st.write(
        "Find metastable states, transition states, and pathways using Transition Path Theory."
    )

    conformations = backend.list_conformations()
    if conformations:
        with st.expander(
                "Load recorded conformation analysis",
                expanded=st.session_state.get(_LAST_CONFORMATIONS) is None,
        ):
            indices = list(range(len(conformations)))


            def _conformation_label(idx: int) -> str:
                entry = conformations[idx]
                output_dir = entry.get("output_dir", "")
                label = Path(output_dir).name if output_dir else f"conformations-{idx}"
                created = entry.get("created_at", "unknown")
                return f"{label} (created {created})"


            selected_conf_idx = st.selectbox(
                "Stored conformations",
                options=indices,
                format_func=_conformation_label,
                key="load_conformations_select",
            )
            if st.button("Show conformations", key="load_conformations_button"):
                entry = conformations[int(selected_conf_idx)]
                loaded = backend.load_conformations(entry)
                if loaded is not None:
                    st.session_state[_LAST_CONFORMATIONS] = loaded
                    st.session_state[_CONFORMATIONS_FEEDBACK] = (
                        "success",
                        f"Loaded conformations from {loaded.output_dir.name}.",
                    )
                else:
                    st.session_state[_CONFORMATIONS_FEEDBACK] = (
                        "error",
                        "Could not load the selected conformations bundle from disk.",
                    )

    shard_groups = backend.shard_summaries()
    selected_paths: List[Path] = []
    topology_path_str = ""
    conf_cv_method = "tica"
    conf_deeptica_projection: Optional[Path] = None
    conf_deeptica_metadata: Optional[Path] = None
    conf_lag = int(st.session_state.get("conf_lag", 10))
    conf_n_clusters = int(st.session_state.get("conf_n_clusters", 100))
    conf_n_components = int(st.session_state.get("conf_n_components", 3))
    conf_committor_thresholds = tuple(
        float(v)
        for v in st.session_state.get("conf_committor_thresholds", (0.1, 0.9))
    )
    conf_cluster_mode = str(st.session_state.get("conf_cluster_mode", "kmeans"))
    conf_cluster_seed = int(st.session_state.get("conf_cluster_seed", 42))
    conf_kmeans_n_init = int(st.session_state.get("conf_kmeans_n_init", 50))
    conf_n_metastable = int(st.session_state.get("conf_n_metastable", 10))
    conf_temperature = float(st.session_state.get("conf_temperature", 300.0))
    conf_n_paths = int(st.session_state.get("conf_n_paths", 10))
    conf_auto_detect = bool(st.session_state.get("conf_auto_detect", True))
    conf_compute_kis = bool(st.session_state.get("conf_compute_kis", True))
    conf_uncertainty = bool(st.session_state.get("conf_uncertainty_analysis", True))
    conf_bootstrap_samples = int(st.session_state.get("conf_bootstrap_samples", 50))

    if not shard_groups:
        st.info("Emit shards to run conformations analysis.")
    else:
        run_ids = [str(entry.get("run_id")) for entry in shard_groups]
        selected_runs = st.multiselect(
            "Shard groups for conformations",
            options=run_ids,
            default=run_ids,
            key="conf_selected_runs",
        )
        selected_paths = _select_shard_paths(shard_groups, selected_runs)
        try:
            _, shard_summary = _summarize_selected_shards(selected_paths)
        except ValueError as exc:
            st.error(f"Shard selection invalid: {exc}")
            st.stop()
        st.write(
            f"Using {len(selected_paths)} shard files for conformations analysis."
        )
        if shard_summary:
            st.caption(shard_summary)

        with st.expander(
                "Configure Conformations Analysis", expanded=False
        ):
            # Topology Selection
            st.markdown(" Topology Selection")
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

            st.divider()

            # Basic Analysis Parameters
            st.markdown(" Basic Analysis Parameters")
            conf_col1, conf_col2, conf_col3 = st.columns(3)
            conf_lag = conf_col1.number_input(
                "Lag (steps)",
                min_value=1,
                value=int(conf_lag),
                step=1,
                key="conf_lag",
                help="Time delay for MSM construction"
            )
            conf_n_components = conf_col2.number_input(
                "TICA components",
                min_value=2,
                value=int(conf_n_components),
                step=1,
                key="conf_n_components",
                help="Number of TICA dimensions for dimensionality reduction"
            )
            conf_temperature = conf_col3.number_input(
                "Temperature (K)",
                min_value=0.0,
                value=float(conf_temperature),
                step=5.0,
                key="conf_temperature",
                help="Reference temperature for free energy calculations"
            )

            st.divider()

            # Clustering Configuration
            st.markdown(" Clustering Configuration")
            conf_cluster_col1, conf_cluster_col2 = st.columns(2)
            conf_n_clusters = conf_cluster_col1.number_input(
                "N microstates (clusters)",
                min_value=50,
                max_value=1000,
                value=int(conf_n_clusters),
                step=10,
                help=(
                    "Controls the number of clusters used when constructing the"
                    " conformational microstates. Increase this to resolve"
                    " additional transition states."
                ),
                key="conf_n_clusters_input",
            )
            conf_cluster_mode = conf_cluster_col2.selectbox(
                "Clustering method",
                options=["kmeans", "minibatchkmeans", "auto"],
                index=["kmeans", "minibatchkmeans", "auto"].index(
                    conf_cluster_mode if conf_cluster_mode in {"kmeans", "minibatchkmeans", "auto"} else "kmeans"
                ),
                key="conf_cluster_mode",
                help="Algorithm for partitioning CV space"
            )

            conf_cluster_col3, conf_cluster_col4 = st.columns(2)
            conf_cluster_seed = conf_cluster_col3.number_input(
                "Clustering seed (-1 for random)",
                min_value=-1,
                value=int(conf_cluster_seed),
                step=1,
                key="conf_cluster_seed",
                help="Random seed for reproducible clustering"
            )
            conf_kmeans_n_init = conf_cluster_col4.number_input(
                "K-means n_init",
                min_value=1,
                value=int(conf_kmeans_n_init),
                step=1,
                key="conf_kmeans_n_init",
                help="Number of k-means initializations"
            )

            st.divider()

            # TPT Configuration
            st.markdown(" Transition Path Theory (TPT) Configuration")
            conf_col4, conf_col5, conf_col6 = st.columns(3)
            conf_n_metastable = conf_col4.number_input(
                "N metastable states",
                min_value=2,
                value=int(conf_n_metastable),
                step=1,
                key="conf_n_metastable_form",
                on_change=_sync_form_metastable_states,
                help="Number of metastable states to identify"
            )
            st.session_state["conf_n_metastable"] = int(conf_n_metastable)
            conf_n_paths = conf_col5.number_input(
                "Max pathways",
                min_value=1,
                value=int(conf_n_paths),
                step=1,
                key="conf_n_paths",
                help="Maximum number of transition pathways to compute"
            )
            conf_compute_kis = conf_col6.checkbox(
                "Compute Kinetic Importance Score",
                value=bool(conf_compute_kis),
                key="conf_compute_kis",
                help="Calculate kinetic importance for each state"
            )

            conf_auto_detect = st.checkbox(
                "Auto-detect source/sink states",
                value=bool(conf_auto_detect),
                key="conf_auto_detect",
                help="Automatically identify source and sink states from metastable populations"
            )

            st.divider()

            # Uncertainty Analysis
            st.markdown(" Uncertainty Analysis")
            conf_col9, conf_col10 = st.columns(2)
            conf_uncertainty = conf_col9.checkbox(
                "Perform uncertainty analysis",
                value=bool(conf_uncertainty),
                help="Run bootstrap estimates for TPT observables and free energies.",
                key="conf_uncertainty_analysis",
            )
            conf_bootstrap_samples = conf_col10.number_input(
                "Bootstrap samples",
                min_value=1,
                value=int(conf_bootstrap_samples),
                step=5,
                help="Number of bootstrap resamples used during uncertainty analysis.",
                key="conf_bootstrap_samples",
                disabled=not conf_uncertainty,
            )

            st.divider()

            # CV Method Selection
            st.markdown(" Collective Variable (CV) Method")
            conf_cv_col1, conf_cv_col2 = st.columns(2)
            conf_cv_method = conf_cv_col1.selectbox(
                "CV method",
                options=["tica", "deeptica"],
                index=0 if conf_cv_method != "deeptica" else 1,
                key="conf_cv_method",
                help="Choose between classical TICA or Deep-TICA for dimensionality reduction"
            )
            conf_deeptica_projection = None
            conf_deeptica_metadata = None
            if conf_cv_method == "deeptica":
                projection_str = conf_cv_col2.text_input(
                    "DeepTICA projection path",
                    value="",
                    key="conf_deeptica_projection",
                    help="Path to a .npz/.npy file containing precomputed DeepTICA CVs.",
                )
                conf_deeptica_projection = Path(projection_str) if projection_str else None
                metadata_str = st.text_input(
                    "DeepTICA whitening metadata (optional)",
                    value="",
                    key="conf_deeptica_metadata",
                    help="Optional JSON file describing the whitening transform for the DeepTICA outputs.",
                )
                conf_deeptica_metadata = Path(metadata_str) if metadata_str else None

    if st.button(
            "Run Conformations Analysis",
            type="primary",
            disabled=(
                    len(selected_paths) == 0
                    or not topology_path_str
                    or (
                            conf_cv_method == "deeptica" and conf_deeptica_projection is None
                    )
            ),
            key="conformations_button",
    ):
        try:
            conf_config = ConformationsConfig(
                lag=int(conf_lag),
                n_clusters=int(conf_n_clusters),
                cluster_mode=str(conf_cluster_mode),
                cluster_seed=(
                    int(conf_cluster_seed)
                    if int(conf_cluster_seed) >= 0
                    else None
                ),
                kmeans_n_init=int(conf_kmeans_n_init),
                n_components=int(conf_n_components),
                n_metastable=int(conf_n_metastable),
                temperature=float(conf_temperature),
                auto_detect_states=bool(conf_auto_detect),
                n_paths=int(conf_n_paths),
                compute_kis=bool(conf_compute_kis),
                uncertainty_analysis=bool(conf_uncertainty),
                bootstrap_samples=int(conf_bootstrap_samples),
                topology_pdb=Path(topology_path_str),
                cv_method=str(conf_cv_method),
                deeptica_projection_path=conf_deeptica_projection,
                deeptica_metadata_path=conf_deeptica_metadata,
                committor_thresholds=tuple(conf_committor_thresholds),
            )

            with st.spinner("Running conformations analysis..."):
                conf_result = backend.run_conformations_analysis(
                    selected_paths, conf_config
                )

            st.session_state[_LAST_CONFORMATIONS] = conf_result
            if conf_result.error:
                st.session_state[_CONFORMATIONS_FEEDBACK] = (
                    "error",
                    f"Conformations analysis failed: {conf_result.error}",
                )
            else:
                st.session_state[_CONFORMATIONS_FEEDBACK] = (
                    "success",
                    f"Conformations analysis complete! Output saved to {conf_result.output_dir.name}",
                )
        except Exception as exc:
            traceback.print_exc()
            st.session_state[_CONFORMATIONS_FEEDBACK] = (
                "error",
                f"Conformations analysis failed: {exc}",
            )

    feedback = st.session_state.get(_CONFORMATIONS_FEEDBACK)
    if isinstance(feedback, tuple) and len(feedback) == 2:
        level, message = feedback
        if level == "success":
            st.success(message)
        elif level == "warning":
            st.warning(message)
        elif level == "info":
            st.info(message)
        else:
            st.error(message)

    last_conf = st.session_state.get(_LAST_CONFORMATIONS)
    if isinstance(last_conf, ConformationsResult):
        _render_conformations_result(last_conf)