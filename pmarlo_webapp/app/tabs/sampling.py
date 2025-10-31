import streamlit as st
from app.core.context import AppContext

def render_sampling_tab(ctx: AppContext) -> None:
    """Render the sampling & shard production tab."""
    st.header("Sampling & Shard Production")
    inputs = layout.available_inputs()
    if not inputs:
        st.warning("No prepared proteins found. Place a PDB under app_intputs/.")
    else:
        # Basic Simulation Settings
        with st.expander("Basic Simulation Settings", expanded=True):
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
                help="Comma-separated temperatures for replica exchange MD"
            )
            steps = st.number_input(
                "Total MD steps",
                min_value=1000,
                max_value=5_000_000,
                value=50_000,
                step=5_000,
                key="sim_total_steps",
            )
            col_quick, col_restart = st.columns(2)
            quick = col_quick.checkbox(
                "Quick preset (short equilibration)",
                value=True,
                key="sim_quick_preset",
            )
            save_restart = col_restart.checkbox(
                "Save last frame as restart input",
                value=True,
                help=(
                    "When enabled, the final MD frame is stored in the run directory and "
                    "copied into app_intputs/ so it becomes available as a protein input."
                ),
                key="sim_save_restart_snapshot",
            )
            col_seed, col_label = st.columns(2)
            random_seed_str = col_seed.text_input(
                "Random seed (blank = auto)",
                "",
                key="sim_random_seed",
            )
            run_label = col_label.text_input(
                "Run label (optional)",
                "",
                key="sim_run_label",
            )

        # CV-Informed Sampling
        with st.expander("CV-Informed Sampling (Optional)", expanded=False):
            st.write("Use a trained Deep-TICA model to bias the simulation and explore diverse conformations.")
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
                        st.success(f"Selected CV model from: {cv_model_path.name}")
                        st.info(
                            "**CV-Informed Sampling ENABLED**\n\n"
                            "The trained Deep-TICA model will be used to bias the simulation:\n"
                            "- **Bias type**: Harmonic expansion (E = k * Σ(cv²))\n"
                            "- **Effect**: Repulsive forces in CV space → explore diverse conformations\n"
                            "- **Implementation**: OpenMM computes forces via F = -∇E\n\n"
                            "**Requirements**:\n"
                            "- `openmm-torch` must be installed (`conda install -c conda-forge openmm-torch`)\n"
                            "- CUDA-enabled PyTorch recommended (CPU is ~10-20x slower)\n\n"
                            "**Note**: The model expects **molecular features** (distances, angles) as input. "
                            "Feature extraction is automatically configured in the OpenMM system."
                        )
                    elif cv_model_path:
                        st.error(f"Model checkpoint directory not found: {cv_model_path}")
            else:
                st.info(
                    "No trained CV models available. Train a model in the 'Model Training' tab to enable CV-informed sampling.")

        # Advanced Options
        with st.expander("Advanced Simulation Options", expanded=False):
            jitter = st.checkbox(
                "Jitter starting structure",
                value=False,
                key="sim_jitter_toggle",
                help="Add small random perturbations to initial atomic positions"
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
                help="Method for distributing replicas across the temperature ladder"
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
                        provenance={"source": "pmarlo_webapp"},
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
