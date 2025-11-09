import streamlit as st
from pathlib import Path
from typing import Any, Optional

from core.context import AppContext
from core.session import (
    _RUN_PENDING,
    _LAST_SIM,
    _LAST_SHARDS,
)
from backend.types import SimulationConfig, ShardRequest
from backend.feature_profiles import (
    get_feature_profile_info,
    validate_profile_for_cv_biasing,
)
from pmarlo.api import parse_temperature_ladder


def render_sampling_tab(ctx: AppContext) -> None:
    """Render the sampling & shard production tab."""
    backend = ctx.backend
    layout = ctx.layout

    st.header("Sampling & Shard Production")

    st.markdown("---")

    inputs = layout.available_inputs()
    if not inputs:
        st.warning("No prepared proteins found. Place a PDB under app_input/.")
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
            # Single-temperature mode toggle
            single_temp_mode = st.checkbox(
                "Single-temperature mode",
                value=False,
                key="sim_single_temp_mode",
                help="Run MD at a single temperature without replica exchange"
            )

            if single_temp_mode:
                temps_raw = st.text_input(
                    "Target temperature (K)",
                    "300",
                    key="sim_temperature_ladder",
                    help="Temperature for MD simulation"
                )
            else:
                temps_raw = st.text_input(
                    "Temperature ladder (K)",
                    "300, 320, 340",
                    key="sim_temperature_ladder",
                    help="Comma-separated temperatures for replica exchange"
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
                    "copied into app_input/ so it becomes available as a protein input."
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

        use_cv_model = False
        cv_model_path: Optional[Path] = None

        # CV-Informed Sampling
        with st.expander("CV-Informed Sampling (Optional)", expanded=False):
            st.write("Use a trained Deep-TICA model to bias the simulation and explore diverse conformations.")
            models = backend.list_models()
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
                    model_entry = models[selected_cv_idx]
                    cv_model_path = backend.resolve_cv_bundle_dir(model_entry)

                    if cv_model_path is None:
                        with st.spinner("Exporting CV bias bundle for the selected model..."):
                            try:
                                exported_dir = backend.ensure_cv_bundle(selected_cv_idx)
                            except Exception as exc:
                                exported_dir = None
                                st.error(
                                    f"CV bundle export failed: {exc}\n\n"
                                    "Verify the model was trained on molecular feature shards "
                                    "and that feature_spec.yaml matches those features."
                                )
                            else:
                                cv_model_path = exported_dir
                                st.success(
                                    f"CV bias bundle exported to {exported_dir}.\n"
                                    "You can now launch CV-informed sampling with this model."
                                )

                    if cv_model_path is not None:
                        st.success(f"Selected CV bundle: {cv_model_path}")
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
                    else:
                        st.error(
                            "**Missing CV Bias Bundle**\n\n"
                            "Selected model is missing the exported CV bias bundle (deeptica_cv_model.* files).\n\n"
                            "**Common causes:**\n"
                            "1. **Feature mismatch**: Model was trained on collective variables (Rg, RMSD) instead of "
                            "molecular features (distances, angles, dihedrals). CV biasing requires molecular features.\n"
                            "2. **Export failed**: Training completed but CV export encountered an error.\n\n"
                            "**Solution:**\n"
                            "- For existing models: Check `CV_BIAS_REQUIREMENTS.md` for details on feature requirements\n"
                            "- For new models: Ensure shards contain molecular features matching `feature_spec.yaml`\n"
                            "- Run the export utility: `poetry run python export_cv_bundle.py <model_path>`"
                        )
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

        # Dynamic button text based on mode
        button_text = "Run single-temperature MD" if single_temp_mode else "Run replica exchange"

        # CRITICAL: Only trigger simulation on button click, not on every rerun
        if st.button(
                button_text,
                type="primary",
                disabled=run_in_progress,
                key="sim_run_button",
        ):
            # Button was just clicked - prepare config and run IMMEDIATELY
            try:
                temps = parse_temperature_ladder(temps_raw)
                seed_val = int(random_seed_str) if random_seed_str.strip() else None

                if use_cv_model and cv_model_path is None:
                    raise RuntimeError(
                        "CV-informed sampling requested but no exported CV bundle was located for the selected model."
                    )

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
                    cv_model_bundle=cv_model_path,
                    single_temperature_mode=bool(single_temp_mode),
                )

                # Mark as in progress BEFORE running to prevent double-clicks
                st.session_state[_RUN_PENDING] = True

                spinner_msg = "Running single-temperature MD..." if single_temp_mode else "Running replica exchange..."
                with st.spinner(spinner_msg):
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
            col_load, col_scan = st.columns([3, 1])

            with col_scan:
                if st.button("Scan for all runs", help="Scan the sims directory for runs not in state"):
                    with st.spinner("Scanning..."):
                        missing = backend.get_missing_state_entries()
                        if missing:
                            # Count runs by status
                            complete = sum(1 for v in missing if v.status.value == "complete")
                            missing_demux = sum(1 for v in missing if v.status.value == "missing_demux")
                            missing_analysis = sum(1 for v in missing if v.status.value == "missing_analysis")

                            msg = f"Found {len(missing)} valid runs not in state:"
                            if complete:
                                msg += f"\n- {complete} complete"
                            if missing_demux:
                                msg += f"\n- {missing_demux} with replica files only (no demux)"
                            if missing_analysis:
                                msg += f"\n- {missing_analysis} missing analysis"
                            st.info(msg)
                        else:
                            st.success("All runs are in state!")

            with col_load:
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
            profile_options = [
                "cv_analysis",
                "molecular_cv_biasing",
                "molecular_custom",
            ]
            feature_profile = st.radio(
                "Feature profile",
                options=profile_options,
                index=profile_options.index("cv_analysis"),
                key="emit_feature_profile",
                help=(
                    "Choose which feature set to extract when creating shards:\n"
                    "- CV analysis uses Rg/RMSD for MSM + clustering\n"
                    "- Molecular profiles extract distances/angles/dihedrals for CV biasing"
                ),
            )
            profile_info = get_feature_profile_info(feature_profile)
            feature_count = profile_info.get("feature_count", "variable")
            st.caption(
                f"{profile_info.get('description', '').strip()} "
                f"(type: {profile_info.get('feature_type', 'cv')}, features: {feature_count})"
            )
            if feature_profile == "molecular_custom":
                st.caption("Uses feature definitions from app/feature_spec.yaml.")
            compatible, compatibility_msg = validate_profile_for_cv_biasing(feature_profile)
            if compatible:
                st.success(compatibility_msg)
            else:
                st.warning(compatibility_msg)

            emit = st.form_submit_button("Emit shard files")
            if emit:
                try:
                    request = ShardRequest(
                        stride=int(stride),
                        temperature=float(shard_temp),
                        seed_start=int(seed_start),
                        frames_per_shard=int(frames_per_shard),
                        hop_frames=int(hop_frames) if hop_frames > 0 else None,
                        feature_profile=feature_profile,
                        reference=(
                            Path(reference_path).expanduser().resolve()
                            if reference_path.strip()
                            else None
                        ),
                    )
                    # Check if this simulation was CV-informed (used deeptica/metabiases)
                    provenance_data = {"source": "pmarlo_webapp"}
                    # Look up the run in state to get CV model info
                    for run_entry in backend.state.runs:
                        if run_entry.get("run_id") == sim.run_id:
                            if run_entry.get("cv_informed"):
                                provenance_data["cv_informed"] = True
                                cv_bundle = run_entry.get("cv_model_bundle")
                                if cv_bundle:
                                    provenance_data["cv_model_bundle"] = cv_bundle
                            break

                    shard_result = backend.emit_shards(
                        sim,
                        request,
                        provenance=provenance_data,
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
