from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import pandas as pd
import streamlit as st

try:  # Prefer package-relative imports when launched via `streamlit run -m`
    from .backend import (
        BuildArtifact,
        BuildConfig,
        ShardRequest,
        SimulationConfig,
        TrainingConfig,
        TrainingResult,
        WorkflowBackend,
        WorkspaceLayout,
    )
    from .plots import plot_fes, plot_msm
except ImportError:  # Fallback for `streamlit run app.py`
    import sys

    _APP_DIR = Path(__file__).resolve().parent
    if str(_APP_DIR) not in sys.path:
        sys.path.insert(0, str(_APP_DIR))
    from backend import (  # type: ignore
        BuildArtifact,
        BuildConfig,
        ShardRequest,
        SimulationConfig,
        TrainingConfig,
        TrainingResult,
        WorkflowBackend,
        WorkspaceLayout,
    )
    from plots import plot_fes, plot_msm  # type: ignore


# Keys used inside st.session_state
_LAST_SIM = "__pmarlo_last_simulation"
_LAST_SHARDS = "__pmarlo_last_shards"
_LAST_TRAIN = "__pmarlo_last_training"
_LAST_TRAIN_CONFIG = "__pmarlo_last_train_cfg"
_LAST_BUILD = "__pmarlo_last_build"
_RUN_PENDING = "__pmarlo_run_pending"


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


def _select_shard_paths(groups: Sequence[Dict[str, object]], run_ids: Sequence[str]) -> List[Path]:
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


def _metrics_table(flags: Dict[str, object]) -> pd.DataFrame:
    rows = []
    for key, value in flags.items():
        if isinstance(value, dict):
            for sub_key, sub_val in value.items():
                rows.append({"metric": f"{key}.{sub_key}", "value": sub_val})
        else:
            rows.append({"metric": key, "value": value})
    return pd.DataFrame(rows) if rows else pd.DataFrame({"metric": [], "value": []})


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
    meta_cols = st.columns(3)
    meta_cols[0].metric("Shards", int(br.n_shards))
    meta_cols[1].metric("Frames", int(br.n_frames))
    meta_cols[2].metric("Features", len(br.feature_names))
    if br.flags:
        st.dataframe(_metrics_table(br.flags), width="stretch")
    if br.messages:
        st.write("Messages:")
        for msg in br.messages:
            st.write(f"- {msg}")


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


def main() -> None:
    st.set_page_config(page_title="PMARLO Joint Learning", layout="wide")
    _ensure_session_defaults()

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

    tab_sampling, tab_training, tab_analysis, tab_assets = st.tabs(
        [
            "Sampling",
            "Model Training",
            "Analysis",
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
            if st.button(
                "Run replica exchange",
                type="primary",
                disabled=run_in_progress,
                key="sim_run_button",
            ):
                st.session_state[_RUN_PENDING] = True

            if st.session_state.get(_RUN_PENDING, False):
                try:
                    temps = _parse_temperature_ladder(temps_raw)
                    seed_val = int(random_seed_str) if random_seed_str.strip() else None
                    config = SimulationConfig(
                        pdb_path=input_choice,
                        temperatures=temps,
                        steps=int(steps),
                        quick=quick,
                        random_seed=seed_val,
                        label=run_label or None,
                        jitter_start=bool(jitter),
                        jitter_sigma_A=float(jitter_sigma),
                        exchange_frequency_steps=(
                            int(exchange_override) if exchange_override > 0 else None
                        ),
                        temperature_schedule_mode=schedule_mode,
                    )
                    with st.spinner("Running replica exchange..."):
                        sim_result = backend.run_sampling(config)
                    st.session_state[_LAST_SIM] = sim_result
                    st.session_state[_LAST_SHARDS] = None
                except Exception as exc:
                    st.error(f"Simulation failed: {exc}")
                finally:
                    st.session_state[_RUN_PENDING] = False

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
                st.json(
                    {
                        "run_id": sim.run_id,
                        "trajectories": [p.name for p in sim.traj_files],
                        "analysis_temperatures": sim.analysis_temperatures,
                    }
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
                temp_default = sim.analysis_temperatures[0] if sim.analysis_temperatures else 300.0
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
                            reference=Path(reference_path).expanduser().resolve()
                            if reference_path.strip()
                            else None,
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
            st.write(f"Selected {len(selected_paths)} shard files.")
            col_a, col_b, col_c = st.columns(3)
            lag = col_a.number_input("Lag (steps)", min_value=1, value=5, step=1, key="train_lag")
            bins_rg = col_b.number_input("Bins for Rg", min_value=8, value=64, step=4, key="train_bins_rg")
            bins_rmsd = col_c.number_input("Bins for RMSD", min_value=8, value=64, step=4, key="train_bins_rmsd")
            col_d, col_e, col_f = st.columns(3)
            seed = col_d.number_input("Training seed", min_value=0, value=1337, step=1, key="train_seed")
            max_epochs = col_e.number_input("Max epochs", min_value=20, value=200, step=10, key="train_max_epochs")
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
                value="128,128",
                help="Comma-separated integers for the Deep-TICA network.",
                key="train_hidden_layers",
            )
            hidden_layers = tuple(int(v.strip()) for v in hidden.split(",") if v.strip()) or (128, 128)
            disabled = len(selected_paths) == 0
            if st.button("Train Deep-TICA model", type="primary", disabled=disabled, key="train_button"):
                try:
                    train_cfg = TrainingConfig(
                        lag=int(lag),
                        bins={"Rg": int(bins_rg), "RMSD_ref": int(bins_rmsd)},
                        seed=int(seed),
                        temperature=float(temperature),
                        max_epochs=int(max_epochs),
                        early_stopping=int(patience),
                        hidden=hidden_layers,
                    )
                    result = backend.train_model(selected_paths, train_cfg)
                    st.session_state[_LAST_TRAIN] = result
                    st.session_state[_LAST_TRAIN_CONFIG] = train_cfg
                    st.success(
                        f"Model stored at {result.bundle_path.name} (hash {result.dataset_hash})."
                    )
                    _show_build_outputs(result)
                    summary = result.build_result.artifacts.get("mlcv_deeptica") if result.build_result else None
                    if summary:
                        st.caption("Deep-TICA summary")
                        st.json(summary)
                except Exception as exc:
                    st.error(f"Training failed: {exc}")

        last_model_path = backend.latest_model_path()
        if last_model_path is not None:
            st.caption(f"Latest model bundle: {last_model_path}")

    with tab_analysis:
        st.header("Build MSM and FES")
        shard_groups = backend.shard_summaries()
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
            st.write(f"Using {len(selected_paths)} shard files for analysis.")
            col_a, col_b, col_c = st.columns(3)
            lag = col_a.number_input("Lag (steps)", min_value=1, value=10, step=1, key="analysis_lag")
            bins_rg = col_b.number_input("Bins for Rg", min_value=8, value=72, step=4, key="analysis_bins_rg")
            bins_rmsd = col_c.number_input("Bins for RMSD", min_value=8, value=72, step=4, key="analysis_bins_rmsd")
            col_d, col_e = st.columns(2)
            seed = col_d.number_input("Build seed", min_value=0, value=2025, step=1, key="analysis_seed")
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
                    lag_ml = st.number_input("Deep-TICA lag", min_value=1, value=int(lag), key="analysis_lag_ml")
                    hidden_ml = st.text_input("Deep-TICA hidden layers", value="128,128", key="analysis_hidden_layers")
                    hidden_layers = tuple(
                        int(v.strip())
                        for v in hidden_ml.split(",")
                        if v.strip()
                    ) or (128, 128)
                    max_epochs = st.number_input("Deep-TICA max epochs", min_value=20, value=200, step=10, key="analysis_max_epochs")
                    patience = st.number_input("Deep-TICA patience", min_value=5, value=25, step=5, key="analysis_patience")
                    deeptica_params = {
                        "lag": int(lag_ml),
                        "n_out": 2,
                        "hidden": hidden_layers,
                        "max_epochs": int(max_epochs),
                        "early_stopping": int(patience),
                        "reweight_mode": "scaled_time",
                    }
            disabled = len(selected_paths) == 0
            if st.button("Build MSM/FES bundle", type="primary", disabled=disabled, key="analysis_build_button"):
                try:
                    build_cfg = BuildConfig(
                        lag=int(lag),
                        bins={"Rg": int(bins_rg), "RMSD_ref": int(bins_rmsd)},
                        seed=int(seed),
                        temperature=float(temperature),
                        learn_cv=bool(learn_cv),
                        deeptica_params=deeptica_params,
                        notes={"source": "app_usecase"},
                    )
                    artifact = backend.build_analysis(selected_paths, build_cfg)
                    st.session_state[_LAST_BUILD] = artifact
                    st.success(
                        f"Bundle {artifact.bundle_path.name} written (hash {artifact.dataset_hash})."
                    )
                    _show_build_outputs(artifact)
                except Exception as exc:
                    st.error(f"Analysis failed: {exc}")

    with tab_assets:
        st.header("Recorded assets")
        runs_df = pd.DataFrame(backend.state.runs)
        shards_df = pd.DataFrame(backend.state.shards)
        models_df = pd.DataFrame(backend.list_models())
        builds_df = pd.DataFrame(backend.list_builds())
        st.subheader("Simulations")
        st.dataframe(runs_df if not runs_df.empty else pd.DataFrame({}), width="stretch")
        st.subheader("Shard batches")
        st.dataframe(shards_df if not shards_df.empty else pd.DataFrame({}), width="stretch")
        st.subheader("Models")
        st.dataframe(models_df if not models_df.empty else pd.DataFrame({}), width="stretch")
        st.subheader("Analysis bundles")
        st.dataframe(builds_df if not builds_df.empty else pd.DataFrame({}), width="stretch")

    st.caption("Run this app with: poetry run streamlit run example_programs/app_usecase/app/app.py")


if __name__ == "__main__":
    main()
