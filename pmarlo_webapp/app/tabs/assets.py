import streamlit as st
from pathlib import Path

from core.context import AppContext
from core.session import (
    _ASSET_RUN_SELECTION,
    _ASSET_SHARD_SELECTION,
    _ASSET_MODEL_SELECTION,
    _ASSET_BUILD_SELECTION,
    _ASSET_CONF_SELECTION,
    _LAST_SIM,
    _LAST_BUILD,
    _LAST_CONFORMATIONS,
    _CONFORMATIONS_FEEDBACK,
    _MODEL_PREVIEW_SELECTION,
    _apply_analysis_config_to_state,
)
from core.tables import (
    _runs_dataframe,
    _shards_dataframe,
    _models_dataframe,
    _builds_dataframe,
    _conformations_dataframe,
)
from core.view_helpers import _model_entry_label
from backend.utils import _sanitize_artifacts

def render_assets_tab(ctx: AppContext) -> None:
    """Render the workspace assets tab."""
    backend = ctx.backend

    st.header("Workspace Assets")
    summary_counts = backend.sidebar_summary()
    summary_cols = st.columns(5)
    summary_cols[0].metric("Runs", summary_counts.get("runs", 0))
    summary_cols[1].metric("Shard batches", summary_counts.get("shards", 0))
    summary_cols[2].metric("Models", summary_counts.get("models", 0))
    summary_cols[3].metric("Analyses", summary_counts.get("builds", 0))
    summary_cols[4].metric("Conformations", summary_counts.get("conformations", 0))

    tab_runs, tab_shards, tab_models, tab_builds, tab_confs = st.tabs(
        ["Runs", "Shards", "Models", "Analyses", "Conformations"]
    )

    with tab_runs:
        runs = list(backend.state.runs)
        if not runs:
            st.info("No recorded simulations yet.")
        else:
            df_runs = _runs_dataframe(runs)
            st.dataframe(df_runs, width="stretch")
            indices = df_runs["Index"].tolist()
            if (
                    st.session_state[_ASSET_RUN_SELECTION] is None
                    or st.session_state[_ASSET_RUN_SELECTION] not in indices
            ):
                st.session_state[_ASSET_RUN_SELECTION] = indices[-1]

            def _run_label(idx: int) -> str:
                entry = runs[idx]
                run_id = entry.get("run_id") or f"run-{idx}"
                created = entry.get("created_at", "unknown")
                return f"{run_id} (created {created})"

            selected_run_idx = st.selectbox(
                "Inspect run",
                options=indices,
                format_func=_run_label,
                key=_ASSET_RUN_SELECTION,
            )
            run_entry = runs[int(selected_run_idx)]
            st.subheader("Run manifest")
            st.json(_sanitize_artifacts(run_entry))

            if st.button(
                    "Load run into Sampling tab",
                    key="assets_load_run",
                    help="Restore this run's metadata so sampling controls can use it.",
            ):
                run_id = run_entry.get("run_id")
                try:
                    loaded_run = backend.load_run(str(run_id))
                except Exception as exc:
                    st.error(f"Failed to load run: {exc}")
                else:
                    if loaded_run is None:
                        st.error("Run bundle is missing from disk.")
                    else:
                        st.session_state[_LAST_SIM] = loaded_run
                        st.success(f"Loaded run {loaded_run.run_id}.")
                        st.rerun()

    with tab_shards:
        shards = list(backend.state.shards)
        if not shards:
            st.info("No shard batches recorded yet.")
        else:
            df_shards = _shards_dataframe(shards)
            st.dataframe(df_shards, width="stretch")
            indices = df_shards["Index"].tolist()
            if (
                    st.session_state[_ASSET_SHARD_SELECTION] is None
                    or st.session_state[_ASSET_SHARD_SELECTION] not in indices
            ):
                st.session_state[_ASSET_SHARD_SELECTION] = indices[-1]

            def _shard_label(idx: int) -> str:
                entry = shards[idx]
                run_id = entry.get("run_id", f"run-{idx}")
                created = entry.get("created_at", "unknown")
                return f"{run_id} ({entry.get('n_shards', 0)} shards, created {created})"

            selected_shard_idx = st.selectbox(
                "Inspect shard batch",
                options=indices,
                format_func=_shard_label,
                key=_ASSET_SHARD_SELECTION,
            )
            shard_entry = shards[int(selected_shard_idx)]
            st.subheader("Shard batch manifest")
            st.json(_sanitize_artifacts(shard_entry))

    with tab_models:
        models = backend.list_models()
        if not models:
            st.info("No Deep-TICA models available.")
        else:
            df_models = _models_dataframe(models)
            st.dataframe(df_models, width="stretch")
            indices = df_models["Index"].tolist()
            if (
                    st.session_state[_ASSET_MODEL_SELECTION] is None
                    or st.session_state[_ASSET_MODEL_SELECTION] not in indices
            ):
                st.session_state[_ASSET_MODEL_SELECTION] = indices[-1]

            selected_model_idx = st.selectbox(
                "Inspect model",
                options=indices,
                format_func=lambda idx: _model_entry_label(models[idx], idx),
                key=_ASSET_MODEL_SELECTION,
            )
            model_entry = models[int(selected_model_idx)]
            st.subheader("Model record")
            st.json(_sanitize_artifacts(model_entry))

            if st.button(
                    "Preview in Model tab",
                    key="asset_preview_model",
                    help="Jump to the Model Preview tab for this bundle.",
            ):
                st.session_state[_MODEL_PREVIEW_SELECTION] = int(selected_model_idx)
                st.rerun()

    with tab_builds:
        builds = backend.list_builds()
        if not builds:
            st.info("No MSM/FES analyses recorded yet.")
        else:
            df_builds = _builds_dataframe(builds)
            st.dataframe(df_builds, width="stretch")
            indices = df_builds["Index"].tolist()
            if (
                    st.session_state[_ASSET_BUILD_SELECTION] is None
                    or st.session_state[_ASSET_BUILD_SELECTION] not in indices
            ):
                st.session_state[_ASSET_BUILD_SELECTION] = indices[-1]

            def _build_label(idx: int) -> str:
                entry = builds[idx]
                bundle = entry.get("bundle", "")
                created = entry.get("created_at", "unknown")
                name = Path(bundle).name if bundle else f"bundle-{idx}"
                return f"{name} (created {created})"

            selected_build_idx = st.selectbox(
                "Inspect analysis bundle",
                options=indices,
                format_func=_build_label,
                key=_ASSET_BUILD_SELECTION,
            )
            build_entry = builds[int(selected_build_idx)]
            st.subheader("Analysis record")
            st.json(_sanitize_artifacts(build_entry))

            if st.button(
                    "Load analysis into MSM tab",
                    key="assets_load_build",
                    help="Restore this bundle into the MSM/FES tab.",
            ):
                try:
                    loaded = backend.load_analysis_bundle(int(selected_build_idx))
                except Exception as exc:
                    st.error(f"Failed to load analysis bundle: {exc}")
                else:
                    if loaded is None:
                        st.error("Analysis bundle is missing from disk.")
                    else:
                        st.session_state[_LAST_BUILD] = loaded
                        try:
                            cfg_loaded = backend.build_config_from_entry(
                                build_entry
                            )
                            _apply_analysis_config_to_state(cfg_loaded)
                        except Exception:
                            pass
                        st.success(
                            f"Loaded analysis bundle {Path(build_entry.get('bundle', '')).name}."
                        )
                        st.rerun()

    with tab_confs:
        conformations = backend.list_conformations()
        if not conformations:
            st.info("No conformations analyses recorded yet.")
        else:
            df_confs = _conformations_dataframe(conformations)
            st.dataframe(df_confs, width="stretch")
            indices = df_confs["Index"].tolist()
            if (
                    st.session_state[_ASSET_CONF_SELECTION] is None
                    or st.session_state[_ASSET_CONF_SELECTION] not in indices
            ):
                st.session_state[_ASSET_CONF_SELECTION] = indices[-1]

            def _conf_label(idx: int) -> str:
                entry = conformations[idx]
                output_dir = entry.get("output_dir", "")
                created = entry.get("created_at", "unknown")
                name = Path(output_dir).name if output_dir else f"conf-{idx}"
                return f"{name} (created {created})"

            selected_conf_idx = st.selectbox(
                "Inspect conformations bundle",
                options=indices,
                format_func=_conf_label,
                key=_ASSET_CONF_SELECTION,
            )
            conf_entry = conformations[int(selected_conf_idx)]
            st.subheader("Conformations record")
            st.json(_sanitize_artifacts(conf_entry))

            if st.button(
                    "Load conformations into analysis tab",
                    key="assets_load_conformations",
                    help="Restore this conformations result into the Conformation Analysis tab.",
            ):
                try:
                    loaded = backend.load_conformations(conf_entry)
                except Exception as exc:
                    st.error(f"Failed to load conformations bundle: {exc}")
                else:
                    if loaded is None:
                        st.error("Conformations artifacts are missing from disk.")
                    else:
                        st.session_state[_LAST_CONFORMATIONS] = loaded
                        st.session_state[_CONFORMATIONS_FEEDBACK] = (
                            "success",
                            f"Loaded conformations from {loaded.output_dir.name}.",
                        )
                        st.rerun()
