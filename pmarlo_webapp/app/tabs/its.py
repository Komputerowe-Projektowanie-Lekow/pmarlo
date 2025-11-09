from pathlib import Path
from typing import Mapping, Sequence

import streamlit as st

from core.context import AppContext
from core.session import (
    _ITS_PENDING_TOPOLOGY,
    _ITS_PENDING_FEATURE_SPEC,
    _LAST_ITS_RESULT,
    _ITS_FEEDBACK,
)
from core.parsers import _format_lag_sequence, _parse_lag_sequence
from core.view_helpers import (
    SHARD_SELECTOR_HELP,
    _infer_default_topology,
    _default_feature_spec_path,
    _summarize_selected_shards,
    render_shard_selection_table,
    summarize_selected_feature_profiles,
)
from core.tables import _timescales_dataframe
from backend.its import calculate_its, plot_its
from pmarlo.api import select_shard_paths
from pmarlo.data.shard import read_shard
from pmarlo.visualization.diagnostics import create_shard_frame_histogram


def _shard_frame_counts(shard_paths: Sequence[Path]) -> list[int]:
    """Return the frame counts for each shard path."""

    counts: list[int] = []
    for raw_path in shard_paths:
        details, _, _ = read_shard(Path(raw_path))
        counts.append(int(details.n_frames))
    return counts


def render_its_tab(ctx: AppContext) -> None:
    """Render the implied timescales tab."""
    backend = ctx.backend
    layout = ctx.layout

    st.header("Implied Timescales")
    shard_groups = backend.shard_summaries()
    if not shard_groups:
        st.info("Emit shard batches to compute implied timescales.")
    else:
        run_ids = [str(entry.get("run_id")) for entry in shard_groups]
        stored_selection = [
            run_id
            for run_id in st.session_state.get("its_selected_runs", [])
            if run_id in run_ids
        ]
        if not stored_selection and run_ids:
            stored_selection = [run_ids[-1]]
        st.session_state["its_selected_runs"] = stored_selection

        auto_topology = _infer_default_topology(backend, layout, stored_selection)
        if auto_topology:
            current_topology = Path(
                st.session_state.get("its_topology_path", "")
            ).expanduser()
            if not st.session_state.get("its_topology_path") or not current_topology.exists():
                st.session_state["its_topology_path"] = str(auto_topology)

        default_spec = _default_feature_spec_path(layout)
        if default_spec and not st.session_state.get("its_feature_spec_path"):
            st.session_state["its_feature_spec_path"] = str(default_spec)
        if not st.session_state.get("its_lag_times_text"):
            st.session_state["its_lag_times_text"] = _format_lag_sequence(
                st.session_state.get("its_lag_times", [])
            )

        pending_topology = st.session_state.get(_ITS_PENDING_TOPOLOGY)
        if pending_topology:
            st.session_state["its_topology_path"] = str(pending_topology)
            st.session_state[_ITS_PENDING_TOPOLOGY] = None
        pending_spec = st.session_state.get(_ITS_PENDING_FEATURE_SPEC)
        if pending_spec:
            st.session_state["its_feature_spec_path"] = str(pending_spec)
            st.session_state[_ITS_PENDING_FEATURE_SPEC] = None

        # Data Selection
        with st.expander("Data Selection", expanded=True):
            selected_runs = render_shard_selection_table(
                "Shard groups",
                shard_groups,
                state_key="its_selected_runs",
                default_behavior="latest",
                help_text=SHARD_SELECTOR_HELP,
            )

            if selected_runs:
                profile_summary = summarize_selected_feature_profiles(
                    shard_groups, selected_runs
                )
                if len(profile_summary["feature_types"]) > 1:
                    st.warning(
                        "Selected shard groups mix CV and molecular features. "
                        "ITS diagnostics assume a consistent feature basis."
                    )
                elif profile_summary["feature_types"]:
                    detected_type = next(iter(profile_summary["feature_types"]))
                    st.caption(f"Detected shard feature type: {detected_type}")
                try:
                    selected_paths = select_shard_paths(shard_groups, selected_runs)
                    _, selection_text = _summarize_selected_shards(selected_paths)
                    st.caption(f"Using {len(selected_paths)} shard files.")
                    if selection_text:
                        st.caption(selection_text)
                except (ValueError, Exception) as exc:
                    st.warning(f"Issue with selection: {exc}")
                    selected_paths = []
            else:
                st.info("Select at least one shard group to compute implied timescales.")
                selected_paths = []

        frames_button = st.button(
            "Frames per shard",
            key="its_frames_per_shard",
            disabled=not selected_paths,
        )

        if frames_button:
            if not selected_paths:
                st.error("Select shard files before plotting frame counts.")
            else:
                try:
                    with st.spinner("Loading shard frame counts..."):
                        resolved_paths = [Path(p) for p in selected_paths]
                        frame_counts = _shard_frame_counts(resolved_paths)
                except Exception as exc:
                    st.error(f"Unable to load shard frame counts: {exc}")
                else:
                    labels = [path.stem for path in resolved_paths]
                    fig = create_shard_frame_histogram(frame_counts, labels)
                    st.pyplot(fig, clear_figure=True, width="stretch")
                    total_frames = sum(frame_counts)
                    avg_frames = (
                        total_frames / len(frame_counts) if frame_counts else 0.0
                    )
                    metric_cols = st.columns(2)
                    metric_cols[0].metric("Sum of frames", f"{total_frames:,}")
                    metric_cols[1].metric(
                        "Average frames per shard", f"{avg_frames:,.2f}"
                    )

        with st.expander("Configuration", expanded=True):

            path_cols = st.columns(2)
            path_cols[0].text_input(
                "Topology (PDB)",
                key="its_topology_path",
                help="Structure associated with the selected shard trajectories.",
            )
            path_cols[1].text_input(
                "Feature specification",
                key="its_feature_spec_path",
                help="Path to the feature_spec.yaml used when emitting the shards.",
            )

            param_cols = st.columns(3)
            param_cols[0].number_input(
                "TICA components",
                min_value=1,
                max_value=128,
                value=int(st.session_state.get("its_tica_dim", 10)),
                step=1,
                key="its_tica_dim",
                help="Number of TICA components retained before clustering.",
            )
            param_cols[1].number_input(
                "Clusters",
                min_value=2,
                max_value=5000,
                value=int(st.session_state.get("its_n_clusters", 200)),
                step=10,
                key="its_n_clusters",
                help="Number of microstates for MSM estimation.",
            )
            param_cols[2].number_input(
                "TICA lag",
                min_value=1,
                max_value=1000,
                value=int(st.session_state.get("its_tica_lag", 10)),
                step=1,
                key="its_tica_lag",
                help="Lag (in MD steps) used during the TICA projection.",
            )
            st.text_input(
                "Lag times (steps)",
                key="its_lag_times_text",
                help="Comma- or semicolon-separated lag times for the ITS scan.",
            )

        compute_btn = st.button(
            "Compute implied timescales", type="primary"
        )


        if compute_btn:
            try:
                lag_values = _parse_lag_sequence(
                    st.session_state.get("its_lag_times_text", "")
                )
                st.session_state["its_lag_times"] = lag_values
                if not selected_paths:
                    raise ValueError(
                        "Select at least one shard group to compute implied timescales."
                    )

                topology_str = st.session_state.get("its_topology_path", "").strip()
                if not topology_str:
                    raise ValueError(
                        "Provide the topology PDB path associated with the selected shards."
                    )
                feature_spec_str = st.session_state.get(
                    "its_feature_spec_path", ""
                ).strip()
                if not feature_spec_str:
                    raise ValueError(
                        "Provide the feature specification path for the selected shards."
                    )

                topo_path = Path(topology_str).expanduser().resolve()
                spec_path = Path(feature_spec_str).expanduser().resolve()

                result = calculate_its(
                    data_directory=layout.shards_dir,
                    topology_path=topo_path,
                    feature_spec_path=spec_path,
                    n_clusters=int(st.session_state["its_n_clusters"]),
                    tica_dim=int(st.session_state["its_tica_dim"]),
                    lag_times=lag_values,
                    shard_paths=selected_paths,
                    tica_lag=int(st.session_state["its_tica_lag"]),
                )
            except Exception as exc:
                st.session_state[_LAST_ITS_RESULT] = None
                st.session_state[_ITS_FEEDBACK] = (
                    "error",
                    f"Implied timescale computation failed: {exc}",
                )
            else:
                st.session_state[_ITS_PENDING_TOPOLOGY] = str(topo_path)
                st.session_state[_ITS_PENDING_FEATURE_SPEC] = str(spec_path)
                st.session_state[_LAST_ITS_RESULT] = result
                st.session_state[_ITS_FEEDBACK] = (
                    "success",
                    f"Computed implied timescales for {len(result.get('lag_times', []))} lag values.",
                )
                # No need to rerun - session state changes will trigger automatic rerun

        feedback = st.session_state.get(_ITS_FEEDBACK)
        if isinstance(feedback, tuple) and len(feedback) == 2:
            level, message = feedback
            renderer = getattr(st, level, st.info)
            renderer(message)
            st.session_state[_ITS_FEEDBACK] = None

        its_result = st.session_state.get(_LAST_ITS_RESULT)
        if isinstance(its_result, Mapping):
            lag_series = its_result.get("lag_times", [])
            times_series = its_result.get("timescales", [])
            try:
                fig = plot_its(lag_series, times_series)
            except Exception as exc:
                st.warning(f"Could not render implied timescale plot: {exc}")
            else:
                st.subheader("Implied Timescales Plot")
                st.pyplot(fig, clear_figure=True, width="stretch")

            table = _timescales_dataframe(lag_series, times_series)
            if not table.empty:
                st.subheader("Implied Timescale Table")
                st.dataframe(table, width="stretch")

            errors = its_result.get("errors") or {}
            if errors:
                st.subheader("Per-lag Diagnostics")
                for lag, message in sorted(errors.items()):
                    st.warning(f"Lag {lag}: {message}")

            metadata = its_result.get("metadata")
            if metadata:
                st.subheader("Metadata")
                st.json(metadata)
