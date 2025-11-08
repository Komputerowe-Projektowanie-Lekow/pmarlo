"""Run discovery tab for discovering and inspecting simulation runs."""

import streamlit as st
import pandas as pd

from core.context import AppContext
from core.session import _RUN_PENDING, _LAST_SIM
from backend.validation import RunStatus


def render_run_discovery_tab(ctx: AppContext) -> None:
    """Render the run discovery & validation tab."""
    backend = ctx.backend

    st.header("Simulation Run Discovery & Validation")

    col1, col2 = st.columns([1, 1])

    with col1:
        if st.button("Scan All Runs", type="primary", key="rd_scan_all_runs"):
            with st.spinner("Scanning simulation directories..."):
                validations = backend.discover_all_runs()
                st.session_state["validations"] = validations
                st.success(f"Found {len(validations)} run directories")

    with col2:
        if st.button("Get Summary", key="rd_get_summary"):
            summary = backend.get_validation_summary()
            st.session_state["validation_summary"] = summary

    if "validation_summary" in st.session_state:
        summary = st.session_state["validation_summary"]
        st.subheader("Summary Statistics")

        cols = st.columns(5)
        cols[0].metric("Total Runs", summary["total_runs"])
        cols[1].metric("In State", summary["in_state"])
        cols[2].metric("Not in State", summary["not_in_state"])
        cols[3].metric("Has Shards", summary["has_shards"])
        cols[4].metric("Can Create Shards", summary["can_create_shards"])

        st.subheader("Status Breakdown")
        status_df = pd.DataFrame([
            {"Status": k, "Count": v}
            for k, v in summary["status_counts"].items()
        ])
        st.dataframe(status_df, use_container_width=True)

    if "validations" in st.session_state:
        validations = st.session_state["validations"]

        st.subheader(f"All Discovered Runs ({len(validations)})")

        filter_col1, filter_col2, filter_col3 = st.columns(3)

        with filter_col1:
            show_status = st.multiselect(
                "Filter by Status",
                options=[s.value for s in RunStatus],
                default=[s.value for s in RunStatus],
                key="rd_filter_status",
            )

        with filter_col2:
            show_in_state = st.selectbox(
                "State Filter",
                options=["All", "In State", "Not in State"],
                index=0,
                key="rd_filter_state",
            )

        with filter_col3:
            show_shards = st.selectbox(
                "Shard Filter",
                options=["All", "Has Shards", "No Shards"],
                index=0,
                key="rd_filter_shards",
            )

        filtered = validations
        if show_status:
            filtered = [v for v in filtered if v.status.value in show_status]
        if show_in_state == "In State":
            filtered = [v for v in filtered if v.in_state]
        elif show_in_state == "Not in State":
            filtered = [v for v in filtered if not v.in_state]
        if show_shards == "Has Shards":
            filtered = [v for v in filtered if v.has_shards]
        elif show_shards == "No Shards":
            filtered = [v for v in filtered if not v.has_shards]

        st.caption(f"Showing {len(filtered)} of {len(validations)} runs")

        data_rows = []
        for v in filtered:
            status_icon = {
                RunStatus.COMPLETE: "",
                RunStatus.INCOMPLETE: "️",
                RunStatus.EMPTY: "",
                RunStatus.MISSING_ANALYSIS: "",
                RunStatus.MISSING_DEMUX: "",  # Usable but using replica files
                RunStatus.MISSING_STATE_ENTRY: "",
                RunStatus.IN_PROGRESS: "",
                RunStatus.CORRUPTED: "",
            }.get(v.status, "")

            data_rows.append({
                "Status": f"{status_icon} {v.status.value}",
                "Run ID": v.run_id,
                "In State": "Yes" if v.in_state else "No",
                "Has Shards": f"{v.shard_count}" if v.has_shards else "No",
                "Trajectories": v.metadata.get("trajectory_count", 0),
                "Demux": v.metadata.get("demux_count", 0),
                "Issues": len(v.issues),
                "Can Create Shards": "Yes" if v.can_create_shards else "No",
            })

        df = pd.DataFrame(data_rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.subheader("Detailed Inspection")

        run_ids = [v.run_id for v in filtered]
        if run_ids:
            selected_run_id = st.selectbox(
                "Select a run to inspect",
                options=run_ids,
                key="rd_validation_inspect_run",
            )

            selected_validation = next(
                (v for v in filtered if v.run_id == selected_run_id), None
            )

            if selected_validation:
                st.markdown(f"### {selected_run_id}")

                info_col1, info_col2, info_col3 = st.columns(3)
                info_col1.metric("Status", selected_validation.status.value)
                info_col2.metric("In State", "Yes" if selected_validation.in_state else "No")
                info_col3.metric("Shards", selected_validation.shard_count if selected_validation.has_shards else "None")

                if selected_validation.metadata:
                    st.markdown("#### Metadata")
                    metadata_display = dict(selected_validation.metadata)
                    if "provenance" in metadata_display:
                        metadata_display.pop("provenance")
                    st.json(metadata_display)

                if selected_validation.issues:
                    st.markdown("#### Issues")
                    for issue in selected_validation.issues:
                        severity_color = {
                            "error": "",
                            "warning": "",
                            "info": "ℹ",
                        }.get(issue.severity, "")

                        st.markdown(f"{severity_color} **{issue.severity.upper()}**: {issue.message}")
                        if issue.details:
                            st.caption(issue.details)

                    # Concise hints for common statuses
                    if selected_validation.status.value == "missing_demux":
                        st.info("Replica trajectories exist but demux output is absent.")

                    if selected_validation.status.value == "corrupted":
                        checkpoint_count = selected_validation.metadata.get("checkpoint_count", 0)
                        latest_checkpoint = selected_validation.metadata.get("latest_checkpoint_step")
                        if checkpoint_count > 0:
                            st.warning(
                                f"Corrupted run detected — {checkpoint_count} checkpoint(s), latest at step {latest_checkpoint}."
                            )
                        else:
                            st.error("Corrupted run without checkpoints; delete and re-run.")
                else:
                    st.success("No issues found")

                st.markdown("#### Actions")

                # Check if this is a corrupted run with checkpoints
                checkpoint_count = selected_validation.metadata.get("checkpoint_count", 0)
                has_checkpoints = checkpoint_count > 0
                is_corrupted = selected_validation.status.value == "corrupted"

                plan_available = bool(selected_validation.metadata.get("run_plan"))
                can_resume = has_checkpoints and selected_validation.status in {
                    RunStatus.CORRUPTED,
                    RunStatus.IN_PROGRESS,
                }
                if can_resume and not plan_available:
                    st.warning("Checkpoint found but configuration metadata is missing.")
                if can_resume and plan_available:
                    st.markdown("##### Recovery")
                    resume_col, info_col = st.columns([2, 1])
                    with resume_col:
                        resume_disabled = bool(st.session_state.get(_RUN_PENDING, False))
                        if st.button(
                            "Resume Run",
                            key=f"rd_resume_{selected_run_id}",
                            type="primary",
                            disabled=resume_disabled,
                        ):
                            try:
                                st.session_state[_RUN_PENDING] = True
                                with st.spinner("Resuming from checkpoint..."):
                                    result = backend.resume_run_from_checkpoint(selected_run_id)
                                st.success(f"Resume started: {result.run_id}")
                                st.session_state[_LAST_SIM] = result
                                # Force refresh so updated status is visible
                                if "validations" in st.session_state:
                                    del st.session_state["validations"]
                                st.rerun()
                            except Exception as exc:
                                st.error(f"Resume failed: {exc}")
                            finally:
                                st.session_state[_RUN_PENDING] = False
                    with info_col:
                        st.metric(
                            "Latest checkpoint",
                            f"Step {selected_validation.metadata.get('latest_checkpoint_step', '—')}",
                        )

                    st.markdown("##### Other Actions")

                action_col1, action_col2, action_col3 = st.columns(3)

                with action_col1:
                    if not selected_validation.in_state and selected_validation.is_valid:
                        if st.button("Add to State", key=f"rd_add_state_{selected_run_id}"):
                            success = backend.add_run_to_state(selected_run_id)
                            if success:
                                st.success(f"Added {selected_run_id} to state")
                                st.rerun()
                            else:
                                st.error(f"Failed to add {selected_run_id} to state")

                with action_col2:
                    if st.button("Open Directory", key=f"rd_open_dir_{selected_run_id}"):
                        import os
                        import platform
                        run_dir = str(selected_validation.run_dir)

                        if platform.system() == "Windows":
                            os.startfile(run_dir)
                        elif platform.system() == "Darwin":
                            import subprocess
                            subprocess.Popen(["open", run_dir])
                        else:
                            import subprocess
                            subprocess.Popen(["xdg-open", run_dir])

                with action_col3:
                    # Show delete button for all runs
                    delete_key = f"rd_delete_{selected_run_id}"
                    confirm_key = f"rd_delete_confirm_{selected_run_id}"

                    # Check if we're in confirmation mode
                    if confirm_key not in st.session_state:
                        st.session_state[confirm_key] = False

                    if not st.session_state[confirm_key]:
                        # Show initial delete button
                        button_type = "secondary" if selected_validation.status == RunStatus.CORRUPTED else "secondary"
                        if st.button("Delete Run", key=delete_key, type=button_type):
                            st.session_state[confirm_key] = True
                            st.rerun()
                    else:
                        # Show confirmation
                        st.warning(f"Really delete {selected_run_id}?")
                        confirm_col1, confirm_col2 = st.columns(2)
                        with confirm_col1:
                            if st.button("Yes, Delete", key=f"{delete_key}_yes", type="primary"):
                                import shutil
                                try:
                                    # Remove from state if present
                                    if selected_validation.in_state:
                                        # Find and remove from state using the remove_run method
                                        for idx, entry in enumerate(backend.state.runs):
                                            if entry.get("run_id") == selected_run_id:
                                                backend.state.remove_run(idx)
                                                break

                                    # Delete the directory
                                    shutil.rmtree(selected_validation.run_dir)
                                    st.success(f"Deleted run {selected_run_id}")
                                    st.session_state[confirm_key] = False
                                    # Clear validations to force rescan
                                    if "validations" in st.session_state:
                                        del st.session_state["validations"]
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Failed to delete run: {e}")
                                    st.session_state[confirm_key] = False
                        with confirm_col2:
                            if st.button("Cancel", key=f"{delete_key}_no"):
                                st.session_state[confirm_key] = False
                                st.rerun()

                # Additional info for specific statuses
                if selected_validation.status == RunStatus.CORRUPTED:
                    st.caption("This is a corrupted run - deletion is recommended.")
                elif selected_validation.can_create_shards:
                    st.info("This run is ready for shard creation. Go to the Sampling tab to create shards.")

        missing_state = backend.get_missing_state_entries()
        if missing_state:
            st.subheader(f"Runs Not in State ({len(missing_state)})")
            st.markdown(
                """
                These runs exist on disk and are valid but are not tracked in state.json.
                Add them to state to make them available in the standard UI.
                """
            )

            if st.button("Add All Valid Runs to State", type="primary", key="rd_add_all_to_state"):
                with st.spinner(f"Adding {len(missing_state)} runs to state..."):
                    added = 0
                    failed = 0
                    for v in missing_state:
                        if backend.add_run_to_state(v.run_id):
                            added += 1
                        else:
                            failed += 1

                    if added > 0:
                        st.success(f"Added {added} runs to state")
                    if failed > 0:
                        st.warning(f"Failed to add {failed} runs")
                    st.rerun()

            missing_df = pd.DataFrame([
                {
                    "Run ID": v.run_id,
                    "Status": v.status.value,
                    "Trajectories": v.metadata.get("trajectory_count", 0),
                    "Demux": v.metadata.get("demux_count", 0),
                }
                for v in missing_state
            ])
            st.dataframe(missing_df, use_container_width=True, hide_index=True)
