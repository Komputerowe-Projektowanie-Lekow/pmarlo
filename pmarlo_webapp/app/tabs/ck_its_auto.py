from pathlib import Path
from typing import Mapping

import numpy as np
import streamlit as st

from core.context import AppContext
from core.session import (
    _CK_ITS_PENDING_TOPOLOGY,
    _CK_ITS_PENDING_FEATURE_SPEC,
    _LAST_CK_ITS_RESULT,
    _CK_ITS_FEEDBACK,
)
from core.parsers import _format_lag_sequence, _parse_lag_sequence
from core.view_helpers import (
    SHARD_SELECTOR_HELP,
    _infer_default_topology,
    _default_feature_spec_path,
    _summarize_selected_shards,
    render_shard_selection_table,
    summarize_selected_feature_profiles,
    format_feature_variable_caption,
)
from pmarlo.api import select_shard_paths


def render_ck_its_tab(ctx: AppContext) -> None:
    """Render the CK+ITS automatic lag selection tab."""
    backend = ctx.backend
    layout = ctx.layout

    st.header("ITS with CK Analysis - Automatic Lag Selection")
    st.markdown(
        """
        This tool combines **Implied Timescales (ITS)** analysis with **Chapman-Kolmogorov (CK)** validation
        to automatically select the optimal lag time for MSM construction.

        The algorithm selects the **smallest lag τ** that:
        - Passes CK test (max error ≤ threshold, typically 10-15%)
        - Has high coverage (giant connected component ≥ 98%)
        - Has sufficient statistics (median microstate count ≥ 100-200)
        """
    )

    shard_groups = backend.shard_summaries()
    if not shard_groups:
        st.info("Emit shard batches to perform CK+ITS lag selection.")
        return

    run_ids = [str(entry.get("run_id")) for entry in shard_groups]
    stored_selection = [
        run_id
        for run_id in st.session_state.get("ck_its_selected_runs", [])
        if run_id in run_ids
    ]
    if not stored_selection and run_ids:
        stored_selection = [run_ids[-1]]
    st.session_state["ck_its_selected_runs"] = stored_selection

    auto_topology = _infer_default_topology(backend, layout, stored_selection)
    if auto_topology:
        current_topology = Path(
            st.session_state.get("ck_its_topology_path", "")
        ).expanduser()
        if not st.session_state.get("ck_its_topology_path") or not current_topology.exists():
            st.session_state["ck_its_topology_path"] = str(auto_topology)

    default_spec = _default_feature_spec_path(layout)
    if default_spec and not st.session_state.get("ck_its_feature_spec_path"):
        st.session_state["ck_its_feature_spec_path"] = str(default_spec)

    # Default candidate lags
    if not st.session_state.get("ck_its_tau_candidates_text"):
        st.session_state["ck_its_tau_candidates_text"] = "25, 50, 75, 100"

    # Default horizons
    if not st.session_state.get("ck_its_horizons_text"):
        st.session_state["ck_its_horizons_text"] = "1, 2, 3, 4, 5"

    pending_topology = st.session_state.get(_CK_ITS_PENDING_TOPOLOGY)
    if pending_topology:
        st.session_state["ck_its_topology_path"] = str(pending_topology)
        st.session_state[_CK_ITS_PENDING_TOPOLOGY] = None

    pending_spec = st.session_state.get(_CK_ITS_PENDING_FEATURE_SPEC)
    if pending_spec:
        st.session_state["ck_its_feature_spec_path"] = str(pending_spec)
        st.session_state[_CK_ITS_PENDING_FEATURE_SPEC] = None

    # Data Selection
    with st.expander("Data Selection", expanded=True):
        selected_runs = render_shard_selection_table(
            "Shard groups",
            shard_groups,
            state_key="ck_its_selected_runs",
            default_behavior="latest",
            help_text=SHARD_SELECTOR_HELP,
        )

        if selected_runs:
            profile_summary = summarize_selected_feature_profiles(
                shard_groups, selected_runs
            )
            if len(profile_summary["feature_types"]) > 1:
                st.warning(
                    "Selected shard groups mix different feature types. "
                    "CK/ITS diagnostics expect a consistent feature basis."
                )
            elif profile_summary["feature_types"]:
                detected_type = next(iter(profile_summary["feature_types"]))
                st.caption(f"Detected shard feature type: {detected_type}")
            feature_variable_caption = format_feature_variable_caption(profile_summary)
            if feature_variable_caption:
                st.caption(
                    "Variables detected in selected shards: "
                    + feature_variable_caption
                )
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
            st.info("Select at least one shard group to perform CK+ITS lag selection.")
            selected_paths = []

    with st.expander("Configuration", expanded=True):
        path_cols = st.columns(2)
        path_cols[0].text_input(
            "Topology (PDB)",
            key="ck_its_topology_path",
            help="Structure associated with the selected shard trajectories.",
        )
        path_cols[1].text_input(
            "Feature specification",
            key="ck_its_feature_spec_path",
            help="Path to the feature_spec.yaml used when emitting the shards.",
        )

        param_cols = st.columns(3)
        param_cols[0].number_input(
            "TICA components",
            min_value=1,
            max_value=128,
            value=int(st.session_state.get("ck_its_tica_dim", 10)),
            step=1,
            key="ck_its_tica_dim",
            help="Number of TICA components retained before clustering.",
        )
        param_cols[1].number_input(
            "Clusters (microstates)",
            min_value=2,
            max_value=5000,
            value=int(st.session_state.get("ck_its_n_clusters", 200)),
            step=10,
            key="ck_its_n_clusters",
            help="Number of microstates for MSM estimation.",
        )
        param_cols[2].number_input(
            "TICA lag",
            min_value=1,
            max_value=1000,
            value=int(st.session_state.get("ck_its_tica_lag", 10)),
            step=1,
            key="ck_its_tica_lag",
            help="Lag (in MD steps) used during the TICA projection.",
        )

        st.text_input(
            "Candidate lag times (steps)",
            key="ck_its_tau_candidates_text",
            help="Comma-separated lag times to evaluate (e.g., 25, 50, 75, 100).",
        )

        st.text_input(
            "CK horizons",
            key="ck_its_horizons_text",
            help="Comma-separated CK test horizons k (e.g., 1, 2, 3, 4, 5).",
        )

        threshold_cols = st.columns(4)
        threshold_cols[0].number_input(
            "CK error threshold",
            min_value=0.01,
            max_value=0.50,
            value=float(st.session_state.get("ck_its_ck_threshold", 0.15)),
            step=0.01,
            key="ck_its_ck_threshold",
            help="Maximum acceptable CK error (default: 0.15 = 15%).",
        )
        threshold_cols[1].number_input(
            "Coverage threshold",
            min_value=0.80,
            max_value=1.00,
            value=float(st.session_state.get("ck_its_coverage_threshold", 0.98)),
            step=0.01,
            key="ck_its_coverage_threshold",
            help="Minimum giant connected component coverage (default: 0.98 = 98%).",
        )
        threshold_cols[2].number_input(
            "Min median count",
            min_value=10,
            max_value=1000,
            value=int(st.session_state.get("ck_its_min_median_count", 100)),
            step=10,
            key="ck_its_min_median_count",
            help="Minimum median microstate count (default: 100).",
        )
        threshold_cols[3].number_input(
            "Diagonal mass threshold",
            min_value=0.0,
            max_value=1.0,
            value=float(st.session_state.get("ck_its_diag_mass_threshold", 0.6)),
            step=0.05,
            key="ck_its_diag_mass_threshold",
            help="Minimum acceptable average diagonal probability (default: 0.6).",
        )

    compute_btn = st.button(
        "Run CK+ITS Lag Selection", type="primary", disabled=not selected_paths
    )

    feedback = st.session_state.get(_CK_ITS_FEEDBACK)

    if compute_btn:
        try:
            tau_candidates = _parse_lag_sequence(
                st.session_state.get("ck_its_tau_candidates_text", "25,50,75,100")
            )
            horizons = _parse_lag_sequence(
                st.session_state.get("ck_its_horizons_text", "1,2,3,4,5")
            )

            if not selected_paths:
                raise ValueError(
                    "Select at least one shard group to perform CK+ITS lag selection."
                )

            topology_str = st.session_state.get("ck_its_topology_path", "").strip()
            if not topology_str:
                raise ValueError(
                    "Provide the topology PDB path associated with the selected shards."
                )

            feature_spec_str = st.session_state.get(
                "ck_its_feature_spec_path", ""
            ).strip()
            if not feature_spec_str:
                raise ValueError(
                    "Provide the feature specification path for the selected shards."
                )

            topo_path = Path(topology_str).expanduser().resolve()
            spec_path = Path(feature_spec_str).expanduser().resolve()

            with st.spinner("Running CK+ITS lag selection (this may take several minutes)..."):
                result = backend.run_ck_its_selection(
                    shard_paths=selected_paths,
                    topology_path=topo_path,
                    feature_spec_path=spec_path,
                    n_clusters=int(st.session_state["ck_its_n_clusters"]),
                    tica_dim=int(st.session_state["ck_its_tica_dim"]),
                    tica_lag=int(st.session_state["ck_its_tica_lag"]),
                    tau_candidates=tau_candidates,
                    horizons=horizons,
                    ck_threshold=float(st.session_state["ck_its_ck_threshold"]),
                    coverage_threshold=float(st.session_state["ck_its_coverage_threshold"]),
                    min_median_count=int(st.session_state["ck_its_min_median_count"]),
                    diag_mass_threshold=float(
                        st.session_state["ck_its_diag_mass_threshold"]
                    ),
                )

        except Exception as exc:
            st.session_state[_LAST_CK_ITS_RESULT] = None
            st.session_state[_CK_ITS_FEEDBACK] = (
                "error",
                f"CK+ITS lag selection failed: {exc}",
            )
        else:
            st.session_state[_CK_ITS_PENDING_TOPOLOGY] = str(topo_path)
            st.session_state[_CK_ITS_PENDING_FEATURE_SPEC] = str(spec_path)
            st.session_state[_LAST_CK_ITS_RESULT] = result
            selected_lag = None
            if isinstance(result, Mapping):
                selected_lag = result.get("selected_lag")
            elif hasattr(result, "selected_lag"):
                selected_lag = result.selected_lag
            if selected_lag is not None:
                feedback_message = f"Selected optimal lag: {selected_lag} steps"
            else:
                feedback_message = "CK+ITS lag selection completed."
            st.session_state[_CK_ITS_FEEDBACK] = (
                "success",
                feedback_message,
            )
            # No need to rerun - session state changes will trigger automatic rerun

    feedback = st.session_state.get(_CK_ITS_FEEDBACK)
    if isinstance(feedback, tuple) and len(feedback) == 2:
        level, message = feedback
        renderer = getattr(st, level, st.info)
        renderer(message)
        st.session_state[_CK_ITS_FEEDBACK] = None

    ck_its_result = st.session_state.get(_LAST_CK_ITS_RESULT)
    if isinstance(ck_its_result, Mapping):
        _display_ck_its_results(ck_its_result)


def _display_ck_its_results(result: Mapping) -> None:
    """Display CK+ITS selection results."""
    import pandas as pd
    import numpy as np

    selected_lag = result.get("selected_lag")
    ck_errors = result.get("ck_errors", {})
    coverage_fractions = result.get("coverage_fractions", {})
    median_counts = result.get("median_counts", {})
    macrostate_counts = result.get("macrostate_counts", {})
    passed_sanity = result.get("passed_sanity", {})
    diag_masses = result.get("diag_masses", {})
    its_timescales = result.get("its_timescales", np.array([]))
    its_lag_times = result.get("its_lag_times", np.array([]))
    diagnostics = result.get("diagnostics", {})

    # Display selected lag prominently
    st.subheader("Selected Optimal Lag")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Selected Lag", f"{selected_lag} steps")
    if selected_lag in ck_errors:
        col2.metric("CK Error", f"{ck_errors[selected_lag]:.3f}")
    if selected_lag in coverage_fractions:
        col3.metric("Coverage", f"{coverage_fractions[selected_lag]:.1%}")
    if selected_lag in diag_masses and diag_masses[selected_lag] is not None:
        col4.metric("Diagonal Mass", f"{diag_masses[selected_lag]:.3f}")

    ignored_lags = diagnostics.get("ignored_tau_candidates") or []
    max_supported_lag = diagnostics.get("max_supported_lag")
    if ignored_lags:
        if max_supported_lag is not None and max_supported_lag > 0:
            st.warning(
                "Ignored lag candidates %s because the trajectories only support "
                "lags up to %d steps."
                % (", ".join(str(lag) for lag in ignored_lags), int(max_supported_lag))
            )
        else:
            st.warning(
                "Ignored lag candidates %s because the trajectories were too short "
                "to evaluate them."
                % (", ".join(str(lag) for lag in ignored_lags))
            )

    tau_hint = diagnostics.get("tau_int")
    recommended = diagnostics.get("recommended_tau_candidates", [])
    if tau_hint is not None and np.isfinite(tau_hint) and recommended:
        st.caption(
            f"Autocorrelation guidance: τ_int≈{tau_hint:.0f} · suggested candidates {recommended}"
        )

    # Plot CK errors
    st.subheader("CK Error vs Lag Time")
    try:
        from core.view_helpers import plot_ck_errors_with_threshold

        fig = plot_ck_errors_with_threshold(
            ck_errors,
            selected_lag,
            threshold=diagnostics.get("ck_threshold", 0.15)
        )
        st.pyplot(fig, clear_figure=True, width="stretch")
    except Exception as e:
        st.warning(f"Could not render CK error plot: {e}")
        # Fallback: display as table
        if ck_errors:
            ck_df = pd.DataFrame([
                {"Lag": lag, "CK Error": error}
                for lag, error in sorted(ck_errors.items())
            ])
            st.dataframe(ck_df, width="stretch")

    # Plot ITS with selected lag highlighted
    if its_timescales.size > 0 and its_lag_times.size > 0:
        st.subheader("Implied Timescales")
        try:
            from core.view_helpers import plot_its_with_selection

            fig = plot_its_with_selection(its_lag_times, its_timescales, selected_lag)
            st.pyplot(fig, clear_figure=True, width="stretch")
        except Exception as e:
            st.warning(f"Could not render ITS plot: {e}")

    # Sanity checks table
    st.subheader("Sanity Checks")
    sanity_data = []
    for lag in sorted(ck_errors.keys()):
        n_macro = macrostate_counts.get(lag, 0)
        mode_indicator = "macro" if n_macro > 0 else "micro"
        sanity_data.append({
            "Lag": lag,
            "CK Error": f"{ck_errors.get(lag, float('nan')):.3f}",
            "Coverage": f"{coverage_fractions.get(lag, 0.0):.1%}",
            "Median Count": median_counts.get(lag, 0),
            "Diagonal Mass": (
                f"{diag_masses.get(lag, float('nan')):.3f}"
                if diag_masses.get(lag) is not None
                else "—"
            ),
            "Macrostates": n_macro if n_macro > 0 else "N/A",
            "Mode": mode_indicator,
            "Passed": "✓" if passed_sanity.get(lag, False) else "✗",
        })
    sanity_df = pd.DataFrame(sanity_data)
    st.dataframe(sanity_df, width="stretch")

    # Diagnostics
    with st.expander("Detailed Diagnostics"):
        st.json(diagnostics)
