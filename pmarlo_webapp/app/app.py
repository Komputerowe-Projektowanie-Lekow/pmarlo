from __future__ import annotations

import streamlit as st

from core.context import build_context
from core import ensure_session_defaults, consume_pending_training_config
from ui.sidebar import render_sidebar
from ui.theme import inject_global_styles

from tabs.sampling import render_sampling_tab
from tabs.training import render_training_tab
from tabs.msm_fes import render_msm_fes_tab
from tabs.conformations import render_conformations_tab
from tabs.validation import render_validation_tab
from tabs.run_discovery import render_run_discovery_tab
from tabs.model_preview import render_model_preview
from tabs.assets import render_assets_tab
from tabs.its import render_its_tab
from tabs.ck_its_auto import render_ck_its_tab


def main() -> None:
    """Main application entry point."""
    st.set_page_config(page_title="PMARLO Joint Learning", layout="wide")
    inject_global_styles()

    # Initialize session state and consume any pending configs
    ensure_session_defaults()
    consume_pending_training_config()

    # Build shared context (backend + layout)
    ctx = build_context()

    # Render sidebar with workspace summary
    render_sidebar(ctx)

    # Create top-level tabs
    tab_conformation, tab_its_container = st.tabs([
        "Conformation Analysis",
        "Implied Timescales",
    ])

    # Conformation Analysis tab contains nested tabs
    with tab_conformation:
        (
            tab_sampling,
            tab_training,
            tab_msm_fes,
            tab_conformations,
            tab_validation,
            tab_run_discovery,
            tab_model_preview,
            tab_assets,
        ) = st.tabs([
            "Sampling",
            "Model Training",
            "MSM/FES Analysis",
            "Conformation Analysis",
            "Free Energy Validation",
            "Run Discovery",
            "Model Preview",
            "Assets",
        ])

        with tab_sampling:
            render_sampling_tab(ctx)

        with tab_training:
            render_training_tab(ctx)

        with tab_msm_fes:
            render_msm_fes_tab(ctx)

        with tab_conformations:
            render_conformations_tab(ctx)

        with tab_validation:
            render_validation_tab(ctx)

        with tab_run_discovery:
            render_run_discovery_tab(ctx)

        with tab_model_preview:
            render_model_preview(ctx)

        with tab_assets:
            render_assets_tab(ctx)

    # Implied Timescales tab with nested ITS and CK+ITS subtabs
    with tab_its_container:
        tab_its_simple, tab_its_ck = st.tabs([
            "ITS",
            "ITS with CK Analysis",
        ])

        with tab_its_simple:
            render_its_tab(ctx)

        with tab_its_ck:
            render_ck_its_tab(ctx)

    st.caption("Run with: poetry run streamlit run pmarlo_webapp/app/app.py")


if __name__ == "__main__":
    main()
