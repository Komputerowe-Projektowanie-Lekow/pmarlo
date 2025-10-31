import streamlit as st
from app.core.context import AppContext

def render_sidebar(ctx: AppContext) -> None:
    summary = backend.sidebar_summary()
    with st.sidebar:
        st.title("Workspace")
        st.caption(str(layout.workspace_dir))
        cols = st.columns(2)
        cols[0].metric("Sim runs", summary.get("runs", 0))
        cols[1].metric("Shard files", summary.get("shards", 0))
        cols = st.columns(3)
        cols[0].metric("Models", summary.get("models", 0))
        cols[1].metric("Bundles", summary.get("builds", 0))
        cols[2].metric("Conformation Sets", summary.get("conformations", 0))
        st.divider()
        inputs = layout.available_inputs()
        if inputs:
            st.write("Available inputs:")
            for pdb in inputs:
                st.caption(pdb.name)
        else:
            st.info("Drop prepared PDB files into app_intputs/ to get started.")
