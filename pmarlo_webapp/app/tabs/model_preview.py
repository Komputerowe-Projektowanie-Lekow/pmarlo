import streamlit as st
from pathlib import Path

from core.context import AppContext
from core.session import (
    _MODEL_PREVIEW_SELECTION,
    _MODEL_PREVIEW_RESULT,
    _LAST_TRAIN,
    _TRAIN_CONFIG_PENDING,
    _TRAIN_FEEDBACK,
)
from core.view_helpers import (
    _model_entry_label,
    _show_build_outputs,
    _render_deeptica_summary,
)
from core.parsers import _format_tau_schedule
from backend.utils import _sanitize_artifacts

def render_model_preview(ctx: AppContext) -> None:
    """Render the Model Preview Tab."""
    backend = ctx.backend
    layout = ctx.layout

    st.header("Model Preview")
    models = backend.list_models()
    if not models:
        st.info("Train or load a Deep-TICA model to preview its artifacts.")
    else:
        indices = list(range(len(models)))
        if (
                st.session_state[_MODEL_PREVIEW_SELECTION] is None
                or st.session_state[_MODEL_PREVIEW_SELECTION] not in indices
        ):
            st.session_state[_MODEL_PREVIEW_SELECTION] = indices[-1]

        selected_idx = st.selectbox(
            "Stored models",
            options=indices,
            format_func=lambda idx: _model_entry_label(models[idx], idx),
            key=_MODEL_PREVIEW_SELECTION,
        )
        entry = models[int(selected_idx)]
        bundle_path = entry.get("bundle", "")
        st.caption(f"Bundle path: {bundle_path}")
        checkpoint_dir = entry.get("checkpoint_dir")
        if checkpoint_dir:
            st.caption(f"Checkpoint directory: {checkpoint_dir}")

        meta_cols = st.columns(4)
        lag_val = entry.get("lag")
        seed_val = entry.get("seed")
        temperature_val = entry.get("temperature")
        epochs_per_tau = entry.get("epochs_per_tau")
        meta_cols[0].metric("Lag (steps)", lag_val if lag_val is not None else "n/a")
        meta_cols[1].metric("Seed", seed_val if seed_val is not None else "n/a")
        meta_cols[2].metric(
            "Temperature (K)",
            f"{float(temperature_val):.1f}" if temperature_val is not None else "n/a",
        )
        meta_cols[3].metric(
            "Epochs per tau",
            epochs_per_tau if epochs_per_tau is not None else "n/a",
        )

        tau_schedule = entry.get("tau_schedule") or []
        hidden_layers = entry.get("hidden") or []
        st.write(
            f"Tau schedule: {_format_tau_schedule(tau_schedule) if tau_schedule else 'n/a'}"
        )
        st.write(
            "Hidden layers: "
            + (", ".join(str(int(h)) for h in hidden_layers) if hidden_layers else "n/a")
        )

        metrics_payload = entry.get("metrics")
        if metrics_payload:
            with st.expander("Training metrics (state snapshot)", expanded=False):
                st.json(_sanitize_artifacts(metrics_payload))

        preview_result = backend.load_model(int(selected_idx))
        if preview_result is None:
            st.error(
                f"Model bundle {bundle_path!s} is missing or unreadable. "
                "Ensure the bundle exists before previewing."
            )
            st.session_state[_MODEL_PREVIEW_RESULT] = None
        else:
            st.session_state[_MODEL_PREVIEW_RESULT] = preview_result
            _show_build_outputs(preview_result)
            summary_artifact = (
                preview_result.build_result.artifacts.get("mlcv_deeptica")
                if preview_result.build_result and preview_result.build_result.artifacts
                else None
            )
            if summary_artifact:
                _render_deeptica_summary(summary_artifact)

            if st.button(
                    "Load into Training tab",
                    key="model_preview_load_training",
                    help="Populate the training tab with this model's configuration.",
            ):
                try:
                    cfg_loaded = backend.training_config_from_entry(entry)
                except Exception as exc:
                    st.error(f"Could not reconstruct training configuration: {exc}")
                else:
                    st.session_state[_LAST_TRAIN] = preview_result
                    st.session_state[_TRAIN_CONFIG_PENDING] = cfg_loaded
                    st.session_state[_TRAIN_FEEDBACK] = (
                        "success",
                        f"Loaded model {Path(bundle_path).name} into the training tab.",
                    )
                    st.rerun()
