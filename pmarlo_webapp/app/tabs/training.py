import json
from pathlib import Path

import pandas as pd
import streamlit as st

from app.core.context import AppContext
from app.core.session import (
    _TRAIN_FEEDBACK,
    _LAST_TRAIN,
    _LAST_TRAIN_CONFIG,
    _TRAIN_CONFIG_PENDING,
)
from app.core.constants import DEEPTICA_SKIP_MESSAGE
from app.core.parsers import _parse_tau_schedule
from app.core.view_helpers import (
    _select_shard_paths,
    _summarize_selected_shards,
    _show_build_outputs,
    _render_deeptica_summary,
)
from app.backend.types import TrainingConfig, TrainingResult

def render_training_tab(ctx: AppContext) -> None:
    """Render the model training tab."""
    backend = ctx.backend
    layout = ctx.layout

    st.header("Train collective-variable model")
    feedback = st.session_state.get(_TRAIN_FEEDBACK)
    if feedback:
        if isinstance(feedback, tuple) and len(feedback) == 2:
            level, message = feedback
        else:
            level, message = ("info", str(feedback))
        display_fn = getattr(st, str(level), st.info)
        display_fn(str(message))
        st.session_state[_TRAIN_FEEDBACK] = None
    shard_groups = backend.shard_summaries()
    if not shard_groups:
        st.info("Emit shards before training a CV model.")
    else:
        # Data Selection
        with st.expander("Data Selection", expanded=True):
            run_ids = [str(entry.get("run_id")) for entry in shard_groups]
            selected_runs = st.multiselect(
                "Shard groups",
                options=run_ids,
                default=run_ids[-1:] if run_ids else [],
            )
            selected_paths = _select_shard_paths(shard_groups, selected_runs)
            try:
                _selection_runs, selection_text = _summarize_selected_shards(
                    selected_paths
                )
            except ValueError as exc:
                st.error(f"Shard selection invalid: {exc}")
                st.stop()
            st.write(f"Using {len(selected_paths)} shard files for training.")
            if selection_text:
                st.caption(selection_text)

        # Basic Training Parameters
        with st.expander("Basic Training Parameters", expanded=True):
            col_a, col_b = st.columns(2)
            lag = col_a.number_input(
                "Lag (steps)", min_value=1, value=5, step=1, key="train_lag",
                help="Time delay for computing time-lagged correlations"
            )
            temperature = col_b.number_input(
                "Reference temperature (K)",
                min_value=0.0,
                value=300.0,
                step=5.0,
                key="train_temperature",
                help="Temperature for reweighting calculations"
            )
            col_c, col_d = st.columns(2)
            seed = col_c.number_input(
                "Training seed", min_value=0, value=1337, step=1, key="train_seed",
                help="Random seed for reproducibility"
            )
            max_epochs = col_d.number_input(
                "Max epochs", min_value=20, value=200, step=10, key="train_max_epochs",
                help="Maximum number of training epochs"
            )
            patience = st.number_input(
                "Early stopping patience",
                min_value=5,
                value=25,
                step=5,
                key="train_patience",
                help="Number of epochs without improvement before stopping"
            )

        # Feature Binning
        with st.expander("Feature Binning", expanded=False):
            col_bins_a, col_bins_b = st.columns(2)
            bins_rg = col_bins_a.number_input(
                "Bins for Rg", min_value=8, value=64, step=4, key="train_bins_rg",
                help="Number of bins for radius of gyration"
            )
            bins_rmsd = col_bins_b.number_input(
                "Bins for RMSD", min_value=8, value=64, step=4, key="train_bins_rmsd",
                help="Number of bins for RMSD from reference"
            )

        # Network Architecture
        with st.expander("Network Architecture", expanded=False):
            hidden = st.text_input(
                "Hidden layer widths",
                value=st.session_state.get("train_hidden_layers", "128,128"),
                help="Comma-separated integers for the Deep-TICA network (e.g., 128,128 for two hidden layers)",
                key="train_hidden_layers",
            )
            hidden_layers = tuple(
                int(v.strip()) for v in hidden.split(",") if v.strip()
            ) or (128, 128)

        # Curriculum Learning
        with st.expander("Curriculum Learning", expanded=False):
            st.write("Configure the multi-tau curriculum training strategy")
            col_tau_a, col_tau_b, col_tau_c = st.columns(3)
            tau_raw = col_tau_a.text_input(
                "Tau schedule (steps)",
                value=st.session_state.get("train_tau_schedule", "2,5,10,20"),
                key="train_tau_schedule",
                help="Comma-separated lag times for curriculum learning"
            )
            val_tau = col_tau_b.number_input(
                "Validation tau (steps)",
                min_value=1,
                value=int(st.session_state.get("train_val_tau", 20)),
                step=1,
                key="train_val_tau",
                help="Lag time used for validation scoring"
            )
            epochs_per_tau = col_tau_c.number_input(
                "Epochs per tau",
                min_value=1,
                value=int(st.session_state.get("train_epochs_per_tau", 15)),
                step=1,
                key="train_epochs_per_tau",
                help="Number of epochs to train at each tau value"
            )

        disabled = len(selected_paths) == 0
        if st.button(
                "Train Deep-TICA model",
                type="primary",
                disabled=disabled,
                key="train_button",
        ):
            try:
                tau_values = _parse_tau_schedule(tau_raw)
            except ValueError as exc:
                st.error(f"Tau schedule error: {exc}")
            else:
                try:
                    train_cfg = TrainingConfig(
                        lag=int(lag),
                        bins={"Rg": int(bins_rg), "RMSD_ref": int(bins_rmsd)},
                        seed=int(seed),
                        temperature=float(temperature),
                        max_epochs=int(max_epochs),
                        early_stopping=int(patience),
                        hidden=hidden_layers,
                        tau_schedule=tuple(tau_values),
                        val_tau=int(val_tau),
                        epochs_per_tau=int(epochs_per_tau),
                    )

                    # Use st.status to show progress during training
                    with st.status("Training Deep-TICA model...", expanded=True) as status:
                        st.write("Loading and preparing shard data...")
                        st.write(f"- Using {len(selected_paths)} shard files")
                        if selection_text:
                            st.write(f"  Runs: {selection_text}")
                        st.write(f"- Lag: {lag}, Bins: Rg={bins_rg}, RMSD={bins_rmsd}")
                        st.write(f"- Max epochs: {max_epochs}, Patience: {patience}")
                        st.write("")
                        st.write("Starting training pipeline...")
                        st.caption("Note: Initial data loading may take several minutes for large datasets.")

                        result = backend.train_model(selected_paths, train_cfg)

                        status.update(label="Training completed!", state="complete", expanded=False)

                    st.session_state[_LAST_TRAIN] = result
                    st.session_state[_LAST_TRAIN_CONFIG] = train_cfg
                    st.session_state[_TRAIN_CONFIG_PENDING] = train_cfg

                    # Show training progress if available
                    if result.checkpoint_dir:
                        progress = backend.get_training_progress(result.checkpoint_dir)
                        if progress and progress.get("status") == "completed":
                            st.success(
                                f"Training completed! Best val score: {progress.get('best_val_score', 0.0):.4f}")

                    st.session_state[_TRAIN_FEEDBACK] = (
                        "success",
                        f"Model stored at {result.bundle_path.name} (hash {result.dataset_hash}).",
                    )
                    st.rerun()
                except RuntimeError as exc:
                    if "Deep-TICA optional dependencies missing" in str(exc):
                        st.warning(DEEPTICA_SKIP_MESSAGE)
                    else:
                        st.error(f"Training failed: {exc}")
                except Exception as exc:
                    st.error(f"Training failed: {exc}")

    # Check for ongoing training and show log viewer
    models_dir = layout.models_dir
    training_dirs = [d for d in models_dir.glob("training-*") if d.is_dir()]
    if training_dirs:
        latest_training = max(training_dirs, key=lambda d: d.name)
        log_file = latest_training / "training.log"
        progress_file = latest_training / "training_progress.json"

        # Check if this is an ongoing training (progress file doesn't exist or status is "training")
        is_training_ongoing = False
        if log_file.exists():
            if not progress_file.exists():
                is_training_ongoing = True
            else:
                try:
                    with progress_file.open("r") as f:
                        progress_data = json.load(f)
                        if progress_data.get("status") == "training":
                            is_training_ongoing = True
                except Exception:
                    pass

        if is_training_ongoing and log_file.exists():
            with st.expander("⚠️ Training in Progress - View Log", expanded=True):
                st.warning(f"Training directory: `{latest_training.name}`")
                st.caption(
                    "Training may take 10-30 minutes depending on data size. Check the log below for progress.")

                if st.button("Refresh Log", key="refresh_train_log_button"):
                    st.rerun()

                try:
                    with log_file.open("r") as f:
                        log_content = f.read()

                    # Show last 50 lines of log
                    log_lines = log_content.strip().split("\n")
                    if len(log_lines) > 50:
                        st.text_area(
                            "Recent Log Entries (last 50 lines)",
                            "\n".join(log_lines[-50:]),
                            height=300,
                            key="train_log_viewer"
                        )
                    else:
                        st.text_area(
                            "Training Log",
                            log_content,
                            height=300,
                            key="train_log_viewer_full"
                        )
                except Exception as e:
                    st.error(f"Could not read log file: {e}")

    last_train: TrainingResult | None = st.session_state.get(_LAST_TRAIN)
    if last_train is not None:
        # Show real-time training progress if available
        if last_train.checkpoint_dir:
            progress = backend.get_training_progress(last_train.checkpoint_dir)
            if progress:
                with st.expander("Training Progress", expanded=True):
                    status = progress.get("status", "unknown")
                    st.write(f"**Status**: {status}")

                    if status == "training":
                        current_epoch = progress.get("current_epoch", 0)
                        total_epochs = progress.get("total_epochs_planned", 0)
                        if total_epochs > 0:
                            st.progress(current_epoch / total_epochs)
                            st.write(f"Epoch {current_epoch} / {total_epochs}")

                    epochs_data = progress.get("epochs", [])
                    if epochs_data:
                        df = pd.DataFrame(epochs_data)

                        col1, col2 = st.columns(2)
                        with col1:
                            if "val_score" in df.columns:
                                st.line_chart(df[["epoch", "val_score"]].set_index("epoch"))
                                st.caption("Validation Score")
                        with col2:
                            if "train_loss" in df.columns:
                                st.line_chart(df[["epoch", "train_loss"]].set_index("epoch"))
                                st.caption("Training Loss")

                        # Show best epoch info
                        best_epoch = progress.get("best_epoch")
                        best_score = progress.get("best_val_score", 0.0)
                        if best_epoch:
                            st.info(f"Best validation score: {best_score:.4f} at epoch {best_epoch}")

        _show_build_outputs(last_train)
        summary = (
            last_train.build_result.artifacts.get("mlcv_deeptica")
            if last_train.build_result
            else None
        )
        if summary:
            _render_deeptica_summary(summary)

    models = backend.list_models()
    if models:
        with st.expander(
                "Load recorded model",
                expanded=st.session_state.get(_LAST_TRAIN) is None,
        ):
            indices = list(range(len(models)))

            def _model_label(idx: int) -> str:
                entry = models[idx]
                bundle_raw = entry.get("bundle", "")
                bundle_name = (
                    Path(bundle_raw).name if bundle_raw else f"model-{idx}"
                )
                created = entry.get("created_at", "unknown")
                return f"{bundle_name} (created {created})"

            selected_idx = st.selectbox(
                "Stored models",
                options=indices,
                format_func=_model_label,
                key="load_model_select",
            )
            if st.button("Show model", key="load_model_button"):
                loaded = backend.load_model(int(selected_idx))
                if loaded is not None:
                    st.session_state[_LAST_TRAIN] = loaded
                    cfg_loaded: TrainingConfig | None = None
                    try:
                        cfg_loaded = backend.training_config_from_entry(
                            models[int(selected_idx)]
                        )
                    except Exception:
                        cfg_loaded = None
                    if cfg_loaded is not None:
                        st.session_state[_LAST_TRAIN_CONFIG] = cfg_loaded
                        st.session_state[_TRAIN_CONFIG_PENDING] = cfg_loaded
                    st.session_state[_TRAIN_FEEDBACK] = (
                        "success",
                        f"Loaded model {loaded.bundle_path.name}.",
                    )
                    st.rerun()
                else:
                    st.error("Could not load the selected model from disk.")

    last_model_path = backend.latest_model_path()
    if last_model_path is not None:
        st.caption(f"Latest model bundle: {last_model_path}")
