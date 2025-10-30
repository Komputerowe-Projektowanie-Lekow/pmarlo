from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence
import json
import traceback
import logging
import os
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from pmarlo.data.shard_io import ShardRunSummary, summarize_shard_runs
try:  # Prefer package-relative imports when launched via `streamlit run -m`
    from pmarlo.transform.build import _sanitize_artifacts

    from .backend import (
        BuildArtifact,
        BuildConfig,
        ConformationsConfig,
        ConformationsResult,
        calculate_its,
        plot_its as backend_plot_its,
        ShardRequest,
        SimulationConfig,
        TrainingConfig,
        TrainingResult,
        WorkflowBackend,
        WorkspaceLayout,
    )
    from .plots import plot_fes, plot_msm
    from .plots.diagnostics import (
        create_fes_validation_plot,
        create_sampling_validation_plot,
        format_warnings,
        plot_autocorrelation_curves,
        plot_canonical_correlations,
    )
except ImportError:  # Fallback for `streamlit run app.py`
    import sys

    _APP_DIR = Path(__file__).resolve().parent
    if str(_APP_DIR) not in sys.path:
        sys.path.insert(0, str(_APP_DIR))
    from backend import (  # type: ignore
        BuildArtifact,
        BuildConfig,
        ConformationsConfig,
        ConformationsResult,
        calculate_its,
        plot_its as backend_plot_its,
        ShardRequest,
        SimulationConfig,
        TrainingConfig,
        TrainingResult,
        WorkflowBackend,
        WorkspaceLayout,
    )
    from plots import plot_fes, plot_msm  # type: ignore
    from plots.diagnostics import (  # type: ignore
        create_fes_validation_plot,
        create_sampling_validation_plot,
        format_warnings,
        plot_autocorrelation_curves,
        plot_canonical_correlations,
    )

    from pmarlo.transform.build import _sanitize_artifacts


# Banner shown when Deep-TICA training is gated off by missing extras
DEEPTICA_SKIP_MESSAGE = (
    "Deep-TICA CV learning was skipped because optional dependencies are not installed."
)

# Keys used inside st.session_state
_LAST_SIM = "__pmarlo_last_simulation"
_LAST_SHARDS = "__pmarlo_last_shards"
_LAST_TRAIN = "__pmarlo_last_training"
_LAST_TRAIN_CONFIG = "__pmarlo_last_train_cfg"
_LAST_BUILD = "__pmarlo_last_build"
_RUN_PENDING = "__pmarlo_run_pending"
_TRAIN_CONFIG_PENDING = "__pmarlo_pending_train_cfg"
_TRAIN_FEEDBACK = "__pmarlo_train_feedback"
_LAST_CONFORMATIONS = "__pmarlo_last_conformations"
_CONFORMATIONS_FEEDBACK = "__pmarlo_conf_feedback"
_LAST_ITS_RESULT = "__pmarlo_last_its"
_ITS_FEEDBACK = "__pmarlo_its_feedback"
_MODEL_PREVIEW_SELECTION = "__pmarlo_model_preview_select"
_MODEL_PREVIEW_RESULT = "__pmarlo_model_preview_result"
_ASSET_RUN_SELECTION = "__pmarlo_asset_run_select"
_ASSET_SHARD_SELECTION = "__pmarlo_asset_shard_select"
_ASSET_MODEL_SELECTION = "__pmarlo_asset_model_select"
_ASSET_BUILD_SELECTION = "__pmarlo_asset_build_select"
_ASSET_CONF_SELECTION = "__pmarlo_asset_conf_select"
_ITS_PENDING_TOPOLOGY = "__pmarlo_its_pending_topology"
_ITS_PENDING_FEATURE_SPEC = "__pmarlo_its_pending_feature_spec"


def _configure_file_logging() -> None:
    """Configure Python logging to write to a timestamped file.

    Creates a log file in example_programs/app_usecase/app_outputs/app_logs/
    with a timestamp in the filename. All logs from the app and pmarlo library
    will be captured in this file.

    This function is idempotent - if logging has already been configured
    (i.e., handlers are present), it will return immediately without
    reconfiguring.

    Raises:
        OSError: If the log directory cannot be created or the log file cannot be opened.
    """
    # Singleton check: Only configure logging once
    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        # Logging already configured, skip
        return

    # Define log directory path
    app_dir = Path(__file__).resolve().parent
    log_dir = app_dir.parent / "app_outputs" / "app_logs"

    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)

    # Create unique log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"app_log_{timestamp}.log"
    log_filepath = log_dir / log_filename

    # Configure logging with file handler

    # Set logging level to capture all messages
    root_logger.setLevel(logging.DEBUG)

    # Create file handler
    file_handler = logging.FileHandler(log_filepath, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)

    # Create formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)

    # Add handler to root logger (affects all loggers including pmarlo)
    root_logger.addHandler(file_handler)

    # Reduce verbosity for noisy third-party libraries
    # Set matplotlib logger to INFO to suppress DEBUG messages
    mpl_logger = logging.getLogger('matplotlib')
    mpl_logger.setLevel(logging.INFO)

    # Set PIL/Pillow logger to INFO to suppress DEBUG messages
    pil_logger = logging.getLogger('PIL')
    pil_logger.setLevel(logging.INFO)

    # Explicitly ensure pmarlo logger remains at DEBUG level
    pmarlo_logger = logging.getLogger('pmarlo')
    pmarlo_logger.setLevel(logging.DEBUG)

    # Log initialization message
    root_logger.info(f"Logging initialized. Log file: {log_filepath}")
    root_logger.info(f"App directory: {app_dir}")


def _sync_sidebar_tica_dim() -> None:
    """Keep sidebar TICA dimension in sync with form inputs."""

    value = int(st.session_state.get("conf_tica_dim", 10))
    st.session_state["conf_n_components"] = value


def _sync_form_tica_dim() -> None:
    """Propagate form-based TICA updates back to the sidebar widget."""

    value = int(st.session_state.get("conf_n_components", 10))
    st.session_state["conf_tica_dim"] = value


def _sync_sidebar_metastable_states() -> None:
    """Synchronize sidebar metastable count with form state."""

    value = int(st.session_state.get("conf_n_metastable_sidebar", 10))
    st.session_state["conf_n_metastable"] = value
    st.session_state["conf_n_metastable_form"] = value


def _sync_form_metastable_states() -> None:
    """Propagate form metastable updates back to the sidebar widget."""

    value = int(st.session_state.get("conf_n_metastable_form", 10))
    st.session_state["conf_n_metastable"] = value
    st.session_state["conf_n_metastable_sidebar"] = value


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


def _select_shard_paths(
    groups: Sequence[Dict[str, object]], run_ids: Sequence[str]
) -> List[Path]:
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


def _format_run_selection_summary(runs: Sequence[ShardRunSummary]) -> str:
    if not runs:
        return ""
    parts: List[str] = []
    for run in runs:
        count = int(run.shard_count)
        shard_label = "shard" if count == 1 else "shards"
        parts.append(
            f"{run.run_id}: {count} {shard_label} @ {run.temperature_K:.1f} K"
        )
    return "; ".join(parts)


def _summarize_selected_shards(
    shard_paths: Sequence[Path],
) -> tuple[List[ShardRunSummary], str]:
    if not shard_paths:
        return [], ""
    summaries = summarize_shard_runs(shard_paths)
    return summaries, _format_run_selection_summary(summaries)


def _default_feature_spec_path(layout: WorkspaceLayout) -> Optional[Path]:
    """Return the packaged feature specification if it exists."""

    candidate = (layout.app_root / "app" / "feature_spec.yaml").resolve()
    if candidate.exists():
        return candidate
    return None


def _infer_default_topology(
    backend: WorkflowBackend,
    layout: WorkspaceLayout,
    run_ids: Sequence[str],
) -> Optional[Path]:
    """Heuristically locate a topology file for the selected shard runs."""

    target_ids = {str(run_id) for run_id in run_ids if run_id}
    for entry in reversed(list(backend.state.runs)):
        run_id = str(entry.get("run_id"))
        if target_ids and run_id not in target_ids:
            continue
        candidate = entry.get("pdb") or entry.get("topology")
        if not candidate:
            continue
        path = Path(candidate).expanduser().resolve()
        if path.exists():
            return path

    inputs = layout.available_inputs()
    if inputs:
        return inputs[0]
    return None


def _format_lag_sequence(values: Sequence[int]) -> str:
    """Render lag time sequence as a comma-separated string."""

    if not values:
        return ""
    return ", ".join(str(int(v)) for v in values)


def _parse_lag_sequence(raw: str) -> List[int]:
    """Parse lag times from a comma- or semicolon-delimited string."""

    tokens = [token.strip() for token in raw.replace(";", ",").split(",")]
    lags: List[int] = []
    for token in tokens:
        if not token:
            continue
        try:
            value = int(float(token))
        except ValueError as exc:  # pragma: no cover - user input driven
            raise ValueError(f"Invalid lag time '{token}'") from exc
        if value < 1:
            raise ValueError(
                f"Lag times must be positive integers; received {value}"
            )
        lags.append(int(value))
    if not lags:
        raise ValueError("Provide at least one lag time for implied timescale analysis.")
    return lags


def _timescales_dataframe(
    lag_times: Sequence[int],
    timescales: Sequence[Sequence[float] | np.ndarray],
) -> pd.DataFrame:
    """Create a rectangular table of implied timescales across lags."""

    arrays = [np.asarray(ts, dtype=float).reshape(-1) for ts in timescales]
    max_len = max((arr.size for arr in arrays), default=0)
    columns = ["Lag (steps)"] + [f"Timescale {idx + 1}" for idx in range(max_len)]
    rows: List[Dict[str, Any]] = []
    for lag, arr in zip(lag_times, arrays):
        row: Dict[str, Any] = {"Lag (steps)": int(lag)}
        for idx in range(max_len):
            if idx < arr.size and np.isfinite(arr[idx]):
                row[f"Timescale {idx + 1}"] = float(arr[idx])
            else:
                row[f"Timescale {idx + 1}"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def _runs_dataframe(entries: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for idx, entry in enumerate(entries):
        traj_files = entry.get("traj_files") or []
        rows.append(
            {
                "Index": idx,
                "Run ID": str(entry.get("run_id", "")),
                "Created": entry.get("created_at", ""),
                "Steps": entry.get("steps"),
                "Trajectories": len(traj_files) if isinstance(traj_files, Sequence) else None,
                "Quick": bool(entry.get("quick", False)),
                "Directory": entry.get("run_dir", ""),
            }
        )
    return pd.DataFrame(rows)


def _shards_dataframe(entries: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for idx, entry in enumerate(entries):
        paths = entry.get("paths") or []
        rows.append(
            {
                "Index": idx,
                "Run ID": str(entry.get("run_id", "")),
                "Created": entry.get("created_at", ""),
                "Shards": len(paths) if isinstance(paths, Sequence) else None,
                "Frames": entry.get("n_frames"),
                "Stride": entry.get("stride"),
                "Hop": entry.get("hop_frames"),
                "Temperature (K)": entry.get("temperature"),
                "Directory": entry.get("directory", ""),
            }
        )
    return pd.DataFrame(rows)


def _models_dataframe(entries: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for idx, entry in enumerate(entries):
        tau_schedule = entry.get("tau_schedule") or []
        hidden = entry.get("hidden") or []
        rows.append(
            {
                "Index": idx,
                "Created": entry.get("created_at", ""),
                "Lag": entry.get("lag"),
                "Seed": entry.get("seed"),
                "Tau schedule": _format_tau_schedule(tau_schedule) if tau_schedule else "",
                "Hidden layers": ", ".join(str(h) for h in hidden) if hidden else "",
                "Dataset hash": entry.get("dataset_hash", ""),
                "Bundle": entry.get("bundle", ""),
            }
        )
    return pd.DataFrame(rows)


def _builds_dataframe(entries: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for idx, entry in enumerate(entries):
        rows.append(
            {
                "Index": idx,
                "Created": entry.get("created_at", ""),
                "Lag": entry.get("lag"),
                "Microstates": entry.get("n_microstates"),
                "Reweight": entry.get("reweight_mode"),
                "Bundle": entry.get("bundle", ""),
                "Debug dir": entry.get("debug_dir", ""),
            }
        )
    return pd.DataFrame(rows)


def _conformations_dataframe(entries: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for idx, entry in enumerate(entries):
        rows.append(
            {
                "Index": idx,
                "Created": entry.get("created_at", ""),
                "Output": entry.get("output_dir", ""),
                "Converged": entry.get("tpt_converged"),
                "Error": entry.get("error", ""),
            }
        )
    return pd.DataFrame(rows)


def _model_entry_label(entry: Mapping[str, Any], idx: int) -> str:
    bundle_raw = entry.get("bundle", "")
    bundle_name = Path(bundle_raw).name if bundle_raw else f"model-{idx}"
    created = entry.get("created_at", "unknown")
    return f"{bundle_name} (created {created})"


def _metrics_table(flags: Dict[str, object]) -> pd.DataFrame:
    """Render build flags in a tabular, Arrow-friendly representation.

    Streamlit converts the returned DataFrame into an Arrow table. Mixed dtypes
    within a column (for example booleans alongside strings) lead Arrow to infer
    an incompatible schema which subsequently raises an ``ArrowInvalid``. The
    build flags frequently contain boolean toggles together with nested
    structures such as diagnostic warning lists, so we normalise them into a
    flat table that stores display strings alongside their original type.
    """

    from collections.abc import Mapping
    from collections.abc import Sequence as _SequenceABC

    try:  # Local import to avoid an unconditional dependency at module import
        import numpy as np  # type: ignore
    except Exception:  # pragma: no cover - NumPy is always available in practice
        np = None  # type: ignore

    def _coerce_scalar(val: object) -> object:
        if np is not None and isinstance(val, np.generic):
            return val.item()
        return val

    def _is_sequence(val: object) -> bool:
        return isinstance(val, _SequenceABC) and not isinstance(
            val, (str, bytes, bytearray)
        )

    def _iter_items(prefix: str, value: object):
        value = _coerce_scalar(value)

        if isinstance(value, Mapping):
            for sub_key, sub_val in value.items():
                next_prefix = f"{prefix}.{sub_key}" if prefix else str(sub_key)
                yield from _iter_items(next_prefix, sub_val)
            return

        if (
            np is not None
            and hasattr(value, "tolist")
            and not isinstance(value, (str, bytes, bytearray))
        ):
            try:
                value = value.tolist()
            except Exception:
                pass

        if _is_sequence(value):
            seq = list(value)
            if not seq:
                yield (prefix, "[]")
                return
            for idx, item in enumerate(seq, start=1):
                suffix = f"[{idx}]"
                next_prefix = f"{prefix}{suffix}" if prefix else suffix
                yield from _iter_items(next_prefix, item)
            return

        yield (prefix, value)

    def _format_value(val: object) -> tuple[str, str]:
        val = _coerce_scalar(val)
        if val is None:
            return "", "NoneType"
        if isinstance(val, bool):
            return ("True" if val else "False"), "bool"
        if isinstance(val, (int, float)):
            return f"{val}", type(val).__name__
        if isinstance(val, bytes):
            try:
                decoded = val.decode("utf-8")
            except Exception:
                decoded = val.decode("utf-8", errors="replace")
            return decoded, "bytes"
        return str(val), type(val).__name__

    rows: List[Dict[str, object]] = []
    for key, raw_value in flags.items():
        for metric_key, metric_value in _iter_items(key, raw_value):
            display, dtype_name = _format_value(metric_value)
            rows.append(
                {
                    "metric": metric_key,
                    "value": display,
                    "value_type": dtype_name,
                }
            )

    if not rows:
        return pd.DataFrame({"metric": [], "value": [], "value_type": []})

    df = pd.DataFrame(rows)
    df["value"] = pd.Series(df["value"], dtype="string")
    df["value_type"] = pd.Series(df["value_type"], dtype="string")
    return df


def _format_tau_schedule(values: Sequence[int]) -> str:
    if not values:
        return ""
    return ", ".join(str(int(v)) for v in values)


def _parse_tau_schedule(raw: str) -> List[int]:
    cleaned = raw.replace(";", ",")
    values: List[int] = []
    for token in cleaned.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            val = int(token)
        except ValueError as exc:  # pragma: no cover - defensive parsing
            raise ValueError(f"Invalid tau value '{token}'") from exc
        if val <= 0:
            raise ValueError("Tau values must be positive integers.")
        values.append(val)
    if not values:
        raise ValueError("Provide at least one tau value.")
    return sorted(set(values))


def _update_state(**kwargs: Any) -> None:
    for key, value in kwargs.items():
        st.session_state[key] = value


def _apply_training_config_to_state(cfg: TrainingConfig) -> None:
    bins = dict(cfg.bins or {})
    hidden_str = ", ".join(str(int(h)) for h in cfg.hidden)
    _update_state(
        train_lag=int(cfg.lag),
        train_bins_rg=int(bins.get("Rg", 64)),
        train_bins_rmsd=int(bins.get("RMSD_ref", 64)),
        train_seed=int(cfg.seed),
        train_max_epochs=int(cfg.max_epochs),
        train_patience=int(cfg.early_stopping),
        train_temperature=float(cfg.temperature),
        train_hidden_layers=hidden_str or "128,128",
        train_tau_schedule=_format_tau_schedule(cfg.tau_schedule),
        train_val_tau=int(cfg.val_tau),
        train_epochs_per_tau=int(cfg.epochs_per_tau),
    )


def _apply_analysis_config_to_state(cfg: BuildConfig) -> None:
    bins = dict(cfg.bins or {})
    if isinstance(cfg.fes_bandwidth, (int, float)):
        bw_value = f"{float(cfg.fes_bandwidth):g}"
    else:
        bw_value = str(cfg.fes_bandwidth)
    mode = str(cfg.reweight_mode)
    mode_norm = mode.upper() if mode.upper() in {"MBAR", "TRAM"} else mode.lower()
    _update_state(
        analysis_lag=int(cfg.lag),
        analysis_bins_rg=int(bins.get("Rg", 72)),
        analysis_bins_rmsd=int(bins.get("RMSD_ref", 72)),
        analysis_seed=int(cfg.seed),
        analysis_temperature=float(cfg.temperature),
        analysis_learn_cv=bool(cfg.learn_cv),
        analysis_apply_whitening=bool(cfg.apply_cv_whitening),
        analysis_cluster_mode=str(cfg.cluster_mode),
        analysis_n_microstates=int(cfg.n_microstates),
        analysis_reweight_mode=mode_norm,
        analysis_fes_method=str(cfg.fes_method),
        analysis_fes_bandwidth=bw_value,
        analysis_min_count_per_bin=int(cfg.fes_min_count_per_bin),
    )


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

    flags: Dict[str, Any] = br.flags or {}

    def _safe_int(value: Any) -> Optional[int]:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    tau_frames = _safe_int(getattr(artifact, "tau_frames", None))
    if tau_frames is None:
        tau_frames = _safe_int(flags.get("analysis_tau_frames"))
    effective_tau = _safe_int(getattr(artifact, "effective_tau_frames", None))
    if effective_tau is None:
        effective_tau = _safe_int(flags.get("analysis_effective_tau_frames"))
    stride_max = _safe_int(getattr(artifact, "effective_stride_max", None))
    if stride_max is None:
        stride_max = _safe_int(flags.get("analysis_effective_stride_max"))
    fingerprint = getattr(artifact, "discretizer_fingerprint", None) or flags.get(
        "discretizer_fingerprint"
    )
    fingerprint_changed = bool(flags.get("discretizer_fingerprint_changed"))
    tau_mismatch = flags.get("analysis_tau_mismatch")
    if not isinstance(tau_mismatch, Mapping):
        tau_mismatch = None
    preview_truncated = flags.get("analysis_preview_truncated") or []
    stride_map = flags.get("analysis_effective_stride_map") or {}
    guardrail_info = (
        flags.get("analysis_guardrail_violations")
        or getattr(artifact, "guardrail_violations", None)
        or []
    )
    analysis_healthy_flag = flags.get("analysis_healthy")
    if analysis_healthy_flag is None:
        analysis_healthy_flag = getattr(artifact, "analysis_healthy", None)

    meta_cols = st.columns(4)
    meta_cols[0].metric("Shards", int(br.n_shards))
    meta_cols[1].metric("Frames", int(br.n_frames))
    meta_cols[2].metric("Features", len(br.feature_names))
    if tau_frames is not None:
        delta_label = None
        if tau_mismatch is not None:
            delta_label = f"req {tau_mismatch.get('requested')}"
        meta_cols[3].metric("tau (frames)", tau_frames, delta=delta_label)
    else:
        meta_cols[3].metric("tau (frames)", "n/a")

    if flags:
        st.dataframe(_metrics_table(br.flags), width="stretch")

    if tau_mismatch is not None:
        st.warning(
            f"Requested tau={tau_mismatch.get('requested')} frames but build used {tau_mismatch.get('actual')}.",
        )

    if fingerprint_changed:
        st.info("Discretizer fingerprint differs from requested configuration; cached mappings were invalidated.")

    if stride_max is not None and stride_max > 1:
        info_msg = (
            f"Effective tau={effective_tau}" if effective_tau is not None else f"Max stride={stride_max}"
        )
        st.info(f"Detected subsampling in shards (max stride={stride_max}). {info_msg}.")

    if analysis_healthy_flag is not None:
        status_text = "healthy" if analysis_healthy_flag else "violations detected"
        st.caption(f"Analysis health: {status_text}")
    if guardrail_info:
        st.warning("Guardrail violations detected:")
        st.json(guardrail_info)

    if preview_truncated:
        formatted = ", ".join(str(item) for item in preview_truncated)
        st.warning(
            "Some shards appear truncated relative to their metadata: "
            f"{formatted}"
        )

    debug_summary = getattr(artifact, "debug_summary", None)
    if debug_summary:
        with st.expander("Analysis metadata", expanded=False):
            total_pairs = debug_summary.get("total_pairs", "n/a")
            zero_rows = debug_summary.get("zero_rows", "n/a")
            st.write(
                f"Total (t, t+tau) pairs: {total_pairs} | Zero rows: {zero_rows}"
            )
            stride_values = debug_summary.get("effective_strides") or []
            if stride_values:
                st.write(f"Effective strides detected: {stride_values}")
            effective_tau_summary = debug_summary.get("effective_tau_frames")
            if effective_tau_summary is not None:
                st.write(f"Effective tau (stride-adjusted): {effective_tau_summary}")
            if stride_map:
                st.write("Per-shard stride map:")
                st.json(stride_map)
            first_ts = debug_summary.get("first_timestamps") or []
            last_ts = debug_summary.get("last_timestamps") or []
            if first_ts or last_ts:
                first_repr = ", ".join(str(val) for val in first_ts) if first_ts else "n/a"
                last_repr = ", ".join(str(val) for val in last_ts) if last_ts else "n/a"
                st.write(f"First timestamps: {first_repr} | Last timestamps: {last_repr}")
            if fingerprint:
                st.write("Discretizer fingerprint:")
                st.json(fingerprint)
            else:
                st.write("Discretizer fingerprint: unavailable")
    elif fingerprint:
        st.caption("Discretizer fingerprint")
        st.json(fingerprint)

    if br.messages:
        st.write("Messages:")
        for msg in br.messages:
            st.write(f"- {msg}")
def _render_deeptica_summary(summary: Dict[str, object]) -> None:
    cleaned = _sanitize_artifacts(summary)
    st.caption("Deep-TICA summary")
    if not isinstance(cleaned, dict):
        st.json(cleaned)
        return
    st.json(cleaned)
    with st.expander("Training diagnostics", expanded=False):
        epochs = list(range(1, len(cleaned.get("val_score_curve", [])) + 1))
        if epochs:
            df_score = pd.DataFrame(
                {"val_score": cleaned.get("val_score_curve", [])}, index=epochs
            )
            st.line_chart(df_score, height=200)
        var_z0 = cleaned.get("var_z0_curve") or []
        if var_z0:
            df_var_z0 = pd.DataFrame(var_z0)
            df_var_z0.index = list(range(1, len(df_var_z0) + 1))
            df_var_z0.columns = [f"z0_{i+1}" for i in range(df_var_z0.shape[1])]
            st.line_chart(df_var_z0, height=200)
        var_zt = cleaned.get("var_zt_curve") or []
        if var_zt:
            df_var_zt = pd.DataFrame(var_zt)
            df_var_zt.index = list(range(1, len(df_var_zt) + 1))
            df_var_zt.columns = [f"zt_{i+1}" for i in range(df_var_zt.shape[1])]
            st.line_chart(df_var_zt, height=200)
        cond_data: Dict[str, object] = {}
        if cleaned.get("cond_c00_curve"):
            cond_data["cond_C00"] = cleaned.get("cond_c00_curve")
        if cleaned.get("cond_ctt_curve"):
            cond_data["cond_Ctt"] = cleaned.get("cond_ctt_curve")
        if cond_data:
            df_cond = pd.DataFrame(cond_data)
            st.line_chart(df_cond, height=200)
        if cleaned.get("grad_norm_curve"):
            df_grad = pd.DataFrame({"grad_norm": cleaned.get("grad_norm_curve")})
            st.line_chart(df_grad, height=200)


def _render_conformations_result(conf_result: ConformationsResult) -> None:
    """Visualise stored or freshly computed conformations analysis results."""

    if conf_result.error:
        st.error(f"Conformations analysis failed: {conf_result.error}")
        st.info(f"Output directory: {conf_result.output_dir}")
        return

    if not conf_result.tpt_converged:
        iteration_count = (
            conf_result.tpt_pathway_iterations
            or conf_result.tpt_summary.get("pathway_iterations")
            or conf_result.tpt_pathway_max_iterations
            or conf_result.tpt_summary.get("pathway_max_iterations")
        )
        if iteration_count is not None:
            st.error(
                "Warning: TPT calculation failed to converge after "
                f"{iteration_count} iterations. Pathway results are unreliable. "
                "Try increasing the number of clusters or adjusting the MSM lag time."
            )
        else:
            st.error(
                "Warning: TPT calculation failed to converge. Pathway results are unreliable. "
                "Try increasing the number of clusters or adjusting the MSM lag time."
            )

    if conf_result.tpt_summary:
        st.subheader("TPT Results")
        cols = st.columns(4)
        cols[0].metric(
            "Rate", f"{conf_result.tpt_summary.get('rate', float('nan')):.3e}"
        )
        cols[1].metric(
            "MFPT", f"{conf_result.tpt_summary.get('mfpt', float('nan')):.1f}"
        )
        cols[2].metric(
            "Total Flux",
            f"{conf_result.tpt_summary.get('total_flux', float('nan')):.3e}",
        )
        cols[3].metric(
            "N Pathways", conf_result.tpt_summary.get("n_pathways", "n/a")
        )

    if conf_result.metastable_states:
        st.subheader("Metastable States")
        meta_df_data = []
        for state_id, state_data in conf_result.metastable_states.items():
            meta_df_data.append(
                {
                    "State": state_id,
                    "Population": f"{float(state_data.get('population', 0.0)):.4f}",
                    "N States": state_data.get("n_states", 0),
                    "PDB": (
                        Path(state_data.get("representative_pdb", "")).name
                        if state_data.get("representative_pdb")
                        else "N/A"
                    ),
                }
            )
        st.dataframe(pd.DataFrame(meta_df_data), width="stretch")

    if conf_result.transition_states:
        st.subheader("Transition States")
        ts_df_data = []
        for ts_data in conf_result.transition_states:
            ts_df_data.append(
                {
                    "State Index": ts_data.get("state_index", ""),
                    "Committor": f"{float(ts_data.get('committor', 0.0)):.3f}",
                    "PDB": (
                        Path(ts_data.get("representative_pdb", "")).name
                        if ts_data.get("representative_pdb")
                        else "N/A"
                    ),
                }
            )
        st.dataframe(pd.DataFrame(ts_df_data), width="stretch")

    if conf_result.pathways:
        st.subheader("Dominant Pathways")
        st.write(conf_result.pathways)

    if conf_result.plots:
        st.subheader("Visualizations")
        plot_cols = st.columns(2)
        plot_idx = 0
        for plot_name, plot_path in conf_result.plots.items():
            if Path(plot_path).exists():
                with plot_cols[plot_idx % 2]:
                    st.image(
                        str(plot_path),
                        caption=plot_name.replace("_", " ").title(),
                    )
                plot_idx += 1

    st.info(f"All conformations saved to: {conf_result.output_dir}")
    if conf_result.representative_pdbs:
        st.write(
            f"{len(conf_result.representative_pdbs)} representative PDB files saved"
        )

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
    st.session_state.setdefault(_TRAIN_CONFIG_PENDING, None)
    st.session_state.setdefault(_TRAIN_FEEDBACK, None)
    st.session_state.setdefault("train_hidden_layers", "128,128")
    st.session_state.setdefault("train_tau_schedule", "2,5,10,20")
    st.session_state.setdefault("train_val_tau", 20)
    st.session_state.setdefault("train_epochs_per_tau", 15)
    st.session_state.setdefault("analysis_cluster_mode", "kmeans")
    st.session_state.setdefault("analysis_n_microstates", 20)
    st.session_state.setdefault("analysis_reweight_mode", "MBAR")
    st.session_state.setdefault("analysis_fes_method", "kde")
    st.session_state.setdefault("analysis_fes_bandwidth", "scott")
    st.session_state.setdefault("analysis_min_count_per_bin", 1)
    st.session_state.setdefault("conf_n_clusters", 100)
    st.session_state.setdefault("its_n_clusters", 200)
    st.session_state.setdefault("its_tica_dim", 10)
    st.session_state.setdefault("its_lag_times", [1, 5, 10, 50, 100, 200])
    st.session_state.setdefault("its_tica_lag", 10)
    st.session_state.setdefault(
        "its_lag_times_text",
        _format_lag_sequence(st.session_state["its_lag_times"]),
    )
    st.session_state.setdefault("its_topology_path", "")
    st.session_state.setdefault("its_feature_spec_path", "")
    st.session_state.setdefault("conf_tica_dim", 10)
    st.session_state.setdefault("conf_n_components", st.session_state["conf_tica_dim"])
    st.session_state.setdefault("conf_n_metastable", 10)
    st.session_state.setdefault(
        "conf_n_metastable_sidebar", st.session_state["conf_n_metastable"]
    )
    st.session_state.setdefault(
        "conf_n_metastable_form", st.session_state["conf_n_metastable"]
    )
    st.session_state.setdefault("conf_committor_thresholds", (0.1, 0.9))
    st.session_state.setdefault(_LAST_CONFORMATIONS, None)
    st.session_state.setdefault(_CONFORMATIONS_FEEDBACK, None)
    st.session_state.setdefault(_LAST_ITS_RESULT, None)
    st.session_state.setdefault(_ITS_FEEDBACK, None)
    st.session_state.setdefault(_MODEL_PREVIEW_SELECTION, None)
    st.session_state.setdefault(_MODEL_PREVIEW_RESULT, None)
    st.session_state.setdefault(_ASSET_RUN_SELECTION, None)
    st.session_state.setdefault(_ASSET_SHARD_SELECTION, None)
    st.session_state.setdefault(_ASSET_MODEL_SELECTION, None)
    st.session_state.setdefault(_ASSET_BUILD_SELECTION, None)
    st.session_state.setdefault(_ASSET_CONF_SELECTION, None)
    st.session_state.setdefault(_ITS_PENDING_TOPOLOGY, None)
    st.session_state.setdefault(_ITS_PENDING_FEATURE_SPEC, None)


def _consume_pending_training_config() -> None:
    pending = st.session_state.get(_TRAIN_CONFIG_PENDING)
    if isinstance(pending, TrainingConfig):
        _apply_training_config_to_state(pending)
    st.session_state[_TRAIN_CONFIG_PENDING] = None



def main() -> None:
    # Configure file logging FIRST, before any other operations
    _configure_file_logging()

    st.set_page_config(page_title="PMARLO Joint Learning", layout="wide")
    _ensure_session_defaults()
    _consume_pending_training_config()

    layout = WorkspaceLayout.from_app_package()
    backend = WorkflowBackend(layout)

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

    tab_conformation, tab_its = st.tabs(
        [
            "Conformation Analysis",
            "Implied Timescales",
        ]
    )

    with tab_conformation:
        (
            tab_sampling,
            tab_training,
            tab_msm_fes,
            tab_conformations,
            tab_validation,
            tab_model_preview,
            tab_assets,
        ) = st.tabs(
            [
                "Sampling",
                "Model Training",
                "MSM/FES Analysis",
                "Conformation Analysis",
                "Free Energy Validation",
                "Model Preview",
                "Assets",
            ]
        )

        with tab_sampling:
            st.header("Sampling & Shard Production")
            inputs = layout.available_inputs()
            if not inputs:
                st.warning("No prepared proteins found. Place a PDB under app_intputs/.")
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
                    temps_raw = st.text_input(
                        "Temperature ladder (K)",
                        "300, 320, 340",
                        key="sim_temperature_ladder",
                        help="Comma-separated temperatures for replica exchange MD"
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
                            "copied into app_intputs/ so it becomes available as a protein input."
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

                # CV-Informed Sampling
                with st.expander("CV-Informed Sampling (Optional)", expanded=False):
                    st.write("Use a trained Deep-TICA model to bias the simulation and explore diverse conformations.")
                    models = backend.list_models()
                    cv_model_path = None
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
                            # Get checkpoint_dir where exported .pt files are, not the .pbz bundle
                            model_entry = models[selected_cv_idx]
                            checkpoint_dir_str = model_entry.get("checkpoint_dir")
                            cv_model_path = Path(checkpoint_dir_str) if checkpoint_dir_str else None

                            if cv_model_path and cv_model_path.exists():
                                st.success(f"Selected CV model from: {cv_model_path.name}")
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
                            elif cv_model_path:
                                st.error(f"Model checkpoint directory not found: {cv_model_path}")
                    else:
                        st.info("No trained CV models available. Train a model in the 'Model Training' tab to enable CV-informed sampling.")

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

                # CRITICAL: Only trigger simulation on button click, not on every rerun
                if st.button(
                    "Run replica exchange",
                    type="primary",
                    disabled=run_in_progress,
                    key="sim_run_button",
                ):
                    # Button was just clicked - prepare config and run IMMEDIATELY
                    try:
                        temps = _parse_temperature_ladder(temps_raw)
                        seed_val = int(random_seed_str) if random_seed_str.strip() else None
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
                            # DISABLED: CV biasing is not production-ready (causes 10-20x slowdown on CPU)
                            # cv_model_bundle=cv_model_path if cv_model_path and cv_model_path.exists() else None,
                            cv_model_bundle=None,
                        )

                        # Mark as in progress BEFORE running to prevent double-clicks
                        st.session_state[_RUN_PENDING] = True

                        with st.spinner("Running replica exchange..."):
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
                    emit = st.form_submit_button("Emit shard files")
                    if emit:
                        try:
                            request = ShardRequest(
                                stride=int(stride),
                                temperature=float(shard_temp),
                                seed_start=int(seed_start),
                                frames_per_shard=int(frames_per_shard),
                                hop_frames=int(hop_frames) if hop_frames > 0 else None,
                                reference=(
                                    Path(reference_path).expanduser().resolve()
                                    if reference_path.strip()
                                    else None
                                ),
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
                                    st.success(f"Training completed! Best val score: {progress.get('best_val_score', 0.0):.4f}")

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
                        st.caption("Training may take 10-30 minutes depending on data size. Check the log below for progress.")

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

        with tab_msm_fes:
            st.header("Build MSM and FES")
            shard_groups = backend.shard_summaries()

            builds = backend.list_builds()
            if builds:
                with st.expander(
                    "Load recorded analysis bundle",
                    expanded=st.session_state.get(_LAST_BUILD) is None,
                ):
                    indices = list(range(len(builds)))

                    def _build_label(idx: int) -> str:
                        entry = builds[idx]
                        bundle_raw = entry.get("bundle", "")
                        bundle_name = (
                            Path(bundle_raw).name if bundle_raw else f"bundle-{idx}"
                        )
                        created = entry.get("created_at", "unknown")
                        return f"{bundle_name} (created {created})"

                    selected_idx = st.selectbox(
                        "Stored analysis bundles",
                        options=indices,
                        format_func=_build_label,
                        key="load_build_select",
                    )
                    if st.button("Show bundle", key="load_build_button"):
                        loaded = backend.load_analysis_bundle(int(selected_idx))
                        if loaded is not None:
                            st.session_state[_LAST_BUILD] = loaded
                            try:
                                cfg_loaded = backend.build_config_from_entry(
                                    builds[int(selected_idx)]
                                )
                                _apply_analysis_config_to_state(cfg_loaded)
                            except Exception:
                                pass
                            st.success(f"Loaded bundle {loaded.bundle_path.name}.")
                            _show_build_outputs(loaded)
                            summary = (
                                loaded.build_result.artifacts.get("mlcv_deeptica")
                                if loaded.build_result
                                else None
                            )
                            if summary:
                                _render_deeptica_summary(summary)
                        else:
                            st.error(
                                "Could not load the selected analysis bundle from disk."
                            )

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
                try:
                    _analysis_runs, analysis_text = _summarize_selected_shards(
                        selected_paths
                    )
                except ValueError as exc:
                    st.error(f"Shard selection invalid: {exc}")
                    st.stop()
                st.write(f"Using {len(selected_paths)} shard files for analysis.")
                if analysis_text:
                    st.caption(analysis_text)

                # General Settings
                with st.expander(" General Settings", expanded=True):
                    col_seed, col_temp = st.columns(2)
                    seed = col_seed.number_input(
                        "Build seed",
                        min_value=0,
                        value=2025,
                        step=1,
                        key="analysis_seed",
                        help="Random seed for reproducibility"
                    )
                    temperature = col_temp.number_input(
                        "Reference temperature (K)",
                        min_value=0.0,
                        value=300.0,
                        step=5.0,
                        key="analysis_temperature",
                        help="Temperature for free energy calculations"
                    )

                # MSM Configuration
                with st.expander(" MSM Configuration", expanded=True):
                    lag = st.number_input(
                        "Lag time (steps)",
                        min_value=1,
                        value=10,
                        step=1,
                        key="analysis_lag",
                        help="Time delay for building the Markov State Model"
                    )
                    col_cluster, col_micro = st.columns(2)
                    cluster_mode = col_cluster.selectbox(
                        "Discretization mode",
                        options=["kmeans", "grid"],
                        index=(
                            0
                            if str(
                                st.session_state.get("analysis_cluster_mode", "kmeans")
                            ).lower()
                            != "grid"
                            else 1
                        ),
                        key="analysis_cluster_mode",
                        help="Method for partitioning CV space into microstates"
                    )
                    n_microstates = col_micro.number_input(
                        "Number of microstates",
                        min_value=2,
                        value=int(st.session_state.get("analysis_n_microstates", 20)),
                        step=1,
                        key="analysis_n_microstates",
                        help="Number of discrete states for MSM construction"
                    )
                    reweight_default = str(
                        st.session_state.get("analysis_reweight_mode", "TRAM")
                    )
                    reweight_index = 0
                    if reweight_default.upper() == "MBAR":
                        reweight_index = 1
                    reweight_mode = st.selectbox(
                        "Reweighting mode",
                        options=["TRAM", "MBAR"],
                        index=reweight_index,
                        key="analysis_reweight_mode",
                        help="Statistical reweighting method for enhanced sampling"
                    )

                # Collective Variable Settings
                with st.expander(" Collective Variable (CV) Settings", expanded=True):
                    col_rg, col_rmsd = st.columns(2)
                    bins_rg = col_rg.number_input(
                        "Bins for Rg",
                        min_value=8,
                        value=72,
                        step=4,
                        key="analysis_bins_rg",
                        help="Number of bins for radius of gyration"
                    )
                    bins_rmsd = col_rmsd.number_input(
                        "Bins for RMSD",
                        min_value=8,
                        value=72,
                        step=4,
                        key="analysis_bins_rmsd",
                        help="Number of bins for RMSD from reference"
                    )
                    apply_whitening = st.checkbox(
                        "Apply CV whitening",
                        value=True,
                        key="analysis_apply_whitening",
                        help="Standardize CVs to have zero mean and unit variance"
                    )
                    learn_cv = st.checkbox(
                        "Re-learn Deep-TICA during build",
                        value=False,
                        key="analysis_learn_cv",
                        help="Train a new Deep-TICA model for dimensionality reduction"
                    )
                    deeptica_params = None
                    if learn_cv:
                        st.markdown("**Deep-TICA Parameters**")
                        reuse = st.checkbox(
                            "Reuse last training hyperparameters",
                            value=st.session_state.get(_LAST_TRAIN_CONFIG) is not None,
                            key="analysis_reuse_train_cfg",
                        )
                        if reuse and st.session_state.get(_LAST_TRAIN_CONFIG) is not None:
                            cfg: TrainingConfig = st.session_state[_LAST_TRAIN_CONFIG]
                            deeptica_params = cfg.deeptica_params()
                        else:
                            lag_ml = st.number_input(
                                "Deep-TICA lag",
                                min_value=1,
                                value=int(lag),
                                key="analysis_lag_ml",
                            )
                            hidden_ml = st.text_input(
                                "Deep-TICA hidden layers",
                                value="128,128",
                                key="analysis_hidden_layers",
                            )
                            hidden_layers = tuple(
                                int(v.strip()) for v in hidden_ml.split(",") if v.strip()
                            ) or (128, 128)
                            max_epochs = st.number_input(
                                "Deep-TICA max epochs",
                                min_value=20,
                                value=200,
                                step=10,
                                key="analysis_max_epochs",
                            )
                            patience = st.number_input(
                                "Deep-TICA patience",
                                min_value=5,
                                value=25,
                                step=5,
                                key="analysis_patience",
                            )
                            deeptica_params = {
                                "lag": int(lag_ml),
                                "n_out": 2,
                                "hidden": hidden_layers,
                                "max_epochs": int(max_epochs),
                                "early_stopping": int(patience),
                                "reweight_mode": "scaled_time",
                            }

                # FES Configuration
                with st.expander(" Free Energy Surface (FES) Configuration", expanded=True):
                    col_fes_method, col_bw, col_min = st.columns(3)
                    fes_method = col_fes_method.selectbox(
                        "FES method",
                        options=["kde", "grid"],
                        index=(
                            0
                            if str(st.session_state.get("analysis_fes_method", "kde")).lower()
                            != "grid"
                            else 1
                        ),
                        key="analysis_fes_method",
                        help="Method for computing the free energy surface: KDE (kernel density) or grid-based"
                    )
                    fes_bandwidth = col_bw.text_input(
                        "FES bandwidth",
                        value=str(st.session_state.get("analysis_fes_bandwidth", "scott")),
                        help="Use 'scott', 'silverman', or a positive float (only for KDE).",
                        key="analysis_fes_bandwidth",
                    )
                    min_count_per_bin = col_min.number_input(
                        "Min count per bin",
                        min_value=0,
                        value=int(st.session_state.get("analysis_min_count_per_bin", 1)),
                        step=1,
                        key="analysis_min_count_per_bin",
                        help="Minimum number of samples required per bin for FES computation"
                    )

                disabled = len(selected_paths) == 0
                if st.button(
                    "Build MSM/FES bundle",
                    type="primary",
                    disabled=disabled,
                    key="analysis_build_button",
                ):
                    print("--- DEBUG: Build MSM/FES button clicked! ---")
                    try:
                        bw_clean = fes_bandwidth.strip()
                        try:
                            bandwidth_val: str | float = (
                                float(bw_clean) if bw_clean else "scott"
                            )
                        except ValueError:
                            bandwidth_val = bw_clean or "scott"
                        reweight_norm = str(reweight_mode)
                        if reweight_norm.upper() in {"MBAR", "TRAM"}:
                            reweight_final = reweight_norm.upper()
                        else:
                            reweight_final = "none"
                        build_cfg = BuildConfig(
                            lag=int(lag),
                            bins={"Rg": int(bins_rg), "RMSD_ref": int(bins_rmsd)},
                            seed=int(seed),
                            temperature=float(temperature),
                            learn_cv=bool(learn_cv),
                            deeptica_params=deeptica_params,
                            notes={"source": "app_usecase"},
                            apply_cv_whitening=bool(apply_whitening),
                            cluster_mode=str(cluster_mode),
                            n_microstates=int(n_microstates),
                            reweight_mode=reweight_final,
                            fes_method=str(fes_method),
                            fes_bandwidth=bandwidth_val,
                            fes_min_count_per_bin=int(min_count_per_bin),
                        )
                        artifact: BuildArtifact | None = None
                        try:
                            artifact = backend.build_analysis(selected_paths, build_cfg)
                        except ValueError as err:
                            error_message = str(err)
                            if "No transition pairs found" in error_message:
                                st.warning(
                                    "Analysis halted: "
                                    "no transition pairs were detected for the current lag. "
                                    f"{error_message}"
                                )
                            else:
                                raise
                        if artifact is not None:
                            st.session_state[_LAST_BUILD] = artifact
                            st.success(
                                f"Bundle {artifact.bundle_path.name} written (hash {artifact.dataset_hash})."
                            )
                            _show_build_outputs(artifact)
                            summary = (
                                artifact.build_result.artifacts.get("mlcv_deeptica")
                                if artifact.build_result
                                else None
                            )
                            if summary:
                                _render_deeptica_summary(summary)
                    except Exception as exc:
                        print(f"--- DEBUG: Analysis failed with exception: {exc}")
                        traceback.print_exc()
                        st.error(f"Analysis failed: {exc}")

        with tab_conformations:
            st.header("TPT Conformations Analysis")
            st.write(
                "Find metastable states, transition states, and pathways using Transition Path Theory."
            )

            conformations = backend.list_conformations()
            if conformations:
                with st.expander(
                    "Load recorded conformation analysis",
                    expanded=st.session_state.get(_LAST_CONFORMATIONS) is None,
                ):
                    indices = list(range(len(conformations)))

                    def _conformation_label(idx: int) -> str:
                        entry = conformations[idx]
                        output_dir = entry.get("output_dir", "")
                        label = Path(output_dir).name if output_dir else f"conformations-{idx}"
                        created = entry.get("created_at", "unknown")
                        return f"{label} (created {created})"

                    selected_conf_idx = st.selectbox(
                        "Stored conformations",
                        options=indices,
                        format_func=_conformation_label,
                        key="load_conformations_select",
                    )
                    if st.button("Show conformations", key="load_conformations_button"):
                        entry = conformations[int(selected_conf_idx)]
                        loaded = backend.load_conformations(entry)
                        if loaded is not None:
                            st.session_state[_LAST_CONFORMATIONS] = loaded
                            st.session_state[_CONFORMATIONS_FEEDBACK] = (
                                "success",
                                f"Loaded conformations from {loaded.output_dir.name}.",
                            )
                        else:
                            st.session_state[_CONFORMATIONS_FEEDBACK] = (
                                "error",
                                "Could not load the selected conformations bundle from disk.",
                            )

            shard_groups = backend.shard_summaries()
            selected_paths: List[Path] = []
            topology_path_str = ""
            conf_cv_method = "tica"
            conf_deeptica_projection: Optional[Path] = None
            conf_deeptica_metadata: Optional[Path] = None
            conf_lag = int(st.session_state.get("conf_lag", 10))
            conf_n_clusters = int(st.session_state.get("conf_n_clusters", 100))
            conf_n_components = int(st.session_state.get("conf_n_components", 3))
            conf_committor_thresholds = tuple(
                float(v)
                for v in st.session_state.get("conf_committor_thresholds", (0.1, 0.9))
            )
            conf_cluster_mode = str(st.session_state.get("conf_cluster_mode", "kmeans"))
            conf_cluster_seed = int(st.session_state.get("conf_cluster_seed", 42))
            conf_kmeans_n_init = int(st.session_state.get("conf_kmeans_n_init", 50))
            conf_n_metastable = int(st.session_state.get("conf_n_metastable", 10))
            conf_temperature = float(st.session_state.get("conf_temperature", 300.0))
            conf_n_paths = int(st.session_state.get("conf_n_paths", 10))
            conf_auto_detect = bool(st.session_state.get("conf_auto_detect", True))
            conf_compute_kis = bool(st.session_state.get("conf_compute_kis", True))
            conf_uncertainty = bool(st.session_state.get("conf_uncertainty_analysis", True))
            conf_bootstrap_samples = int(st.session_state.get("conf_bootstrap_samples", 50))

            if not shard_groups:
                st.info("Emit shards to run conformations analysis.")
            else:
                run_ids = [str(entry.get("run_id")) for entry in shard_groups]
                selected_runs = st.multiselect(
                    "Shard groups for conformations",
                    options=run_ids,
                    default=run_ids,
                    key="conf_selected_runs",
                )
                selected_paths = _select_shard_paths(shard_groups, selected_runs)
                try:
                    _, shard_summary = _summarize_selected_shards(selected_paths)
                except ValueError as exc:
                    st.error(f"Shard selection invalid: {exc}")
                    st.stop()
                st.write(
                    f"Using {len(selected_paths)} shard files for conformations analysis."
                )
                if shard_summary:
                    st.caption(shard_summary)

                with st.expander(
                    "Configure Conformations Analysis", expanded=False
                ):
                    # Topology Selection
                    st.markdown(" Topology Selection")
                    available_topologies = layout.available_inputs()
                    topology_select_col, topology_manual_col = st.columns(2)
                    selected_topology: Optional[Path] = None
                    if available_topologies:
                        selected_topology = topology_select_col.selectbox(
                            "Topology PDB (from app_intputs/)",
                            options=available_topologies,
                            format_func=lambda p: p.name,
                            key="conf_topology_select",
                        )
                    else:
                        topology_select_col.warning(
                            "No PDB files detected in app_intputs/. Provide the topology used during sampling."
                        )

                    manual_topology_entry = topology_manual_col.text_input(
                        "Custom topology PDB path",
                        value="",
                        help=(
                            "Optional manual override. Provide an absolute path or a path relative to the workspace "
                            "for the topology used when running the simulations."
                        ),
                        key="conf_topology_manual",
                    ).strip()

                    topology_path_str = (
                        manual_topology_entry
                        if manual_topology_entry
                        else (str(selected_topology) if selected_topology is not None else "")
                    )

                    if topology_path_str:
                        topology_candidate = Path(topology_path_str)
                        if not topology_candidate.is_absolute():
                            topology_candidate = (layout.workspace_dir / topology_candidate).resolve()
                        if not topology_candidate.exists():
                            st.warning(
                                f"Topology PDB {topology_path_str!s} does not exist. The analysis run will fail until a valid file is provided."
                            )

                    st.divider()

                    # Basic Analysis Parameters
                    st.markdown(" Basic Analysis Parameters")
                    conf_col1, conf_col2, conf_col3 = st.columns(3)
                    conf_lag = conf_col1.number_input(
                        "Lag (steps)",
                        min_value=1,
                        value=int(conf_lag),
                        step=1,
                        key="conf_lag",
                        help="Time delay for MSM construction"
                    )
                    conf_n_components = conf_col2.number_input(
                        "TICA components",
                        min_value=2,
                        value=int(conf_n_components),
                        step=1,
                        key="conf_n_components",
                        help="Number of TICA dimensions for dimensionality reduction"
                    )
                    conf_temperature = conf_col3.number_input(
                        "Temperature (K)",
                        min_value=0.0,
                        value=float(conf_temperature),
                        step=5.0,
                        key="conf_temperature",
                        help="Reference temperature for free energy calculations"
                    )

                    st.divider()

                    # Clustering Configuration
                    st.markdown(" Clustering Configuration")
                    conf_cluster_col1, conf_cluster_col2 = st.columns(2)
                    conf_n_clusters = conf_cluster_col1.number_input(
                        "N microstates (clusters)",
                        min_value=50,
                        max_value=1000,
                        value=int(conf_n_clusters),
                        step=10,
                        help=(
                            "Controls the number of clusters used when constructing the"
                            " conformational microstates. Increase this to resolve"
                            " additional transition states."
                        ),
                        key="conf_n_clusters_input",
                    )
                    conf_cluster_mode = conf_cluster_col2.selectbox(
                        "Clustering method",
                        options=["kmeans", "minibatchkmeans", "auto"],
                        index=["kmeans", "minibatchkmeans", "auto"].index(
                            conf_cluster_mode if conf_cluster_mode in {"kmeans", "minibatchkmeans", "auto"} else "kmeans"
                        ),
                        key="conf_cluster_mode",
                        help="Algorithm for partitioning CV space"
                    )

                    conf_cluster_col3, conf_cluster_col4 = st.columns(2)
                    conf_cluster_seed = conf_cluster_col3.number_input(
                        "Clustering seed (-1 for random)",
                        min_value=-1,
                        value=int(conf_cluster_seed),
                        step=1,
                        key="conf_cluster_seed",
                        help="Random seed for reproducible clustering"
                    )
                    conf_kmeans_n_init = conf_cluster_col4.number_input(
                        "K-means n_init",
                        min_value=1,
                        value=int(conf_kmeans_n_init),
                        step=1,
                        key="conf_kmeans_n_init",
                        help="Number of k-means initializations"
                    )

                    st.divider()

                    # TPT Configuration
                    st.markdown(" Transition Path Theory (TPT) Configuration")
                    conf_col4, conf_col5, conf_col6 = st.columns(3)
                    conf_n_metastable = conf_col4.number_input(
                        "N metastable states",
                        min_value=2,
                        value=int(conf_n_metastable),
                        step=1,
                        key="conf_n_metastable_form",
                        on_change=_sync_form_metastable_states,
                        help="Number of metastable states to identify"
                    )
                    st.session_state["conf_n_metastable"] = int(conf_n_metastable)
                    conf_n_paths = conf_col5.number_input(
                        "Max pathways",
                        min_value=1,
                        value=int(conf_n_paths),
                        step=1,
                        key="conf_n_paths",
                        help="Maximum number of transition pathways to compute"
                    )
                    conf_compute_kis = conf_col6.checkbox(
                        "Compute Kinetic Importance Score",
                        value=bool(conf_compute_kis),
                        key="conf_compute_kis",
                        help="Calculate kinetic importance for each state"
                    )

                    conf_auto_detect = st.checkbox(
                        "Auto-detect source/sink states",
                        value=bool(conf_auto_detect),
                        key="conf_auto_detect",
                        help="Automatically identify source and sink states from metastable populations"
                    )

                    st.divider()

                    # Uncertainty Analysis
                    st.markdown(" Uncertainty Analysis")
                    conf_col9, conf_col10 = st.columns(2)
                    conf_uncertainty = conf_col9.checkbox(
                        "Perform uncertainty analysis",
                        value=bool(conf_uncertainty),
                        help="Run bootstrap estimates for TPT observables and free energies.",
                        key="conf_uncertainty_analysis",
                    )
                    conf_bootstrap_samples = conf_col10.number_input(
                        "Bootstrap samples",
                        min_value=1,
                        value=int(conf_bootstrap_samples),
                        step=5,
                        help="Number of bootstrap resamples used during uncertainty analysis.",
                        key="conf_bootstrap_samples",
                        disabled=not conf_uncertainty,
                    )

                    st.divider()

                    # CV Method Selection
                    st.markdown(" Collective Variable (CV) Method")
                    conf_cv_col1, conf_cv_col2 = st.columns(2)
                    conf_cv_method = conf_cv_col1.selectbox(
                        "CV method",
                        options=["tica", "deeptica"],
                        index=0 if conf_cv_method != "deeptica" else 1,
                        key="conf_cv_method",
                        help="Choose between classical TICA or Deep-TICA for dimensionality reduction"
                    )
                    conf_deeptica_projection = None
                    conf_deeptica_metadata = None
                    if conf_cv_method == "deeptica":
                        projection_str = conf_cv_col2.text_input(
                            "DeepTICA projection path",
                            value="",
                            key="conf_deeptica_projection",
                            help="Path to a .npz/.npy file containing precomputed DeepTICA CVs.",
                        )
                        conf_deeptica_projection = Path(projection_str) if projection_str else None
                        metadata_str = st.text_input(
                            "DeepTICA whitening metadata (optional)",
                            value="",
                            key="conf_deeptica_metadata",
                            help="Optional JSON file describing the whitening transform for the DeepTICA outputs.",
                        )
                        conf_deeptica_metadata = Path(metadata_str) if metadata_str else None

            if st.button(
                "Run Conformations Analysis",
                type="primary",
                disabled=(
                    len(selected_paths) == 0
                    or not topology_path_str
                    or (
                        conf_cv_method == "deeptica" and conf_deeptica_projection is None
                    )
                ),
                key="conformations_button",
            ):
                try:
                    conf_config = ConformationsConfig(
                        lag=int(conf_lag),
                        n_clusters=int(conf_n_clusters),
                        cluster_mode=str(conf_cluster_mode),
                        cluster_seed=(
                            int(conf_cluster_seed)
                            if int(conf_cluster_seed) >= 0
                            else None
                        ),
                        kmeans_n_init=int(conf_kmeans_n_init),
                        n_components=int(conf_n_components),
                        n_metastable=int(conf_n_metastable),
                        temperature=float(conf_temperature),
                        auto_detect_states=bool(conf_auto_detect),
                        n_paths=int(conf_n_paths),
                        compute_kis=bool(conf_compute_kis),
                        uncertainty_analysis=bool(conf_uncertainty),
                        bootstrap_samples=int(conf_bootstrap_samples),
                        topology_pdb=Path(topology_path_str),
                        cv_method=str(conf_cv_method),
                        deeptica_projection_path=conf_deeptica_projection,
                        deeptica_metadata_path=conf_deeptica_metadata,
                        committor_thresholds=tuple(conf_committor_thresholds),
                    )

                    with st.spinner("Running conformations analysis..."):
                        conf_result = backend.run_conformations_analysis(
                            selected_paths, conf_config
                        )

                    st.session_state[_LAST_CONFORMATIONS] = conf_result
                    if conf_result.error:
                        st.session_state[_CONFORMATIONS_FEEDBACK] = (
                            "error",
                            f"Conformations analysis failed: {conf_result.error}",
                        )
                    else:
                        st.session_state[_CONFORMATIONS_FEEDBACK] = (
                            "success",
                            f"Conformations analysis complete! Output saved to {conf_result.output_dir.name}",
                        )
                except Exception as exc:
                    traceback.print_exc()
                    st.session_state[_CONFORMATIONS_FEEDBACK] = (
                        "error",
                        f"Conformations analysis failed: {exc}",
                    )

            feedback = st.session_state.get(_CONFORMATIONS_FEEDBACK)
            if isinstance(feedback, tuple) and len(feedback) == 2:
                level, message = feedback
                if level == "success":
                    st.success(message)
                elif level == "warning":
                    st.warning(message)
                elif level == "info":
                    st.info(message)
                else:
                    st.error(message)

            last_conf = st.session_state.get(_LAST_CONFORMATIONS)
            if isinstance(last_conf, ConformationsResult):
                _render_conformations_result(last_conf)

        with tab_model_preview:
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

        with tab_assets:
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

        with tab_validation:
            st.header("Free Energy Validation")

            shard_groups = backend.shard_summaries()
            if not shard_groups:
                st.info("Emit shards first to generate validation plots.")
            else:
                run_ids = [str(entry.get("run_id")) for entry in shard_groups]
                selected_runs = st.multiselect(
                    "Select shard groups for validation",
                    options=run_ids,
                    default=run_ids,
                    key="validation_selected_runs",
                )
                selected_paths = _select_shard_paths(shard_groups, selected_runs)

                if not selected_paths:
                    st.warning("Select at least one shard group to generate validation plots.")
                else:
                    try:
                        _, shard_summary = _summarize_selected_shards(selected_paths)
                        st.caption(f"Using {len(selected_paths)} shards: {shard_summary}")
                    except ValueError as exc:
                        st.error(f"Invalid shard selection: {exc}")
                        st.stop()

                    # TICA Parameters
                    st.subheader("TICA Projection Parameters")
                    col1, col2, col3 = st.columns(3)
                    val_n_components = col1.number_input(
                        "TICA components",
                        min_value=2,
                        max_value=20,
                        value=3,
                        key="validation_n_components",
                    )
                    val_lag = col2.number_input(
                        "TICA lag",
                        min_value=1,
                        max_value=100,
                        value=10,
                        key="validation_lag",
                    )
                    val_temperature = col3.number_input(
                        "Temperature (K)",
                        min_value=0.0,
                        value=300.0,
                        step=10.0,
                        key="validation_temperature",
                    )

                    st.divider()

                    # --- Sampling Plot Controls ---
                    st.subheader("Sampling Plot Appearance Controls")
                    st.markdown(
                        """
                        Adjust visualization parameters for the trajectory sampling validation plot (1D histogram on TICA 1).
                        """
                    )
                    col_samp1, col_samp2, col_samp3 = st.columns(3)
                    with col_samp1:
                        st.number_input(
                            "Max Trajectory Length to Plot",
                            min_value=100,
                            max_value=20000,
                            value=st.session_state.get("val_plot_max_len", 1000),
                            step=100,
                            key="val_plot_max_len",
                            help="Maximum number of frames per shard to visualize."
                        )
                    with col_samp2:
                        st.number_input(
                            "Histogram Bins",
                            min_value=10,
                            max_value=500,
                            value=st.session_state.get("val_plot_hist_bins", 150),
                            step=10,
                            key="val_plot_hist_bins",
                            help="Number of bins for the 1D histogram."
                        )
                    with col_samp3:
                        st.number_input(
                            "Trajectory Point Stride",
                            min_value=1,
                            max_value=100,
                            value=st.session_state.get("val_plot_stride", 10),
                            step=1,
                            key="val_plot_stride",
                            help="Plot every N-th point of the trajectory path for clarity."
                        )

                    st.divider()

                    # --- FES Plot Controls ---
                    st.subheader("Free Energy Surface (FES) Plot Controls")
                    st.markdown(
                        """
                        Customize the 2D Free Energy Surface visualization on TICA 1 vs TICA 2.
                        """
                    )
                    col_fes1, col_fes2, col_fes3 = st.columns(3)
                    with col_fes1:
                        st.slider(
                            "Max Free Energy (kT)",
                            min_value=1.0,
                            max_value=20.0,
                            value=st.session_state.get("fes_plot_max_kt", 7.0),
                            step=0.5,
                            key="fes_plot_max_kt",
                            help="Cap the color scale at this energy value (in kT)."
                        )
                    with col_fes2:
                        st.number_input(
                            "Contour Levels",
                            min_value=5,
                            max_value=100,
                            value=st.session_state.get("fes_plot_levels", 25),
                            step=5,
                            key="fes_plot_levels",
                            help="Number of levels for the contour plot."
                        )
                    with col_fes3:
                        # Get list of available colormaps
                        available_colormaps = sorted([
                            'viridis', 'plasma', 'inferno', 'magma', 'cividis',
                            'coolwarm', 'RdYlBu', 'seismic', 'twilight', 'jet'
                        ])
                        default_cmap = st.session_state.get("fes_plot_cmap", "viridis")
                        if default_cmap not in available_colormaps:
                            default_cmap = "viridis"
                        cmap_index = available_colormaps.index(default_cmap)

                        st.selectbox(
                            "Colormap",
                            available_colormaps,
                            index=cmap_index,
                            key="fes_plot_cmap"
                        )

                    st.checkbox(
                        "Show Contour Lines",
                        value=st.session_state.get("fes_plot_lines", True),
                        key="fes_plot_lines"
                    )

                    st.divider()

                    if st.button("Generate Validation Plots", key="run_validation", type="primary"):
                        try:
                            from pmarlo.data.aggregate import load_shards_as_dataset
                            from pmarlo.markov_state_model.reduction import reduce_features
                            from pmarlo.markov_state_model.free_energy import generate_2d_fes

                            with st.spinner("Loading shard data..."):
                                dataset = load_shards_as_dataset(selected_paths)
                                X_list = dataset.get("X", [])
                                if X_list is None or (isinstance(X_list, list) and len(X_list) == 0):
                                    st.error("No feature data found in shards.")
                                    st.stop()

                            with st.spinner("Computing TICA projection..."):
                                projection = reduce_features(
                                    X_list,
                                    method="tica",
                                    lag=int(val_lag),
                                    n_components=int(val_n_components),
                                )
                                if projection is None:
                                    st.error("TICA projection failed.")
                                    st.stop()

                            st.success(f"Computed TICA projection with shape {projection.shape}")

                            col_left, col_right = st.columns(2)

                            with col_left:
                                st.subheader("Sampling Connectivity")
                                with st.spinner("Generating sampling plot..."):
                                    try:
                                        # Get trajectory lengths from the dataset
                                        traj_lengths = [len(traj) for traj in X_list]
                                        # Split projection back into per-trajectory arrays
                                        projection_list = []
                                        start_idx = 0
                                        for length in traj_lengths:
                                            projection_list.append(projection[start_idx:start_idx + length])
                                            start_idx += length

                                        # Create a mock app_state object with projection data
                                        class MockAppState:
                                            def __init__(self, proj_list):
                                                self.projection_data = proj_list

                                        mock_state = MockAppState(projection_list)
                                        sampling_fig = create_sampling_validation_plot(mock_state)

                                        if sampling_fig and hasattr(sampling_fig, 'axes'):
                                            st.pyplot(
                                                sampling_fig,
                                                clear_figure=True,
                                                width="stretch",
                                            )
                                        else:
                                            st.warning("Could not generate sampling validation plot.")
                                    except Exception as sampling_err:
                                        st.warning(f"Could not generate sampling plot: {sampling_err}")
                                        import traceback
                                        with st.expander("Show error details"):
                                            st.code(traceback.format_exc())

                            with col_right:
                                st.subheader("Free Energy Surface")
                                with st.spinner("Computing FES..."):
                                    try:
                                        if projection.ndim == 2 and projection.shape[1] >= 2:
                                            fes_result = generate_2d_fes(
                                                cv1=projection[:, 0],
                                                cv2=projection[:, 1],
                                                temperature=float(val_temperature),
                                                bins=(50, 50),
                                            )

                                            # Create a mock app_state object with FES data
                                            class MockAppStateFES:
                                                def __init__(self, fes_result):
                                                    # Extract bin centers from edges
                                                    x_centers = 0.5 * (fes_result.xedges[:-1] + fes_result.xedges[1:])
                                                    y_centers = 0.5 * (fes_result.yedges[:-1] + fes_result.yedges[1:])

                                                    # Create meshgrid
                                                    xx, yy = np.meshgrid(x_centers, y_centers)
                                                    self.fes_grid = [xx, yy]

                                                    # Extract free energy values
                                                    self.fes_data = fes_result.F

                                            mock_state_fes = MockAppStateFES(fes_result)
                                            fes_fig = create_fes_validation_plot(mock_state_fes)

                                            if fes_fig and hasattr(fes_fig, 'axes'):
                                                st.pyplot(
                                                    fes_fig,
                                                    clear_figure=True,
                                                    width="stretch",
                                                )
                                            else:
                                                st.warning("Could not generate Free Energy Surface plot.")
                                        else:
                                            st.warning("Need at least 2 TICA components for FES.")
                                    except Exception as fes_err:
                                        st.warning(f"Could not generate FES plot: {fes_err}")
                                        import traceback
                                        with st.expander("Show error details"):
                                            st.code(traceback.format_exc())

                        except Exception as e:
                            st.error(f"Validation failed: {e}")
                            import traceback
                            with st.expander("Show error details"):
                                st.code(traceback.format_exc())
    with tab_its:
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

            with st.form("its_configuration"):
                selected_runs = st.multiselect(
                    "Shard groups",
                    options=run_ids,
                    key="its_selected_runs",
                    help="Select shard batches that should contribute to the ITS analysis.",
                )
                selected_paths = _select_shard_paths(shard_groups, selected_runs)
                selection_text = ""
                if selected_paths:
                    try:
                        _, selection_text = _summarize_selected_shards(selected_paths)
                    except ValueError as exc:
                        st.error(f"Shard selection invalid: {exc}")
                        selected_paths = []
                    else:
                        st.caption(f"Using {len(selected_paths)} shard files.")
                        if selection_text:
                            st.caption(selection_text)
                else:
                    st.info("Select at least one shard group to compute implied timescales.")

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
                compute_btn = st.form_submit_button(
                    "Compute implied timescales", type="primary"
                )

            selected_runs = st.session_state.get("its_selected_runs", [])
            selected_paths = _select_shard_paths(shard_groups, selected_runs)
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
                    st.rerun()

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
                    fig = backend_plot_its(lag_series, times_series)
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
    st.caption(
        "Run this app with: poetry run streamlit run example_programs/app_usecase/app/app.py"
    )


if __name__ == "__main__":
    main()
