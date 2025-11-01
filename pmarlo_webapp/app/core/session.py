import streamlit as st
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from backend.types import TrainingConfig, BuildConfig

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

def _apply_training_config_to_state(cfg: "TrainingConfig") -> None:
    from core.parsers import _format_tau_schedule

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

def _consume_pending_training_config() -> None:
    from backend.types import TrainingConfig

    pending = st.session_state.get(_TRAIN_CONFIG_PENDING)
    if isinstance(pending, TrainingConfig):
        _apply_training_config_to_state(pending)
    st.session_state[_TRAIN_CONFIG_PENDING] = None

def _update_state(**kwargs: Any) -> None:
    for key, value in kwargs.items():
        st.session_state[key] = value

def _ensure_session_defaults() -> None:
    from core.parsers import _format_lag_sequence

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

def _apply_analysis_config_to_state(cfg: "BuildConfig") -> None:
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