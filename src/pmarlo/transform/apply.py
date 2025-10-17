import functools
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, Tuple, cast

import numpy as np

from pmarlo.utils.path_utils import ensure_directory

from ..experiments.benchmark_utils import get_environment_info
from .plan import TransformPlan

logger = logging.getLogger(__name__)


class StepHandler(Protocol):
    def __call__(self, context: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]: ...


def smooth_fes(dataset, **kwargs):
    return dataset


def reorder_states(dataset, **kwargs):
    return dataset


def fill_gaps(dataset, **kwargs):
    return dataset


@dataclass
class _LearnCVOutcome:
    summary: Dict[str, Any]
    per_shard: List[Dict[str, Any]]
    warnings: List[str]
    pairs_total: int


class _LearnCVAbort(RuntimeError):
    def __init__(self, outcome: _LearnCVOutcome):
        super().__init__(outcome.summary.get("reason", "learn_cv_abort"))
        self.outcome = outcome


def _capture_env_payload() -> Dict[str, Any]:
    """Return environment metadata with defensive fallbacks."""

    try:
        info = dict(get_environment_info())
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.debug("Failed to capture environment info: %s", exc)
        info = {}
    info.setdefault("python_exe", sys.executable)
    return info


def _extract_missing_modules(exc: BaseException) -> List[str]:
    names: set[str] = set()
    seen: set[int] = set()

    def _recurse(err: BaseException | None) -> None:
        if err is None:
            return
        key = id(err)
        if key in seen:
            return
        seen.add(key)
        if isinstance(err, ModuleNotFoundError):
            name = getattr(err, "name", None)
            if name:
                names.add(str(name).split(".")[0])
        msg = str(err) if err else ""
        for token in (
            "lightning",
            "pytorch_lightning",
            "torch",
            "mlcolvar",
            "sklearn",
        ):
            if token in msg:
                names.add(token)
        _recurse(getattr(err, "__cause__", None))
        if not getattr(err, "__suppress_context__", False):
            _recurse(getattr(err, "__context__", None))

    _recurse(exc)
    return sorted(names)


def _format_missing_reason(mods: Sequence[str]) -> str:
    payload = ",".join(sorted(set(mods))) if mods else "unknown"
    return f"missing_dependency:{payload}"


def _probe_optional_modules(names: Sequence[str]) -> List[str]:
    import importlib

    discovered: List[str] = []
    for module_name in names:
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            extracted = _extract_missing_modules(exc)
            if extracted:
                discovered.extend(extracted)
            else:
                discovered.append(module_name)
    return sorted({str(name).split(".")[0] for name in discovered})


def _compute_pairs_metadata(
    lag_value: int,
    shard_ranges: Sequence[Tuple[int, int]],
    shards_meta: Optional[Sequence[Dict[str, Any]]],
    total_frames: int,
) -> Tuple[List[Dict[str, Any]], int, List[str]]:
    per: List[Dict[str, Any]] = []
    warnings: List[str] = []
    entries = list(shards_meta or [])
    for idx, entry in enumerate(entries):
        start = shard_ranges[idx][0] if idx < len(shard_ranges) else 0
        stop = shard_ranges[idx][1] if idx < len(shard_ranges) else 0
        frames = max(0, stop - start)
        pairs = max(0, frames - lag_value)
        shard_id = str(entry.get("id", f"shard_{idx:04d}"))
        per.append(
            {
                "id": shard_id,
                "start": int(start),
                "stop": int(stop),
                "frames": int(frames),
                "pairs": int(pairs),
            }
        )
        if pairs <= 0:
            warnings.append(f"shard_no_pairs:{shard_id}")
    total_pairs = int(sum(item["pairs"] for item in per))
    if total_pairs <= 0:
        warnings.append("pairs_total=0")
    if total_frames < max(16, lag_value * 2):
        warnings.append("low_frame_count")
    return per, total_pairs, warnings


def _finalize_learn_cv_context(
    context: Dict[str, Any],
    dataset: Dict[str, Any],
    uses_data_key: bool,
    total_frames: int,
    outcome: _LearnCVOutcome,
) -> Dict[str, Any]:
    summary = dict(outcome.summary)
    summary.setdefault("method", "deeptica")
    summary.setdefault("lag", summary.get("lag", None))
    summary.setdefault("lag_used", summary.get("lag_used"))
    summary.setdefault("n_out", 0)
    summary.setdefault("skipped", not summary.get("applied", False))
    cleaned_per = [
        {
            "id": item.get("id"),
            "start": int(item.get("start", 0)),
            "stop": int(item.get("stop", 0)),
            "frames": int(item.get("frames", 0)),
            "pairs": int(item.get("pairs", 0)),
        }
        for item in outcome.per_shard
    ]
    summary["per_shard"] = cleaned_per
    summary["n_shards"] = len(cleaned_per)
    summary["frames_total"] = total_frames
    summary.setdefault("pairs_total", int(outcome.pairs_total))
    warnings_clean = sorted({str(w) for w in outcome.warnings if w})
    if "warnings" in summary:
        warnings_clean.extend(str(w) for w in summary["warnings"] if w)
    summary["warnings"] = sorted({str(w) for w in warnings_clean})
    if isinstance(summary.get("missing"), list):
        summary["missing"] = sorted({str(m) for m in summary["missing"] if m})
    summary["env"] = _capture_env_payload()

    artifacts = dataset.setdefault("__artifacts__", {})
    artifacts["mlcv_deeptica"] = summary
    if uses_data_key:
        context["data"] = dataset
    return context


def _make_dependency_outcome(
    *,
    lag_value: int,
    missing: Sequence[str],
    warnings: List[str],
    per_shard: List[Dict[str, Any]],
    pairs_total: int,
) -> _LearnCVOutcome:
    reason = _format_missing_reason(missing)
    summary = {
        "applied": False,
        "skipped": True,
        "reason": reason,
        "lag": int(lag_value),
        "lag_used": None,
        "n_out": 0,
        "missing": list(missing),
    }
    all_warnings = list(warnings) + ["missing_dependencies"]
    return _LearnCVOutcome(summary, per_shard, all_warnings, pairs_total)


def _extract_learn_cv_dataset(context: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    dataset: Optional[Dict[str, Any]] = None
    uses_data_key = False
    if isinstance(context, dict) and isinstance(context.get("data"), dict):
        dataset = context["data"]
        uses_data_key = True
    elif isinstance(context, dict):
        dataset = context
    if not isinstance(dataset, dict):
        raise RuntimeError("LEARN_CV requires a mapping dataset with CV arrays")
    if "X" not in dataset:
        raise RuntimeError("LEARN_CV expects dataset['X'] containing CV features")
    return dataset, uses_data_key


def _prepare_learn_cv_arrays(dataset: Dict[str, Any]) -> Tuple[
    np.ndarray,
    List[Dict[str, Any]],
    List[Tuple[int, int]],
    List[np.ndarray],
]:
    X_all = np.asarray(dataset.get("X"), dtype=np.float64)
    if X_all.ndim != 2 or X_all.shape[0] == 0:
        raise RuntimeError("LEARN_CV requires a non-empty 2D feature matrix")

    shards_meta = dataset.get("__shards__")
    if not isinstance(shards_meta, list) or not shards_meta:
        shards_meta = [{"id": "shard_0000", "start": 0, "stop": X_all.shape[0]}]

    shard_ranges: List[Tuple[int, int]] = []
    X_list: List[np.ndarray] = []
    for entry in shards_meta:
        try:
            start = int(entry.get("start", 0))
            stop = int(entry.get("stop", start))
        except Exception:
            continue
        start = max(0, start)
        stop = max(start, min(stop, X_all.shape[0]))
        if stop <= start:
            continue
        shard_ranges.append((start, stop))
        X_list.append(X_all[start:stop])

    if not X_list:
        raise RuntimeError("LEARN_CV requires at least one shard with frames")

    return X_all, shards_meta, shard_ranges, X_list


def _collect_lag_candidates(params: Dict[str, Any], tau_requested: int) -> List[int]:
    def _coerce_one(value: Any) -> Optional[int]:
        try:
            coerced = int(value)
        except Exception:
            return None
        return coerced if coerced > 0 else None

    candidates: List[int] = []
    primary = _coerce_one(params.get("lag", tau_requested))
    candidates.append(primary or tau_requested)

    fallback = params.get("lag_fallback")
    if isinstance(fallback, (list, tuple)):
        for entry in fallback:
            coerced = _coerce_one(entry)
            if coerced is not None:
                candidates.append(coerced)
    elif fallback is not None:
        coerced = _coerce_one(fallback)
        if coerced is not None:
            candidates.append(coerced)

    if not candidates:
        candidates = [tau_requested]
    return candidates


@dataclass
class _LagSelection:
    cfg: Any
    tau: int
    per_shard_info: List[Dict[str, Any]]
    pairs_estimate: int
    warnings: List[str]
    attempt_details: List[Dict[str, Any]]
    seen_lags: List[int]


def _select_deeptica_config(
    *,
    params: Dict[str, Any],
    tau_requested: int,
    DeepTICAConfig: Any,
    shards_meta: List[Dict[str, Any]],
    shard_ranges: List[Tuple[int, int]],
    total_frames: int,
) -> _LagSelection:
    cfg_fields = getattr(DeepTICAConfig, "__annotations__", {}).keys()
    cfg_kwargs_base = {k: params[k] for k in params if k in cfg_fields and k != "lag"}
    if int(cfg_kwargs_base.get("n_out", params.get("n_out", 2))) < 2:
        cfg_kwargs_base["n_out"] = 2

    candidate_sequence = _collect_lag_candidates(params, tau_requested)

    seen_lags: List[int] = []
    attempt_details: List[Dict[str, Any]] = []
    cfg = None
    tau = tau_requested
    per_shard_info: List[Dict[str, Any]] = []
    pairs_estimate = 0
    warnings: List[str] = []

    for candidate in candidate_sequence:
        lag_value = int(max(1, candidate))
        if lag_value in seen_lags:
            continue
        seen_lags.append(lag_value)

        attempt_kwargs = dict(cfg_kwargs_base)
        attempt_kwargs["lag"] = lag_value
        cfg_attempt = DeepTICAConfig(**attempt_kwargs)
        tau_attempt = int(max(1, getattr(cfg_attempt, "lag", lag_value)))
        per_shard_attempt, pairs_attempt, warnings_attempt = _compute_pairs_metadata(
            tau_attempt, shard_ranges, shards_meta, total_frames
        )
        attempt_details.append(
            {
                "lag": int(lag_value),
                "pairs_total": int(pairs_attempt),
                "status": "ok" if pairs_attempt > 0 else "no_pairs",
                "per_shard_pairs": [int(item["pairs"]) for item in per_shard_attempt],
                "warnings": [str(w) for w in warnings_attempt],
            }
        )
        cfg = cfg_attempt
        tau = tau_attempt
        per_shard_info = per_shard_attempt
        pairs_estimate = pairs_attempt
        warnings = warnings_attempt
        if pairs_estimate > 0:
            break

    if cfg is None:
        raise RuntimeError("Failed to instantiate Deep-TICA configuration")

    return _LagSelection(
        cfg=cfg,
        tau=tau,
        per_shard_info=per_shard_info,
        pairs_estimate=pairs_estimate,
        warnings=warnings,
        attempt_details=attempt_details,
        seen_lags=seen_lags,
    )


def _build_no_pairs_outcome(
    *,
    cfg: Any,
    selection: _LagSelection,
    extra_warnings: Optional[List[str]] = None,
) -> _LearnCVOutcome:
    warnings = list(selection.warnings)
    if extra_warnings:
        warnings.extend(extra_warnings)
    summary = {
        "applied": False,
        "skipped": True,
        "reason": "no_pairs",
        "lag": int(getattr(cfg, "lag", selection.tau)),
        "lag_used": None,
        "n_out": 0,
        "lag_candidates": [int(v) for v in selection.seen_lags],
        "lag_fallback": [int(v) for v in selection.seen_lags],
        "attempts": selection.attempt_details,
    }
    return _LearnCVOutcome(
        summary,
        selection.per_shard_info,
        warnings,
        selection.pairs_estimate,
    )


def _build_pair_indices(
    shard_ranges: Sequence[Tuple[int, int]], tau: int
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    i_parts: List[np.ndarray] = []
    j_parts: List[np.ndarray] = []
    for start, stop in shard_ranges:
        length = stop - start
        if length <= tau:
            continue
        idx = np.arange(start, stop - tau, dtype=np.int64)
        if idx.size == 0:
            continue
        i_parts.append(idx)
        j_parts.append(idx + tau)
    if not i_parts:
        return None
    return np.concatenate(i_parts), np.concatenate(j_parts)


def _resolve_model_loader(
    params: Dict[str, Any], train_deeptica: Any
) -> Callable[..., Any]:
    try:
        from . import build as build_mod  # Local import to avoid circular dependency

        load_model = getattr(build_mod, "_load_or_train_model", None)
    except Exception:
        load_model = None

    if load_model is None:

        def load_model(
            X_seq: Sequence[np.ndarray],
            lagged_pairs: Tuple[np.ndarray, np.ndarray],
            cfg_obj: Any,
            *,
            weights: Optional[np.ndarray] = None,
            train_fn: Optional[Any] = None,
            **_: Any,
        ) -> Any:
            fn = train_fn or train_deeptica
            return fn(X_seq, lagged_pairs, cfg_obj, weights=weights)

    return load_model


def _classify_training_failure(exc: BaseException) -> Tuple[str, Dict[str, Any]]:
    import traceback as _traceback

    payload: Dict[str, Any] = {
        "error": str(exc),
        "traceback": _traceback.format_exc(),
    }
    missing = _extract_missing_modules(exc)
    if missing:
        payload["missing"] = missing
        return _format_missing_reason(missing), payload
    name = exc.__class__.__name__
    if "PmarloApiIncompatibilityError" in name:
        return "api_incompatibility", payload
    return "exception", payload


def _make_training_failure_outcome(
    *,
    cfg: Any,
    selection: _LagSelection,
    warnings: List[str],
    exc: BaseException,
) -> _LearnCVOutcome:
    reason, extra = _classify_training_failure(exc)
    summary = {
        "applied": False,
        "skipped": True,
        "reason": reason,
        "lag": int(getattr(cfg, "lag", selection.tau)),
        "lag_used": None,
        "n_out": 0,
    }
    missing_extra = extra.pop("missing", None)
    if missing_extra:
        summary["missing"] = missing_extra
    summary.update(extra)
    return _LearnCVOutcome(
        summary,
        selection.per_shard_info,
        list(warnings) + [reason],
        selection.pairs_estimate,
    )


def _convert_history_array(value: Any) -> Any:
    if value is None:
        return None
    try:
        return np.asarray(value, dtype=np.float64).tolist()
    except Exception:
        return value


def _collect_history_metrics(history: Dict[str, Any]) -> Dict[str, Any]:
    output_mean = history.get("output_mean") if isinstance(history, dict) else None
    output_transform = (
        history.get("output_transform") if isinstance(history, dict) else None
    )
    history_flag = (
        bool(history.get("output_transform_applied"))
        if isinstance(history, dict)
        else False
    )
    export_transform_applied = bool(
        output_mean is not None and output_transform is not None
    )
    transform_applied_flag = history_flag or export_transform_applied
    initial_objective_value = history.get("initial_objective")
    initial_objective_float = (
        0.0 if initial_objective_value is None else float(initial_objective_value)
    )

    summary = {
        "wall_time_s": float(history.get("wall_time_s", 0.0)),
        "initial_objective": (
            initial_objective_float if initial_objective_value is not None else None
        ),
        "output_variance": history.get("output_variance"),
        "loss_curve_last": (
            float(history["loss_curve"][-1])
            if isinstance(history.get("loss_curve"), list) and history.get("loss_curve")
            else None
        ),
        "objective_last": (
            float(history["objective_curve"][-1])
            if isinstance(history.get("objective_curve"), list)
            and history.get("objective_curve")
            else None
        ),
        "val_score_last": (
            float(history["val_score_curve"][-1])
            if isinstance(history.get("val_score_curve"), list)
            and history.get("val_score_curve")
            else None
        ),
        "var_z0_last": (
            history.get("var_z0_curve", [None])[-1]
            if isinstance(history.get("var_z0_curve"), list)
            and history.get("var_z0_curve")
            else None
        ),
        "var_zt_last": (
            history.get("var_zt_curve", [None])[-1]
            if isinstance(history.get("var_zt_curve"), list)
            and history.get("var_zt_curve")
            else None
        ),
        "cond_c00_last": (
            float(history.get("cond_c00_curve", [None])[-1])
            if isinstance(history.get("cond_c00_curve"), list)
            and history.get("cond_c00_curve")
            else None
        ),
        "cond_ctt_last": (
            float(history.get("cond_ctt_curve", [None])[-1])
            if isinstance(history.get("cond_ctt_curve"), list)
            and history.get("cond_ctt_curve")
            else None
        ),
        "grad_norm_last": (
            float(history.get("grad_norm_curve", [None])[-1])
            if isinstance(history.get("grad_norm_curve"), list)
            and history.get("grad_norm_curve")
            else None
        ),
        "output_mean": output_mean,
        "output_transform": output_transform,
        "output_transform_applied": transform_applied_flag,
    }

    if summary.get("output_mean") is not None:
        summary["output_mean"] = _convert_history_array(summary["output_mean"])
    if summary.get("output_transform") is not None:
        summary["output_transform"] = _convert_history_array(
            summary["output_transform"]
        )

    for key in (
        "loss_curve",
        "objective_curve",
        "val_score_curve",
        "var_z0_curve",
        "var_zt_curve",
        "cond_c00_curve",
        "cond_ctt_curve",
        "grad_norm_curve",
        "val_score",
    ):
        if isinstance(history.get(key), list) and history.get(key):
            summary[key] = history[key]

    return summary


def _save_trained_model(
    model: Any,
    *,
    model_dir: Optional[str],
    params: Dict[str, Any],
) -> Tuple[Optional[str], List[str]]:
    saved_prefix = None
    saved_files: List[str] = []
    if model_dir:
        try:
            base_dir = Path(model_dir)
            ensure_directory(base_dir)
            stem = (
                params.get("model_prefix")
                or f"deeptica-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            )
            base_path = base_dir / stem
            model.save(base_path)
            saved_prefix = str(base_path)
            for suffix in (
                ".json",
                ".pt",
                ".scaler.pt",
                ".history.json",
                ".history.csv",
            ):
                candidate_path = base_path.with_suffix(suffix)
                if candidate_path.exists():
                    saved_files.append(str(candidate_path))
        except Exception as exc:
            logger.warning("Failed to persist Deep-TICA model: %s", exc)
    return saved_prefix, saved_files


def _ensure_deeptica_module(
    *,
    tau_requested: int,
    params: Dict[str, Any],
    per_shard_info: List[Dict[str, Any]],
    warnings: List[str],
    pairs_estimate: int,
) -> Any:
    try:
        import pmarlo.features.deeptica as deeptica_mod
    except ImportError as exc:
        outcome = _make_dependency_outcome(
            lag_value=tau_requested,
            missing=_extract_missing_modules(exc),
            warnings=warnings,
            per_shard=per_shard_info,
            pairs_total=pairs_estimate,
        )
        raise _LearnCVAbort(outcome) from exc

    missing_exc = getattr(deeptica_mod, "_IMPORT_ERROR", None)
    if missing_exc is not None:
        outcome = _make_dependency_outcome(
            lag_value=tau_requested,
            missing=_extract_missing_modules(missing_exc),
            warnings=warnings,
            per_shard=per_shard_info,
            pairs_total=pairs_estimate,
        )
        raise _LearnCVAbort(outcome)

    probe_missing = _probe_optional_modules(["lightning", "pytorch_lightning"])
    if probe_missing:
        outcome = _make_dependency_outcome(
            lag_value=tau_requested,
            missing=probe_missing,
            warnings=warnings,
            per_shard=per_shard_info,
            pairs_total=pairs_estimate,
        )
        raise _LearnCVAbort(outcome)

    return deeptica_mod


def learn_cv_step(context: Dict[str, Any], **params) -> Dict[str, Any]:
    """Train learned CVs (Deep-TICA) and replace dataset features."""

    dataset, uses_data_key = _extract_learn_cv_dataset(context)
    _require_deeptica_method(params)

    X_all, shards_meta, shard_ranges, X_list = _prepare_learn_cv_arrays(dataset)
    total_frames = int(X_all.shape[0])
    finalize = functools.partial(
        _finalize_learn_cv_context,
        context,
        dataset,
        uses_data_key,
        total_frames,
    )

    tau_requested = int(max(1, params.get("lag", 5)))
    per_shard_info, pairs_estimate, warnings = _compute_pairs_metadata(
        tau_requested, shard_ranges, shards_meta, total_frames
    )

    try:
        deeptica_mod = _ensure_deeptica_module(
            tau_requested=tau_requested,
            params=params,
            per_shard_info=per_shard_info,
            warnings=warnings,
            pairs_estimate=pairs_estimate,
        )
    except _LearnCVAbort as abort:
        return finalize(abort.outcome)

    DeepTICAConfig = deeptica_mod.DeepTICAConfig
    train_deeptica = getattr(deeptica_mod, "train_deeptica")

    selection = _select_deeptica_config(
        params=params,
        tau_requested=tau_requested,
        DeepTICAConfig=DeepTICAConfig,
        shards_meta=shards_meta,
        shard_ranges=shard_ranges,
        total_frames=total_frames,
    )

    lag_value = int(getattr(selection.cfg, "lag", selection.tau))
    warnings_list = _prepare_selection_warnings(selection, lag_value)

    if selection.pairs_estimate <= 0:
        outcome = _build_no_pairs_outcome(
            cfg=selection.cfg, selection=selection, extra_warnings=["no_pairs"]
        )
        return finalize(outcome)

    pair_indices = _build_pair_indices(shard_ranges, selection.tau)
    if pair_indices is None:
        outcome = _build_no_pairs_outcome(
            cfg=selection.cfg, selection=selection, extra_warnings=["no_pairs"]
        )
        return finalize(outcome)

    try:
        model = _train_deeptica_model(
            params,
            train_deeptica,
            selection.cfg,
            X_list,
            pair_indices,
            warnings_list,
            selection,
        )
    except _LearnCVAbort as abort:
        return finalize(abort.outcome)

    Y = _transform_features_with_model(model, X_all)
    n_out = _store_transformed_features(dataset, Y)

    history = _ensure_training_history(model)
    metrics = _collect_history_metrics(history)

    summary = _build_success_summary(
        selection,
        lag_value,
        pair_indices,
        n_out,
        metrics,
    )
    _attach_model_artifacts(summary, model, params)

    outcome = _LearnCVOutcome(
        summary,
        selection.per_shard_info,
        warnings_list,
        int(pair_indices[0].shape[0]),
    )
    return finalize(outcome)


def _require_deeptica_method(params: Dict[str, Any]) -> None:
    method = str(params.get("method", "deeptica")).lower()
    if method != "deeptica":
        raise RuntimeError(f"LEARN_CV method '{method}' is not supported")


def _prepare_selection_warnings(selection: _LagSelection, lag_value: int) -> List[str]:
    warnings_list = list(selection.warnings)
    if selection.seen_lags and lag_value != int(selection.seen_lags[0]):
        warnings_list.append(f"lag_fallback_used:{lag_value}")
    selection.warnings = warnings_list
    return warnings_list


def _train_deeptica_model(
    params: Dict[str, Any],
    train_deeptica: Any,
    cfg: Any,
    X_list: List[np.ndarray],
    pair_indices: Tuple[np.ndarray, np.ndarray],
    warnings_list: List[str],
    selection: _LagSelection,
):
    load_model = _resolve_model_loader(params, train_deeptica)
    try:
        return load_model(
            X_list,
            pair_indices,
            cfg,
            weights=None,
            model_dir=params.get("model_dir"),
            model_prefix=params.get("model_prefix"),
            train_fn=train_deeptica,
        )
    except Exception as exc:  # pragma: no cover - exercised via tests
        outcome = _make_training_failure_outcome(
            cfg=cfg, selection=selection, warnings=warnings_list, exc=exc
        )
        raise _LearnCVAbort(outcome) from exc


def _transform_features_with_model(model: Any, X_all: np.ndarray) -> np.ndarray:
    try:
        transformed = model.transform(X_all).astype(np.float64, copy=False)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to transform CVs with Deep-TICA model: {exc}"
        ) from exc
    if transformed.ndim != 2 or transformed.shape[0] != X_all.shape[0]:
        raise RuntimeError("Deep-TICA returned invalid transformed features")
    if transformed.shape[1] < 2:
        raise RuntimeError("Deep-TICA produced fewer than two components; expected >=2")
    return transformed


def _store_transformed_features(dataset: Dict[str, Any], Y: np.ndarray) -> int:
    n_out = int(Y.shape[1])
    dataset["X"] = Y
    dataset["cv_names"] = tuple(f"DeepTICA_{i+1}" for i in range(n_out))
    dataset["periodic"] = tuple(False for _ in range(n_out))
    return n_out


def _ensure_training_history(model: Any) -> Dict[str, Any]:
    history = dict(getattr(model, "training_history", {}) or {})
    setattr(model, "training_history", history)
    return history


def _build_success_summary(
    selection: _LagSelection,
    lag_value: int,
    pair_indices: Tuple[np.ndarray, np.ndarray],
    n_out: int,
    metrics: Dict[str, Any],
) -> Dict[str, Any]:
    summary = {
        "applied": True,
        "skipped": False,
        "reason": "ok",
        "method": "deeptica",
        "lag": lag_value,
        "lag_used": lag_value,
        "n_out": n_out,
        "pairs_total": int(pair_indices[0].shape[0]),
        "lag_candidates": [int(v) for v in selection.seen_lags],
        "lag_fallback": [int(v) for v in selection.seen_lags],
        "attempts": selection.attempt_details,
    }
    summary.update(metrics)
    return summary


def _attach_model_artifacts(
    summary: Dict[str, Any], model: Any, params: Dict[str, Any]
) -> None:
    saved_prefix, saved_files = _save_trained_model(
        model, model_dir=params.get("model_dir"), params=params
    )
    if saved_prefix:
        summary["model_prefix"] = saved_prefix
    if saved_files:
        summary["model_files"] = saved_files
        summary["files"] = list(saved_files)


# Pipeline stage adapters
def protein_preparation(context: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """Adapter for protein preparation stage."""
    from ..protein.protein import Protein

    pdb_file = kwargs.get("pdb_file") or context.get("pdb_file")
    if not pdb_file:
        raise ValueError("pdb_file required for protein preparation")

    protein = Protein(pdb_file)
    prepared_pdb = protein.prepare_structure()

    context["protein"] = protein
    context["prepared_pdb"] = prepared_pdb
    logger.info(f"Protein prepared: {prepared_pdb}")
    return context


def system_setup(context: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """Adapter for system setup stage."""
    protein = context.get("protein")
    if not protein:
        raise ValueError("protein required for system setup")

    # System setup logic would go here
    context["system_prepared"] = True
    logger.info("System setup completed")
    return context


def replica_initialization(context: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """Adapter for replica initialization stage."""
    from ..replica_exchange.config import RemdConfig
    from ..replica_exchange.replica_exchange import ReplicaExchange

    prepared_pdb = context.get("prepared_pdb")
    temperatures = kwargs.get("temperatures") or context.get("temperatures", [300.0])
    output_dir = kwargs.get("output_dir") or context.get("output_dir", "output")

    if not prepared_pdb:
        raise ValueError("prepared_pdb required for replica initialization")

    config = RemdConfig(
        input_pdb=str(prepared_pdb),
        temperatures=temperatures,
        output_dir=str(output_dir),
    )

    replica_exchange = ReplicaExchange.from_config(config)
    context["replica_exchange"] = replica_exchange
    context["remd_config"] = config
    logger.info(f"Replica exchange initialized with {len(temperatures)} replicas")
    return context


def energy_minimization(context: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """Adapter for energy minimization stage."""
    replica_exchange = context.get("replica_exchange")
    if not replica_exchange:
        raise ValueError("replica_exchange required for energy minimization")

    # Energy minimization would be handled by replica exchange
    context["energy_minimized"] = True
    logger.info("Energy minimization completed")
    return context


def gradual_heating(context: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """Adapter for gradual heating stage."""
    replica_exchange = context.get("replica_exchange")
    if not replica_exchange:
        raise ValueError("replica_exchange required for gradual heating")

    context["heated"] = True
    logger.info("Gradual heating completed")
    return context


def equilibration(context: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """Adapter for equilibration stage."""
    replica_exchange = context.get("replica_exchange")
    if not replica_exchange:
        raise ValueError("replica_exchange required for equilibration")

    context["equilibrated"] = True
    logger.info("Equilibration completed")
    return context


def production_simulation(context: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """Adapter for production simulation stage."""
    from ..replica_exchange.replica_exchange import run_remd_simulation

    prepared_pdb = context.get("prepared_pdb")
    if not prepared_pdb:
        raise ValueError("prepared_pdb required for production simulation")

    steps = kwargs.get("steps") or context.get("steps", 1000)
    output_dir = kwargs.get("output_dir") or context.get("output_dir", "output")
    temperatures = kwargs.get("temperatures") or context.get("temperatures", [300.0])

    trajectory_file = run_remd_simulation(
        pdb_file=str(prepared_pdb),
        output_dir=str(output_dir),
        total_steps=steps,
        temperatures=temperatures,
    )

    # run_remd_simulation returns a single file path or None
    trajectory_files = [trajectory_file] if trajectory_file else []
    context["trajectory_files"] = trajectory_files
    logger.info(
        f"Production simulation completed, generated {len(trajectory_files)} trajectories"
    )
    return context


def trajectory_demux(context: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """Adapter for trajectory demultiplexing stage."""
    trajectory_files = context.get("trajectory_files", [])
    if not trajectory_files:
        logger.warning("No trajectory files found for demultiplexing")
        return context

    # Demultiplexing logic would go here
    context["demux_completed"] = True
    logger.info("Trajectory demultiplexing completed")
    return context


def trajectory_analysis(context: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """Adapter for trajectory analysis stage."""
    trajectory_files = context.get("trajectory_files", [])
    if not trajectory_files:
        logger.warning("No trajectory files found for analysis")
        return context

    # Analysis logic would go here
    context["analysis_completed"] = True
    logger.info("Trajectory analysis completed")
    return context


def msm_build(context: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """Adapter for MSM building stage."""
    from ..markov_state_model.enhanced_msm import run_complete_msm_analysis

    trajectory_files = context.get("trajectory_files", [])
    if not trajectory_files:
        raise ValueError("trajectory_files required for MSM building")

    n_states = kwargs.get("n_states") or context.get("n_states", 50)
    output_dir = kwargs.get("output_dir") or context.get("output_dir", "output")

    # Get topology file from context
    prepared_pdb = context.get("prepared_pdb")
    if not prepared_pdb:
        raise ValueError("prepared_pdb required for MSM building (used as topology)")

    # Run MSM analysis
    msm_result = run_complete_msm_analysis(
        trajectory_files=trajectory_files,
        topology_file=str(prepared_pdb),
        n_states=n_states,
        output_dir=str(output_dir),
    )

    context["msm_result"] = msm_result
    logger.info(f"MSM built with {n_states} states")
    return context


def build_analysis(context: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """Adapter for build analysis stage."""
    msm_result = context.get("msm_result")
    if not msm_result:
        logger.warning("No MSM result found for build analysis")
        return context

    # Build analysis logic would go here
    context["build_analysis_completed"] = True
    logger.info("Build analysis completed")
    return context


def build_step(context: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """Adapter for BUILD step that delegates to build_result."""
    from .build import AppliedOpts, BuildOpts, build_result
    from .plan import TransformPlan

    # Extract dataset from context
    dataset = context.get("data", context)

    # Create build options from step params and context
    opts_params = kwargs.copy()
    opts = BuildOpts(
        **{k: v for k, v in opts_params.items() if k in BuildOpts.__dataclass_fields__}
    )

    # Create applied options
    applied = AppliedOpts()

    # Create empty transform plan (build_result expects one)
    plan = TransformPlan(steps=())

    # Call build_result
    result = build_result(dataset, opts=opts, plan=plan, applied=applied)

    # Store result in context
    context["build_result"] = result
    context["build_completed"] = True

    logger.info("BUILD step completed")
    return context


def reduce_step(context: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """Adapter for REDUCE step that applies dimensionality reduction."""
    from ..markov_state_model.reduction import reduce_features

    # Extract data from context
    data = context.get("data")
    if data is None:
        logger.warning("No data found for reduction step")
        return context

    # Get reduction parameters
    method = kwargs.get("method", "pca")
    n_components = kwargs.get("n_components", 2)
    lag = kwargs.get("lag", 1)
    scale = kwargs.get("scale", True)

    # Extract feature matrix
    if isinstance(data, dict) and "X" in data:
        X = data["X"]
    elif hasattr(data, "X"):
        X = data.X
    elif isinstance(data, np.ndarray):
        X = data
    else:
        logger.warning("Could not extract feature matrix for reduction")
        return context

    try:
        # Apply reduction
        X_reduced = reduce_features(
            X,
            method=method,
            n_components=n_components,
            lag=lag,
            scale=scale,
            **{
                k: v
                for k, v in kwargs.items()
                if k not in ["method", "n_components", "lag", "scale"]
            },
        )

        # Store reduced data back in context
        if isinstance(data, dict):
            context["data"] = data.copy()
            context["data"]["X"] = X_reduced
            context["data"]["X_original"] = X  # Keep original for reference
        else:
            context["data"] = X_reduced
            context["X_original"] = X

        context["reduction_applied"] = True
        context["reduction_method"] = method
        context["reduction_components"] = n_components

        logger.info(
            f"REDUCE step completed using {method} with {n_components} components"
        )

    except Exception as e:
        logger.error(f"Reduction step failed: {e}")
        # Continue without reduction

    return context


_STEP_HANDLERS: Dict[str, StepHandler] = cast(
    Dict[str, StepHandler],
    {
        "SMOOTH_FES": smooth_fes,
        "LEARN_CV": learn_cv_step,
        "REDUCE": reduce_step,
        "REORDER_STATES": reorder_states,
        "FILL_GAPS": fill_gaps,
        "PROTEIN_PREPARATION": protein_preparation,
        "SYSTEM_SETUP": system_setup,
        "REPLICA_INITIALIZATION": replica_initialization,
        "ENERGY_MINIMIZATION": energy_minimization,
        "GRADUAL_HEATING": gradual_heating,
        "EQUILIBRATION": equilibration,
        "PRODUCTION_SIMULATION": production_simulation,
        "TRAJECTORY_DEMUX": trajectory_demux,
        "TRAJECTORY_ANALYSIS": trajectory_analysis,
        "MSM_BUILD": msm_build,
        "BUILD_ANALYSIS": build_analysis,
        "BUILD": build_step,
    },
)


def apply_transform_plan(dataset, plan: TransformPlan):
    """Apply a transform plan to data or context."""

    context = dataset.copy() if isinstance(dataset, dict) else {"data": dataset}

    for step in plan.steps:
        handler = _STEP_HANDLERS.get(step.name)
        if handler is None:
            logger.warning(f"Unknown transform step: {step.name}")
            continue
        context = handler(context, **step.params)

    if "data" in context and not isinstance(dataset, dict):
        return context["data"]

    return context
