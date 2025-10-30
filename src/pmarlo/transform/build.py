from __future__ import annotations

import base64
import json
import logging
import math
import os
from dataclasses import asdict, dataclass, field, is_dataclass, replace
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)

import numpy as np

from pmarlo import constants as const
from pmarlo.utils.json_io import load_json_file
from pmarlo.utils.path_utils import ensure_directory
from pmarlo.utils.temperature import collect_temperature_values

from ..analysis import compute_diagnostics
from ..analysis.fes import ensure_fes_inputs_whitened
from ..analysis.msm import ensure_msm_inputs_whitened
from ..markov_state_model._msm_utils import build_simple_msm
from .plan import TransformPlan, TransformStep
from .progress import ProgressCB
from .runner import apply_plan as _apply_plan

logger = logging.getLogger("pmarlo")


# --- Deep-TICA helpers ------------------------------------------------------


def _load_or_train_model(
    X_list: Sequence[np.ndarray],
    lagged_pairs: Tuple[np.ndarray, np.ndarray],
    cfg: Any,
    *,
    weights: Optional[np.ndarray] = None,
    model_dir: Optional[str] = None,  # noqa: ARG001 - compatibility shim
    model_prefix: Optional[str] = None,  # noqa: ARG001 - compatibility shim
    train_fn: Optional[Callable[..., Any]] = None,
) -> Any:
    """Return a Deep-TICA model, training one when persistence is unavailable."""

    del model_dir, model_prefix  # retained for forward-compatibility
    if train_fn is None:
        from pmarlo.features.deeptica import train_deeptica as _train

        trainer: Callable[..., Any] = _train
    else:
        trainer = train_fn
    return trainer(X_list, lagged_pairs, cfg, weights=weights)


# --- Shard selection helpers -------------------------------------------------


@lru_cache(maxsize=512)
def _load_shard_metadata_cached(path_str: str) -> Dict[str, Any]:
    try:
        raw = load_json_file(path_str)
        return cast(Dict[str, Any], raw) if isinstance(raw, dict) else {}
    except Exception:
        return {}


def _get_shard_metadata(path: Path) -> Dict[str, Any]:
    return _load_shard_metadata_cached(str(Path(path)))


def _is_demux_shard(path: Path, meta: Optional[Dict[str, Any]] = None) -> bool:
    data = meta if meta is not None else _get_shard_metadata(path)
    if isinstance(data, dict):
        sources: list[Dict[str, Any]] = []
        source = data.get("source")
        if isinstance(source, dict):
            sources.append(source)
        provenance = data.get("provenance")
        if isinstance(provenance, dict):
            sources.append(provenance)
            inner_source = provenance.get("source")
            if isinstance(inner_source, dict):
                sources.append(inner_source)
        for candidate in sources:
            kind = str(candidate.get("kind", "")).lower()
            if kind:
                if kind == "demux":
                    return True
            for key in ("traj", "path", "file", "source_path"):
                raw = candidate.get(key)
                if isinstance(raw, str) and "demux" in raw.lower():
                    return True
    return "demux" in path.stem.lower()


def _coerce_float(value: Any) -> Optional[float]:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(val):
        return None
    return val


def _collect_demux_temperatures(meta: Mapping[str, Any] | None) -> List[float]:
    """Compatibility shim delegating to the shared temperature collector."""

    return list(
        collect_temperature_values(meta, dedupe_tol=const.NUMERIC_PROGRESS_MIN_FRACTION)
    )


def _temperature_matches_target(
    temperatures: Sequence[float], target: float, tolerance: float
) -> bool:
    tol = tolerance if tolerance >= 0.0 else 0.0
    return any(abs(temp - target) <= tol for temp in temperatures)


def _apply_demux_filter(
    shards: Sequence[Path],
    *,
    demux_temperature: Optional[float],
    tolerance: float,
) -> List[Path]:
    filtered: List[Path] = []
    for shard_path in shards:
        meta = _get_shard_metadata(shard_path)
        if not _is_demux_shard(shard_path, meta):
            continue
        if demux_temperature is None:
            filtered.append(shard_path)
            continue
        temps = collect_temperature_values(
            meta, dedupe_tol=const.NUMERIC_PROGRESS_MIN_FRACTION
        )
        if not temps:
            continue
        if _temperature_matches_target(temps, float(demux_temperature), tolerance):
            filtered.append(shard_path)
    return filtered


def _select_mode_subset(
    shards: Sequence[Path], mode: str, max_shards: Optional[int]
) -> List[Path]:
    limit = max_shards or 10
    if mode == "first":
        return list(shards[:limit])
    if mode == "last":
        return list(shards[-limit:])
    if mode == "random":
        import random

        shuffled = list(shards)
        random.shuffle(shuffled)
        return shuffled[:limit]
    if mode == "all":
        return list(shards if max_shards is None else shards[:max_shards])
    raise ValueError(f"Unknown selection mode: {mode}")


def select_shards(
    all_shards: Sequence[Union[str, Path]],
    *,
    mode: str = "demux",
    max_shards: Optional[int] = None,
    sort_key: Optional[Callable[[Union[str, Path]], Any]] = None,
    demux_temperature: Optional[float] = None,
    demux_temperature_tolerance: float = 0.5,
) -> List[Path]:
    shards = [Path(s) for s in all_shards]

    if sort_key is not None:
        shards = sorted(shards, key=sort_key)

    tol = demux_temperature_tolerance if demux_temperature_tolerance >= 0 else 0.0

    if mode == "demux":
        filtered = _apply_demux_filter(
            shards,
            demux_temperature=demux_temperature,
            tolerance=tol,
        )
        if max_shards is not None and len(filtered) > max_shards:
            filtered = filtered[:max_shards]
        return filtered

    return _select_mode_subset(shards, mode, max_shards)


def group_demux_shards_by_temperature(
    shard_paths: Sequence[Union[str, Path]],
    *,
    tolerance: float = 0.5,
) -> Dict[float, List[Path]]:
    tol = tolerance if tolerance >= 0.0 else 0.0
    groups: Dict[float, List[Path]] = {}

    for raw_path in shard_paths:
        shard_path = Path(raw_path)
        meta = _get_shard_metadata(shard_path)
        if not _is_demux_shard(shard_path, meta):
            continue
        temps = collect_temperature_values(
            meta, dedupe_tol=const.NUMERIC_PROGRESS_MIN_FRACTION
        )
        if not temps:
            continue
        temperature = temps[0]

        key = None
        for existing in groups:
            if abs(existing - temperature) <= tol:
                key = existing
                break
        if key is None:
            key = float(round(temperature, 3))
            groups[key] = []
        groups[key].append(shard_path)

    return groups


def _serialize_array_payload(arr: Optional[np.ndarray]) -> Optional[Dict[str, Any]]:
    if arr is None:
        return None
    return {
        "dtype": str(arr.dtype),
        "shape": list(arr.shape),
        "data": base64.b64encode(arr.tobytes()).decode("ascii"),
    }


def _serialize_generic_payload(obj: Any) -> Any:
    if obj is None:
        return None
    if hasattr(obj, "to_dict"):
        return obj.to_dict()  # type: ignore[call-arg]
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return _sanitize_artifacts(obj)


def _serialize_fes_payload(obj: Any) -> Any:
    from ..markov_state_model.free_energy import FESResult

    if obj is None:
        return None
    if isinstance(obj, FESResult):
        return {
            "F": _serialize_array_payload(obj.F),
            "xedges": _serialize_array_payload(obj.xedges),
            "yedges": _serialize_array_payload(obj.yedges),
            "levels_kJmol": _serialize_array_payload(obj.levels_kJmol),
            "metadata": _sanitize_artifacts(obj.metadata),
        }
    if isinstance(obj, dict):
        return obj
    return _serialize_generic_payload(obj)


def _serialize_applied_opts(
    applied: Optional[AppliedOpts],
) -> Optional[Dict[str, Any]]:
    if applied is None:
        return None
    payload = asdict(applied)
    shards = payload.get("selected_shards")
    if isinstance(shards, list):
        payload["selected_shards"] = [str(s) for s in shards]
    return payload


# --- Configuration classes ---------------------------------------------------


@dataclass(frozen=True)
class BuildOpts:
    plan: Optional[TransformPlan] = None
    shard_selection_mode: str = "demux"
    max_shards: Optional[int] = None
    seed: Optional[int] = None
    temperature: float = 300.0
    lag_candidates: Optional[Tuple[int, ...]] = None
    count_mode: str = "sliding"
    n_clusters: int = 200
    n_states: int = 50
    lag_time: int = 10
    msm_mode: str = "kmeans+msm"
    enable_fes: bool = True
    fes_temperature: float = 300.0
    fes_bins: Optional[Tuple[int, int]] = None
    fes_smoothing_mode: str = "never"
    fes_target_sd_kT: Optional[float] = None
    fes_alpha: float = 1e-6
    fes_h0: float = 1.2
    fes_ess_ref: float = 50.0
    fes_h_min: float = 0.4
    fes_h_max: float = 3.0
    enable_tram: bool = False
    tram_lag: int = 1
    tram_n_iter: int = 100
    output_format: str = "json"
    save_trajectories: bool = False
    save_plots: bool = True
    n_jobs: int = 1
    memory_limit_gb: Optional[float] = None
    chunk_size: int = 1000
    debug: bool = False
    verbose: bool = False
    kmeans_kwargs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.lag_candidates is not None:
            object.__setattr__(
                self, "lag_candidates", tuple(int(x) for x in self.lag_candidates)
            )
        if self.temperature is not None:
            object.__setattr__(self, "fes_temperature", float(self.temperature))
        if self.fes_bins is not None:
            if not isinstance(self.fes_bins, tuple) or len(self.fes_bins) != 2:
                raise ValueError("fes_bins must be a tuple of two integers")
            if any(not isinstance(b, int) or b <= 0 for b in self.fes_bins):
                raise ValueError("fes_bins must contain two positive integers")
            object.__setattr__(
                self, "fes_bins", tuple(int(x) for x in self.fes_bins)
            )

        mode = str(self.fes_smoothing_mode).lower()
        if mode not in {"never", "auto", "always"}:
            raise ValueError(
                "fes_smoothing_mode must be one of {'never', 'auto', 'always'}"
            )
        object.__setattr__(self, "fes_smoothing_mode", mode)

        if self.fes_target_sd_kT is not None:
            target = float(self.fes_target_sd_kT)
            if target <= 0:
                raise ValueError("fes_target_sd_kT must be positive when provided")
            object.__setattr__(self, "fes_target_sd_kT", target)

        def _coerce_positive(name: str) -> float:
            value = float(getattr(self, name))
            if value <= 0:
                raise ValueError(f"{name} must be positive")
            return value

        alpha = _coerce_positive("fes_alpha")
        h0 = _coerce_positive("fes_h0")
        ess_ref = _coerce_positive("fes_ess_ref")
        h_min = _coerce_positive("fes_h_min")
        h_max = _coerce_positive("fes_h_max")
        if h_min > h_max:
            raise ValueError("fes_h_min must be <= fes_h_max")

        object.__setattr__(self, "fes_alpha", alpha)
        object.__setattr__(self, "fes_h0", h0)
        object.__setattr__(self, "fes_ess_ref", ess_ref)
        object.__setattr__(self, "fes_h_min", h_min)
        object.__setattr__(self, "fes_h_max", h_max)

    def with_plan(self, plan: TransformPlan) -> "BuildOpts":
        return replace(self, plan=plan)

    def with_shards(
        self, mode: str = "demux", max_shards: Optional[int] = None
    ) -> "BuildOpts":
        return replace(self, shard_selection_mode=mode, max_shards=max_shards)

    def with_msm(
        self, n_clusters: int = 200, n_states: int = 50, lag_time: int = 10
    ) -> "BuildOpts":
        return replace(
            self, n_clusters=n_clusters, n_states=n_states, lag_time=lag_time
        )


@dataclass
class AppliedOpts:
    bins: Optional[Dict[str, int]] = None
    lag: Optional[int] = None
    macrostates: Optional[int] = None
    notes: Dict[str, Any] = field(default_factory=dict)
    original_opts: Optional[BuildOpts] = None
    selected_shards: List[Path] = field(default_factory=list)
    actual_plan: Optional[TransformPlan] = None
    effective_n_jobs: int = 1
    effective_memory_limit: Optional[float] = None
    start_time: Optional[str] = None
    hostname: Optional[str] = None
    git_commit: Optional[str] = None

    @classmethod
    def from_opts(
        cls,
        opts: BuildOpts,
        selected_shards: List[Path],
        plan: Optional[TransformPlan] = None,
    ) -> "AppliedOpts":
        import socket
        from datetime import datetime

        now = datetime.now().isoformat()
        return cls(
            original_opts=opts,
            selected_shards=list(selected_shards),
            actual_plan=plan or opts.plan,
            effective_n_jobs=opts.n_jobs,
            effective_memory_limit=opts.memory_limit_gb,
            start_time=now,
            hostname=socket.gethostname(),
        )


@dataclass
class RunMetadata:
    run_id: str
    start_time: str
    end_time: Optional[str] = None
    duration_seconds: Optional[float] = None
    hostname: Optional[str] = None
    git_commit: Optional[str] = None
    python_version: Optional[str] = None
    pmarlo_version: Optional[str] = None
    success: bool = False
    error_message: Optional[str] = None
    transform_plan: Optional[Tuple[TransformStep, ...]] = None
    applied_opts: Optional[AppliedOpts] = None
    fes: Optional[Dict[str, Any]] = None
    dataset_hash: Optional[str] = None
    digest: Optional[str] = None
    seed: Optional[int] = None
    temperature: Optional[float] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunMetadata":
        payload = dict(data)
        if payload.get("applied_opts") is not None:
            payload["applied_opts"] = AppliedOpts(**payload["applied_opts"])
        if payload.get("transform_plan") is not None:
            steps: List[TransformStep] = []
            for step in payload["transform_plan"]:
                if isinstance(step, TransformStep):
                    steps.append(step)
                else:
                    steps.append(TransformStep(**step))
            payload["transform_plan"] = tuple(steps)
        return cls(
            **{k: v for k, v in payload.items() if k in cls.__dataclass_fields__}
        )

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        if self.transform_plan is not None:
            out["transform_plan"] = [asdict(step) for step in self.transform_plan]
        if self.applied_opts is not None:
            applied = asdict(self.applied_opts)
            shards = applied.get("selected_shards")
            if isinstance(shards, list):
                applied["selected_shards"] = [str(s) for s in shards]
            out["applied_opts"] = applied
        return out


@dataclass
class BuildResult:
    transition_matrix: Optional[np.ndarray] = None
    stationary_distribution: Optional[np.ndarray] = None
    msm: Optional[Any] = None
    fes: Optional[Any] = None
    tram: Optional[Any] = None
    metadata: Optional[RunMetadata] = None
    applied_opts: Optional[AppliedOpts] = None
    n_frames: int = 0
    n_shards: int = 0
    feature_names: List[str] = field(default_factory=list)
    cluster_populations: Optional[np.ndarray] = None
    artifacts: Dict[str, Any] = field(default_factory=dict)
    messages: List[str] = field(default_factory=list)
    flags: Dict[str, Any] = field(default_factory=dict)
    diagnostics: Optional[Dict[str, Any]] = None

    def to_json(self) -> str:
        applied_dict = _serialize_applied_opts(self.applied_opts)
        data = {
            "transition_matrix": _serialize_array_payload(self.transition_matrix),
            "stationary_distribution": _serialize_array_payload(
                self.stationary_distribution
            ),
            "msm": _serialize_generic_payload(self.msm),
            "fes": _serialize_fes_payload(self.fes),
            "tram": _serialize_generic_payload(self.tram),
            "metadata": self.metadata.to_dict() if self.metadata else None,
            "applied_opts": applied_dict,
            "n_frames": self.n_frames,
            "n_shards": self.n_shards,
            "feature_names": self.feature_names,
            "cluster_populations": _serialize_array_payload(self.cluster_populations),
            "artifacts": _sanitize_artifacts(self.artifacts),
            "messages": list(self.messages),
            "flags": _sanitize_artifacts(dict(self.flags)),
            "diagnostics": _sanitize_artifacts(self.diagnostics),
        }

        return json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False)

    @classmethod
    def from_json(cls, text: str) -> "BuildResult":
        from ..markov_state_model.free_energy import FESResult

        data = json.loads(text)
        metadata = (
            RunMetadata.from_dict(data["metadata"])
            if data.get("metadata") is not None
            else None
        )

        def _decode_array(obj: Optional[Dict[str, Any]]) -> Optional[np.ndarray]:
            if obj is None:
                return None
            dtype = np.dtype(obj["dtype"])
            shape = tuple(obj["shape"])
            data_bytes = base64.b64decode(obj["data"])
            return np.frombuffer(data_bytes, dtype=dtype).reshape(shape)

        def _decode_fes(obj: Optional[Any]) -> Optional[Any]:
            if obj is None:
                return None
            if isinstance(obj, dict) and {"F", "xedges", "yedges"} <= obj.keys():
                try:
                    return FESResult(
                        F=_decode_array(obj.get("F")),
                        xedges=_decode_array(obj.get("xedges")),
                        yedges=_decode_array(obj.get("yedges")),
                        levels_kJmol=_decode_array(obj.get("levels_kJmol")),
                        metadata=obj.get("metadata", {}),
                    )
                except Exception:
                    return obj
            return obj

        applied_dict = data.get("applied_opts") or None
        applied_obj = (
            AppliedOpts(**applied_dict) if isinstance(applied_dict, dict) else None
        )

        return cls(
            transition_matrix=_decode_array(data.get("transition_matrix")),
            stationary_distribution=_decode_array(data.get("stationary_distribution")),
            msm=data.get("msm"),
            fes=_decode_fes(data.get("fes")),
            tram=data.get("tram"),
            metadata=metadata,
            applied_opts=applied_obj,
            n_frames=data.get("n_frames", 0),
            n_shards=data.get("n_shards", 0),
            feature_names=data.get("feature_names", []),
            cluster_populations=_decode_array(data.get("cluster_populations")),
            artifacts=data.get("artifacts", {}),
            messages=list(data.get("messages", [])),
            flags=data.get("flags", {}),
            diagnostics=data.get("diagnostics"),
        )


# --- Build functions ---------------------------------------------------------


def _prepare_applied_state(
    opts: BuildOpts,
    plan: Optional[TransformPlan],
    applied: Optional[AppliedOpts],
) -> Tuple[Optional[TransformPlan], AppliedOpts]:
    plan_to_use = plan or opts.plan
    if applied is None:
        applied_obj = AppliedOpts.from_opts(opts, [], plan=plan_to_use)
    else:
        applied_obj = applied
        if applied_obj.original_opts is None:
            applied_obj.original_opts = opts
        if applied_obj.actual_plan is None and plan_to_use is not None:
            applied_obj.actual_plan = plan_to_use
    return plan_to_use, applied_obj


def _inject_model_dir(
    plan: Optional[TransformPlan], applied: AppliedOpts
) -> Optional[TransformPlan]:
    if plan is None or not isinstance(applied.notes, dict):
        return plan
    raw_model_dir = applied.notes.get("model_dir")
    model_dir_str: Optional[str] = None
    if raw_model_dir:
        try:
            model_dir_str = str(raw_model_dir)
        except Exception:
            model_dir_str = None
    if not model_dir_str:
        return plan
    updated_steps = []
    plan_changed = False
    for step in plan.steps:
        if step.name == "LEARN_CV" and "model_dir" not in step.params:
            params = dict(step.params)
            params["model_dir"] = model_dir_str
            updated_steps.append(TransformStep(step.name, params))
            plan_changed = True
        else:
            updated_steps.append(step)
    return TransformPlan(steps=tuple(updated_steps)) if plan_changed else plan


def _create_metadata(
    opts: BuildOpts, plan: Optional[TransformPlan], applied: AppliedOpts
) -> Tuple[RunMetadata, datetime]:
    import platform
    import socket
    from datetime import datetime as _dt

    start_dt = _dt.now()
    metadata = RunMetadata(
        run_id=_generate_run_id(),
        start_time=start_dt.isoformat(),
        hostname=socket.gethostname(),
        transform_plan=tuple(plan.steps) if plan else None,
        applied_opts=applied,
        seed=opts.seed,
        temperature=opts.temperature,
        python_version=platform.python_version(),
    )
    return metadata, start_dt


def _maybe_apply_plan(
    plan: Optional[TransformPlan],
    dataset: Any,
    applied: AppliedOpts,
    progress_callback: Optional[ProgressCB],
) -> Any:
    if plan is None:
        return dataset
    logger.info("Applying transform plan with %d steps", len(plan.steps))
    result = _apply_plan(plan, dataset, progress_callback=progress_callback)
    applied.actual_plan = plan
    return result


def _extract_artifacts_dict(dataset: Any) -> Dict[str, Any]:
    if isinstance(dataset, dict):
        raw = dataset.get("__artifacts__")
        if isinstance(raw, dict):
            sanitized = _sanitize_artifacts(raw)
            if isinstance(sanitized, dict):
                return cast(Dict[str, Any], sanitized)
            return {}
    return {}


def _record_deeptica_notes(artifacts: Dict[str, Any], applied: AppliedOpts) -> None:
    try:
        if "mlcv_deeptica" not in artifacts:
            return
        note = {"method": "deeptica"}
        summary = artifacts.get("mlcv_deeptica", {})
        if isinstance(summary, dict):
            for key in ("lag", "n_out", "pairs_total", "model_prefix"):
                if key in summary and summary[key] is not None:
                    note[key] = summary[key]
        if applied.notes is None:
            applied.notes = {}
        applied.notes["mlcv"] = note
    except Exception:
        pass


def _record_cv_bins(working_dataset: Any, applied: AppliedOpts) -> None:
    try:
        matrix = _extract_feature_matrix(working_dataset)
        if matrix is None:
            return
        cv1, cv2 = matrix[:, 0], matrix[:, 1]
        n1, n2 = _resolve_bin_counts(applied)
        edges = {
            "cv1": _compute_bin_edges(cv1, n1),
            "cv2": _compute_bin_edges(cv2, n2),
        }
        if applied.notes is None:
            applied.notes = {}
        applied.notes["cv_bin_edges"] = edges
    except Exception:
        pass


def _extract_feature_matrix(working_dataset: Any) -> Optional[np.ndarray]:
    if not isinstance(working_dataset, dict):
        return None
    raw = working_dataset.get("X")
    matrix = np.asarray(raw, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] < 2 or matrix.shape[0] <= 0:
        return None
    return matrix


def _resolve_bin_counts(applied: AppliedOpts) -> Tuple[int, int]:
    default = (32, 32)
    if not isinstance(applied.bins, dict):
        return default
    try:
        n1 = int(applied.bins.get("cv1", default[0]))
        n2 = int(applied.bins.get("cv2", default[1]))
        return max(2, n1), max(2, n2)
    except Exception:
        return default


def _compute_bin_edges(values: np.ndarray, count: int) -> List[float]:
    finite_min = float(np.nanmin(values))
    finite_max = float(np.nanmax(values))
    if (
        not np.isfinite(finite_min)
        or not np.isfinite(finite_max)
        or finite_max <= finite_min
    ):
        finite_min, finite_max = -1.0, 1.0
    bins = int(max(2, count)) + 1
    edges = np.linspace(finite_min, finite_max, bins, dtype=float)
    return [float(x) for x in edges]


def _build_msm_payload(
    working_dataset: Any, opts: BuildOpts, applied: AppliedOpts
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[Any]]:
    if opts.msm_mode == "none":
        return None, None, None
    logger.info("Building MSM...")
    msm_result = _build_msm(working_dataset, opts, applied)
    if isinstance(msm_result, tuple):
        if len(msm_result) == 2:
            # Old format: (T, pi)
            return msm_result[0], msm_result[1], None
        elif len(msm_result) == 3:
            # New format: (T, pi, msm_data)
            return msm_result[0], msm_result[1], msm_result[2]
    return None, None, msm_result


def _build_fes_payload(
    working_dataset: Any, opts: BuildOpts, applied: AppliedOpts, metadata: RunMetadata
) -> Optional[Any]:
    if not opts.enable_fes:
        return None
    logger.info("Building FES...")
    fes_raw = _build_fes(working_dataset, opts, applied)
    if isinstance(fes_raw, dict) and "result" in fes_raw:
        result = fes_raw.get("result")
        names = _extract_fes_names(fes_raw)
        bins_tuple = _derive_fes_bins(applied, names)
        metadata.fes = {
            "bins": bins_tuple,
            "names": names,
            "temperature": opts.temperature,
        }
        return result
    metadata.fes = None
    if isinstance(fes_raw, dict) and fes_raw.get("skipped"):
        return None
    return fes_raw


def _extract_fes_names(fes_payload: dict[str, Any]) -> Tuple[str, ...]:
    raw = (fes_payload.get("cv1_name"), fes_payload.get("cv2_name"))
    return tuple(name for name in raw if isinstance(name, str) and name)


def _derive_fes_bins(
    applied: AppliedOpts, names: Tuple[str, ...]
) -> Optional[Tuple[int, ...]]:
    if not names or not isinstance(applied.bins, dict):
        return None
    direct = _lookup_named_bins(applied.bins, names)
    if direct is not None:
        return direct
    ordered = [int(v) for v in applied.bins.values() if _is_positive_int(v)]
    if len(ordered) >= len(names):
        return tuple(ordered[: len(names)])
    return None


def _lookup_named_bins(
    bins_cfg: dict[str, Any], names: Tuple[str, ...]
) -> Optional[Tuple[int, ...]]:
    candidate: List[int] = []
    for name in names:
        key = str(name)
        raw_value = bins_cfg.get(key)
        if raw_value is None:
            raw_value = bins_cfg.get(key.lower())
        if raw_value is None or not _is_positive_int(raw_value):
            return None
        candidate.append(int(raw_value))
    return tuple(candidate)


def _is_positive_int(value: Any) -> bool:
    try:
        return int(value) > 0
    except Exception:
        return False


def _build_tram_payload(
    working_dataset: Any, opts: BuildOpts, applied: AppliedOpts
) -> Optional[Any]:
    if not opts.enable_tram:
        return None
    logger.info("Building TRAM...")
    return _build_tram(working_dataset, opts, applied)


def _update_metadata_end(metadata: RunMetadata, start_dt: datetime) -> None:
    from datetime import datetime as _dt

    end_dt = _dt.now()
    metadata.end_time = end_dt.isoformat()
    metadata.duration_seconds = (end_dt - start_dt).total_seconds()
    metadata.success = True


def _resolve_shard_counts(applied: AppliedOpts, dataset: Any) -> int:
    if applied.selected_shards:
        return len(applied.selected_shards)
    if isinstance(dataset, dict):
        shards_meta = dataset.get("__shards__")
        if isinstance(shards_meta, list):
            return len(shards_meta)
    return 0


def _ensure_selected_shards(applied: AppliedOpts, dataset: Any) -> None:
    if applied.selected_shards:
        return
    if isinstance(dataset, dict):
        shards_meta = dataset.get("__shards__")
        if isinstance(shards_meta, list):
            try:
                applied.selected_shards = [
                    Path(str(item.get("id", ""))) for item in shards_meta
                ]
            except Exception:
                applied.selected_shards = []


def _attach_fes_quality(artifacts: Dict[str, Any], fes_payload: Any) -> Dict[str, Any]:
    try:
        from ..markov_state_model.free_energy import FESResult

        if isinstance(fes_payload, FESResult):
            quality = _extract_fes_quality_artifact(fes_payload)
            if quality:
                enriched = dict(artifacts)
                enriched["fes_quality"] = _sanitize_artifacts(quality)
                return enriched
    except Exception:
        logger.debug("Failed to derive FES quality artifact", exc_info=True)
    return artifacts


def _collect_flags(
    transition_matrix: Optional[np.ndarray],
    fes_payload: Optional[Any],
    tram_payload: Optional[Any],
    artifacts: Dict[str, Any],
) -> Dict[str, Any]:
    flags: Dict[str, Any] = {}
    if transition_matrix is not None and transition_matrix.size > 0:
        flags["has_msm"] = True
    if fes_payload is not None:
        flags["has_fes"] = True
    if tram_payload not in (None, {}):
        flags["has_tram"] = True
    if "mlcv_deeptica" in artifacts:
        summary = artifacts["mlcv_deeptica"]
        if isinstance(summary, dict):
            flags["mlcv_deeptica_applied"] = bool(summary.get("applied"))
    return flags


def _compute_diagnostics_safe(
    dataset: Any, transition_matrix: Optional[np.ndarray]
) -> Optional[Dict[str, Any]]:
    try:
        diag_mass_val = None
        if transition_matrix is not None and transition_matrix.size > 0:
            diag_mass_val = float(
                np.trace(transition_matrix) / transition_matrix.shape[0]
            )
        diagnostics = compute_diagnostics(dataset, diag_mass=diag_mass_val)
        return diagnostics
    except Exception:
        logger.debug("Failed to compute diagnostics", exc_info=True)
        return None


def build_result(
    dataset: Any,
    opts: Optional[BuildOpts] = None,
    plan: Optional[TransformPlan] = None,
    applied: Optional[AppliedOpts] = None,
    *,
    progress_callback: Optional[ProgressCB] = None,
) -> BuildResult:
    if opts is None:
        opts = BuildOpts()

    plan_to_use, applied_obj = _prepare_applied_state(opts, plan, applied)
    plan_to_use = _inject_model_dir(plan_to_use, applied_obj)
    metadata, start_dt = _create_metadata(opts, plan_to_use, applied_obj)

    try:
        working_dataset = _maybe_apply_plan(
            plan_to_use, dataset, applied_obj, progress_callback
        )

        artifacts = _extract_artifacts_dict(working_dataset)
        _record_deeptica_notes(artifacts, applied_obj)
        _record_cv_bins(working_dataset, applied_obj)

        transition_matrix, stationary_distribution, msm_payload = _build_msm_payload(
            working_dataset, opts, applied_obj
        )
        fes_payload = _build_fes_payload(working_dataset, opts, applied_obj, metadata)
        tram_payload = _build_tram_payload(working_dataset, opts, applied_obj)

        _update_metadata_end(metadata, start_dt)

        n_frames = _count_frames(working_dataset)
        feature_names = _extract_feature_names(working_dataset)
        _ensure_selected_shards(applied_obj, dataset)
        n_shards = _resolve_shard_counts(applied_obj, dataset)

        artifacts = _attach_fes_quality(artifacts, fes_payload)
        flags = _collect_flags(transition_matrix, fes_payload, tram_payload, artifacts)
        diagnostics = _compute_diagnostics_safe(working_dataset, transition_matrix)
        if diagnostics and diagnostics.get("warnings"):
            flags.setdefault("diagnostic_warnings", diagnostics["warnings"])

        return BuildResult(
            transition_matrix=transition_matrix,
            stationary_distribution=stationary_distribution,
            msm=msm_payload,
            fes=fes_payload,
            tram=tram_payload,
            metadata=metadata,
            applied_opts=applied_obj,
            n_frames=n_frames,
            n_shards=n_shards,
            feature_names=feature_names,
            artifacts=artifacts,
            flags=flags,
            diagnostics=diagnostics,
        )

    except Exception as exc:
        logger.error("Build failed: %s", exc)
        metadata.error_message = str(exc)
        metadata.success = False
        raise


def _generate_run_id() -> str:
    import time

    return f"build_{int(time.time())}_{os.getpid()}"


def _safe_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except Exception:
        return None


def _frames_from_len(dataset: Any) -> Optional[int]:
    if not hasattr(dataset, "__len__"):
        return None
    n_frames_attr = getattr(dataset, "n_frames", None)
    if n_frames_attr is not None:
        value = _safe_int(n_frames_attr)
        if value is not None:
            return value
    try:
        return _safe_int(len(dataset))
    except Exception:
        return None


def _frames_from_attr(dataset: Any) -> Optional[int]:
    if hasattr(dataset, "n_frames"):
        return _safe_int(getattr(dataset, "n_frames"))
    return None


def _frames_from_mapping(dataset: Any) -> Optional[int]:
    if isinstance(dataset, dict) and "X" in dataset:
        try:
            return _safe_int(np.asarray(dataset["X"]).shape[0])
        except Exception:
            return None
    return None


def _count_frames(dataset: Any) -> int:
    for extractor in (_frames_from_len, _frames_from_attr, _frames_from_mapping):
        value = extractor(dataset)
        if value is not None:
            return value
    return 0


def _extract_feature_names(dataset: Any) -> List[str]:
    try:
        if hasattr(dataset, "feature_names"):
            return list(dataset.feature_names)
        if hasattr(dataset, "columns"):
            return list(dataset.columns)
        if isinstance(dataset, dict) and "cv_names" in dataset:
            return [str(x) for x in dataset.get("cv_names", [])]
        return []
    except Exception:
        return []


def _extract_cvs(
    dataset: Any,
) -> Optional[Tuple[np.ndarray, np.ndarray, Tuple[str, str], Tuple[bool, bool]]]:
    try:
        if isinstance(dataset, dict):
            X = dataset.get("X")
            if X is None:
                return None
            X = np.asarray(X, dtype=np.float64)
            if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] < 2:
                return None
            names = dataset.get("cv_names") or ()
            if not isinstance(names, (list, tuple)):
                names = ()
            name_pair = (
                str(names[0]) if len(names) > 0 else "cv1",
                str(names[1]) if len(names) > 1 else "cv2",
            )
            periodic = dataset.get("periodic") or ()
            if not isinstance(periodic, (list, tuple)):
                periodic = ()
            periodic_pair = (
                bool(periodic[0]) if len(periodic) > 0 else False,
                bool(periodic[1]) if len(periodic) > 1 else False,
            )
            return X[:, 0], X[:, 1], name_pair, periodic_pair
        if hasattr(dataset, "X"):
            X = np.asarray(getattr(dataset, "X"), dtype=np.float64)
            if X.ndim == 2 and X.shape[1] >= 2:
                return X[:, 0], X[:, 1], ("cv1", "cv2"), (False, False)
    except Exception:
        return None
    return None


def _ensure_msm_whitened(dataset: Any) -> None:
    if not isinstance(dataset, dict):
        return
    try:
        ensure_msm_inputs_whitened(dataset)
    except Exception:
        logger.debug("Failed to apply CV whitening before MSM build", exc_info=True)


def _infer_cv_issue(cv_name: str, value: float) -> str:
    name = (cv_name or "").lower()
    if np.isnan(value):
        root = "Value became NaN during preprocessing"
    else:
        root = "Value reached an infinite magnitude"
    if "rmsd" in name:
        return f"{root}; check reference alignment and atom selections."
    if name in {"rg", "radius_of_gyration"}:
        return f"{root}; verify that the structure has non-zero extent and no missing atoms."
    return f"{root}; inspect upstream CV computation."


def _locate_shard(
    shards: Sequence[dict], frame_idx: int
) -> tuple[dict | None, int | None]:
    for shard in shards:
        try:
            start = int(shard.get("start", 0))
            stop = int(shard.get("stop", start))
        except Exception:
            continue
        if start <= frame_idx < stop:
            return shard, frame_idx - start
    return None, None


def _resolve_diagnostics_dir(
    dataset: Mapping[str, Any], opts: BuildOpts
) -> Path | None:
    candidates: list[Path] = []
    custom_dir = getattr(opts, "diagnostics_dir", None)
    if custom_dir:
        try:
            candidates.append(Path(custom_dir))
        except Exception:
            pass
    shards = dataset.get("__shards__") if isinstance(dataset, Mapping) else None
    if isinstance(shards, Sequence) and shards:
        first = shards[0]
        if isinstance(first, Mapping):
            source_path = first.get("source_path")
            if source_path:
                try:
                    candidates.append(Path(source_path).parent / "diagnostics")
                except Exception:
                    pass
    candidates.append(Path.cwd() / "pmarlo_diagnostics")
    for candidate in candidates:
        try:
            ensure_directory(candidate)
            return candidate
        except Exception:
            continue
    return None


def _prepare_cv_names(dataset: Mapping[str, Any], X: np.ndarray) -> list[str]:
    names = dataset.get("cv_names") if isinstance(dataset, Mapping) else None
    if not isinstance(names, (list, tuple)):
        names = tuple()
    resolved = [
        str(n) if isinstance(n, str) else f"cv{idx}"
        for idx, n in enumerate(names or [])
    ]
    if len(resolved) < X.shape[1]:
        resolved.extend(f"cv{idx}" for idx in range(len(resolved), X.shape[1]))
    return resolved


def _report_non_finite_cv_values(
    dataset: Mapping[str, Any], names: Sequence[str], X: np.ndarray
) -> None:
    invalid_mask = ~np.isfinite(X)
    if not invalid_mask.any():
        return
    counts = invalid_mask.sum(axis=0)
    for idx, count in enumerate(counts):
        if count:
            logger.error(
                "Detected %d non-finite values in CV '%s'",
                int(count),
                names[idx] if idx < len(names) else f"cv{idx}",
            )
    shards_meta = dataset.get("__shards__") if isinstance(dataset, Mapping) else None
    shards_seq = shards_meta if isinstance(shards_meta, Sequence) else ()
    bad_locations = np.argwhere(invalid_mask)
    for frame_idx, cv_idx in bad_locations[:20]:
        cv_name = names[cv_idx] if cv_idx < len(names) else f"cv{cv_idx}"
        value = X[frame_idx, cv_idx]
        shard_info, local_idx = _locate_shard(shards_seq, int(frame_idx))
        shard_id = (
            shard_info.get("id") if isinstance(shard_info, Mapping) else "unknown"
        )
        traj_path = (
            shard_info.get("source_path") if isinstance(shard_info, Mapping) else None
        )
        cause = _infer_cv_issue(cv_name, value)
        logger.error(
            "Invalid %s detected for CV '%s' at global frame %d "
            "(shard=%s, local_frame=%s, traj=%s). %s",
            "NaN" if np.isnan(value) else "Infinity",
            cv_name,
            int(frame_idx),
            shard_id,
            "n/a" if local_idx is None else int(local_idx),
            traj_path,
            cause,
        )


def _log_cv_statistics(names: Sequence[str], X: np.ndarray) -> None:
    for idx in range(X.shape[1]):
        col = X[:, idx]
        finite = col[np.isfinite(col)]
        cv_name = names[idx] if idx < len(names) else f"cv{idx}"
        if finite.size == 0:
            logger.warning(
                "CV '%s' contains no finite values across %d frames",
                cv_name,
                X.shape[0],
            )
            continue
        logger.info(
            "CV '%s' stats over %d finite frames: mean=%.6f std=%.6f "
            "min=%.6f max=%.6f",
            cv_name,
            int(finite.size),
            float(np.mean(finite)),
            float(np.std(finite)),
            float(np.min(finite)),
            float(np.max(finite)),
        )


def _save_cv_distribution_plot(
    dataset: Mapping[str, Any], opts: BuildOpts, names: Sequence[str], X: np.ndarray
) -> None:
    if X.shape[1] < 2:
        return
    finite_rows = np.all(np.isfinite(X[:, :2]), axis=1)
    finite_vals = X[finite_rows, :2]
    if not finite_vals.size:
        logger.warning(
            "Skipping CV distribution plot; insufficient finite values in first two CVs"
        )
        return
    sample = finite_vals
    max_points = 100000
    if sample.shape[0] > max_points:
        rng = np.random.default_rng(42)
        idxs = rng.choice(sample.shape[0], size=max_points, replace=False)
        sample = sample[idxs]
    diagnostics_dir = _resolve_diagnostics_dir(dataset, opts)
    if diagnostics_dir is None:
        logger.warning("Unable to determine diagnostics directory for CV plot")
        return
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 5))
        hb = ax.hist2d(
            sample[:, 0],
            sample[:, 1],
            bins=64,
            cmap="viridis",
        )
        fig.colorbar(hb[3], ax=ax, label="count")
        ax.set_xlabel(names[0] if names else "cv0")
        ax.set_ylabel(names[1] if len(names) > 1 else "cv1")
        ax.set_title("CV Distribution (finite values)")
        fig.tight_layout()
        out_path = diagnostics_dir / "cv_distribution.png"
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        logger.info("Saved CV distribution plot to '%s'", out_path)
    except Exception as exc:
        logger.warning("Failed to save CV distribution plot: %s", exc)


def _diagnose_cv_matrix(
    dataset: Mapping[str, Any], opts: BuildOpts, matrix: np.ndarray
) -> None:
    try:
        X = np.asarray(matrix, dtype=np.float64)
        if X.ndim != 2 or X.size == 0:
            return
        names = _prepare_cv_names(dataset, X)
        _report_non_finite_cv_values(dataset, names, X)
        _log_cv_statistics(names, X)
        _save_cv_distribution_plot(dataset, opts, names, X)
    except Exception:
        logger.debug("CV diagnostics failed", exc_info=True)


def _cluster_continuous_trajectories(
    dataset: Dict[str, Any], opts: BuildOpts
) -> Optional[List[np.ndarray]]:
    if "X" not in dataset:
        return None
    X = dataset["X"]
    if not isinstance(X, np.ndarray) or X.size == 0:
        logger.warning("No continuous CV data available for clustering")
        return None
    logger.info(
        "No discrete trajectories found, clustering continuous CV data for MSM..."
    )
    n_samples = int(X.shape[0])
    requested_clusters = int(max(1, opts.n_clusters))
    if n_samples < requested_clusters:
        logger.warning(
            "Insufficient continuous samples (%d) for %d requested clusters; "
            "skipping MSM clustering",
            n_samples,
            requested_clusters,
        )
        return None
    _diagnose_cv_matrix(dataset, opts, X)
    from ..markov_state_model.clustering import cluster_microstates

    clustering = cluster_microstates(
        X,
        n_states=opts.n_states,
        method="kmeans",
        random_state=opts.seed,
        **(opts.kmeans_kwargs or {}),
    )
    labels = clustering.labels
    if labels is None or labels.size == 0:
        logger.warning("Clustering failed to produce labels")
        return None

    shards_info = dataset.get("__shards__", [])
    if shards_info:
        dtrajs: List[np.ndarray] = []
        for idx, shard_info in enumerate(shards_info):
            start = int(shard_info.get("start", 0))
            stop = int(shard_info.get("stop", start))
            if stop > start:
                shard_labels = labels[start:stop]
                dtraj_array = shard_labels.astype(np.int32)
                dtrajs.append(dtraj_array)

                # Log shape after clustering/featurization for this shard
                logger.info(f"Featurized shard {idx}: features shape={dtraj_array.shape}")

        logger.info("Created %d discrete trajectories from clustering", len(dtrajs))
        return dtrajs
    logger.info("Created 1 discrete trajectory from clustering")
    return [labels.astype(np.int32)]


def _prepare_dtrajs(dataset: Any, opts: BuildOpts) -> Optional[List[np.ndarray]]:
    raw = dataset
    if isinstance(dataset, dict):
        raw = dataset.get("dtrajs")
    if raw and not (isinstance(raw, list) and all(d is None for d in raw)):
        dtrajs_list = list(raw) if isinstance(raw, list) else [np.asarray(raw)]

        # Log shape after loading pre-computed discrete trajectories for each shard
        logger.info("Using pre-computed discrete trajectories (dtrajs)")
        for idx, dtraj in enumerate(dtrajs_list):
            dtraj_array = np.asarray(dtraj, dtype=np.int32)
            logger.info(f"Featurized shard {idx}: features shape={dtraj_array.shape}")

        return dtrajs_list
    if isinstance(dataset, dict):
        return _cluster_continuous_trajectories(dataset, opts)
    logger.warning("No dtrajs or continuous data available for MSM building")
    return None


def _clean_dtrajs(dtrajs: Sequence[Any]) -> List[np.ndarray]:
    clean: List[np.ndarray] = []
    for traj in dtrajs:
        if traj is None:
            continue
        arr = np.asarray(traj, dtype=np.int32).reshape(-1)
        if arr.size:
            clean.append(arr)
    return clean


def _observed_state_count(dtrajs: Sequence[np.ndarray]) -> int:
    observed: set[int] = set()
    for traj in dtrajs:
        arr = np.asarray(traj, dtype=int).reshape(-1)
        if arr.size:
            observed.update(int(v) for v in arr if int(v) >= 0)
    return len(observed)


def _compute_msm_statistics(
    dtrajs: Sequence[np.ndarray], n_states: int, lag_time: int
) -> tuple[np.ndarray, int, np.ndarray]:
    """Compute transition counts and state counts from discrete trajectories.

    Returns:
        (counts, total_pairs, state_counts)
    """
    counts = np.zeros((n_states, n_states), dtype=float)
    state_counts = np.zeros((n_states,), dtype=float)
    total_pairs = 0

    for dtraj in dtrajs:
        arr = np.asarray(dtraj, dtype=np.int32)

        # Count state visits
        for state in arr:
            if 0 <= state < n_states:
                state_counts[state] += 1.0

        # Count transitions with sliding window
        if arr.size > lag_time:
            for i in range(0, arr.size - lag_time):
                s_i = int(arr[i])
                s_j = int(arr[i + lag_time])
                if 0 <= s_i < n_states and 0 <= s_j < n_states:
                    counts[s_i, s_j] += 1.0
                    total_pairs += 1

    return counts, total_pairs, state_counts


def _build_msm(dataset: Any, opts: BuildOpts, applied: AppliedOpts) -> Any:
    _ensure_msm_whitened(dataset)
    dtrajs = _prepare_dtrajs(dataset, opts)
    if not dtrajs:
        return None
    clean = _clean_dtrajs(dtrajs)
    if not clean:
        return None
    lag_time = int(opts.lag_time)
    applied_lag = getattr(applied, "lag", None) if applied is not None else None
    if applied_lag is not None:
        candidate = int(applied_lag)
        if candidate <= 0:
            raise ValueError("Applied MSM lag must be a positive integer")
        lag_time = candidate
    if not any(traj.size > lag_time for traj in clean):
        max_length = max(traj.size for traj in clean)
        raise ValueError(
            f"MSM lag {lag_time} exceeds all trajectory lengths; "
            f"longest trajectory length is {max_length}"
        )
    T, pi = build_simple_msm(
        clean,
        n_states=opts.n_states,
        lag=lag_time,
        count_mode=str(opts.count_mode),
    )
    if pi.size == 0 or not np.isfinite(np.sum(pi)) or np.sum(pi) == 0.0:
        raise RuntimeError("Failed to compute a valid stationary distribution")

    # Also compute and return detailed MSM statistics
    n_states = T.shape[0]
    counts, total_pairs, state_counts = _compute_msm_statistics(
        clean, n_states, lag_time
    )

    msm_data = {
        "transition_matrix": T,
        "stationary_distribution": pi,
        "counts": counts,
        "state_counts": state_counts,
        "counted_pairs": {"all": total_pairs},
        "n_states": n_states,
        "lag_time": lag_time,
        "dtrajs": clean,  # Store for debugging
    }

    return T, pi, msm_data


def _build_fes(dataset: Any, opts: BuildOpts, applied: AppliedOpts) -> Any:
    return default_fes_builder(dataset, opts, applied)


def _build_tram(dataset: Any, opts: BuildOpts, applied: AppliedOpts) -> Any:
    try:
        return default_tram_builder(dataset, opts, applied)
    except Exception as e:
        logger.warning("TRAM build failed: %s", e)
        return {"skipped": True, "reason": f"tram_error: {e}"}


# --- Default builders ---------------------------------------------------------


def _coerce_cv_arrays(
    cv1: np.ndarray, cv2: np.ndarray
) -> tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[str]]:
    try:
        a = np.asarray(cv1, dtype=float).reshape(-1)
        b = np.asarray(cv2, dtype=float).reshape(-1)
    except Exception as exc:
        logger.warning("Failed to coerce CVs to float: %s", exc)
        return None, None, "cv_coercion_failed"
    if a.size == 0 or b.size == 0:
        return None, None, "empty_cvs"
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        return None, None, "non_finite_cvs"
    return a, b, None


def _generate_fes(
    a: np.ndarray,
    b: np.ndarray,
    names: Tuple[str, str],
    periodic: Tuple[bool, bool],
    opts: BuildOpts,
) -> Dict[str, Any]:
    from pmarlo.markov_state_model.free_energy import generate_2d_fes

    fes = generate_2d_fes(
        a,
        b,
        temperature=opts.temperature,
        periodic=periodic,
        config=opts,
    )
    return {"result": fes, "cv1_name": names[0], "cv2_name": names[1]}


def default_fes_builder(
    dataset: Any, opts: BuildOpts, applied: AppliedOpts
) -> Any | None:
    """Build a simple free energy surface."""

    if isinstance(dataset, dict):
        try:
            ensure_fes_inputs_whitened(dataset)
        except Exception:
            logger.debug("Failed to apply CV whitening before FES build", exc_info=True)

    cv_pair = _extract_cvs(dataset)
    if cv_pair is None:
        return {"skipped": True, "reason": "no_cvs"}

    cv1, cv2, names, periodic = cv_pair
    a, b, error = _coerce_cv_arrays(cv1, cv2)
    if error is not None:
        return {"skipped": True, "reason": error}

    assert a is not None and b is not None
    if np.ptp(a) == 0.0 or np.ptp(b) == 0.0:
        return {"skipped": True, "reason": "constant_cvs"}
    return _generate_fes(a, b, names, periodic, opts)


def default_tram_builder(
    dataset: Any, opts: BuildOpts, applied: AppliedOpts
) -> Any | None:
    logger.info("TRAM builder not yet implemented")
    return {"skipped": True, "reason": "not_implemented"}


# --- Artifact helpers --------------------------------------------------------


def _fes_metadata_summary(meta: Dict[str, Any]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    frac = meta.get("empty_bins_fraction")
    try:
        summary["empty_bins_fraction"] = float(frac) if frac is not None else 0.0
    except Exception:
        summary["empty_bins_fraction"] = 0.0
    warn = meta.get("sparse_warning")
    summary["warn_sparse"] = bool(warn)
    if warn:
        summary["sparse_warning"] = str(warn)
    banner = meta.get("sparse_banner")
    if banner:
        summary["sparse_banner"] = str(banner)
    method = meta.get("method")
    if method:
        summary["method"] = str(method)
    smoothing_meta = meta.get("smoothing")
    if isinstance(smoothing_meta, Mapping):
        smoothing_summary: Dict[str, Any] = {}
        mode = smoothing_meta.get("mode")
        if mode is not None:
            smoothing_summary["mode"] = str(mode)
        target = smoothing_meta.get("target_sd_kT")
        try:
            smoothing_summary["target_sd_kT"] = float(target) if target is not None else None
        except Exception:
            smoothing_summary["target_sd_kT"] = None
        applied = smoothing_meta.get("applied_fraction")
        try:
            smoothing_summary["applied_fraction"] = float(applied) if applied is not None else 0.0
        except Exception:
            smoothing_summary["applied_fraction"] = 0.0
        for key in ("alpha", "h0", "ess_ref", "h_min", "h_max"):
            value = smoothing_meta.get(key)
            if value is not None:
                try:
                    smoothing_summary[key] = float(value)
                except Exception:
                    continue
        summary["smoothing"] = smoothing_summary
    return summary


def _coerce_temperature(value: Any) -> Optional[float]:
    coerced = _coerce_float(value) if value is not None else None
    return coerced


def _extract_fes_quality_artifact(fes_obj: Any) -> Dict[str, Any]:
    """Derive a lightweight quality summary from an :class:`FESResult`."""

    quality: Dict[str, Any] = {}
    metadata = getattr(fes_obj, "metadata", {})
    if isinstance(metadata, dict):
        quality.update(_fes_metadata_summary(metadata))
        temperature_source = metadata.get("temperature")
    else:
        temperature_source = None
    temperature = getattr(fes_obj, "temperature", None)
    if temperature is None:
        temperature = temperature_source
    temp_val = _coerce_temperature(temperature)
    if temp_val is not None:
        quality["temperature_K"] = temp_val
    return quality


# --- Utility functions --------------------------------------------------------


def validate_build_opts(opts: BuildOpts) -> List[str]:
    warnings = []

    if opts.n_clusters <= 0:
        warnings.append("n_clusters must be positive")
    if opts.n_states <= 0:
        warnings.append("n_states must be positive")
    if opts.lag_time <= 0:
        warnings.append("lag_time must be positive")
    if opts.n_states > opts.n_clusters:
        warnings.append("n_states should not exceed n_clusters")

    if opts.fes_temperature <= 0:
        warnings.append("fes_temperature must be positive")

    if opts.tram_lag <= 0:
        warnings.append("tram_lag must be positive")
    if opts.tram_n_iter <= 0:
        warnings.append("tram_n_iter must be positive")

    if opts.n_jobs <= 0:
        warnings.append("n_jobs must be positive")
    if opts.chunk_size <= 0:
        warnings.append("chunk_size must be positive")

    return warnings


def _sanitize_artifacts(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        value = float(obj)
        return value if math.isfinite(value) else None
    if isinstance(obj, str):
        return obj
    if isinstance(obj, np.ndarray):
        return _sanitize_artifacts(obj.tolist())
    if isinstance(obj, dict):
        return {str(k): _sanitize_artifacts(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_artifacts(v) for v in obj]
    return str(obj)


def estimate_memory_usage(dataset: Any, opts: BuildOpts) -> float:
    """Estimate the memory required to build the MSM pipeline.

    The estimation relies on the dataset exposing both the number of frames and
    a sequence of feature names.  The build process assumes these values are
    present – if they are missing we raise an error instead of attempting to
    continue with guessed defaults.
    """

    n_frames = _count_frames(dataset)
    if n_frames <= 0:
        raise ValueError("Cannot estimate memory usage: dataset has no frames")

    feature_names = _extract_feature_names(dataset)
    if not feature_names:
        raise ValueError(
            "Cannot estimate memory usage: dataset has no feature names"
        )

    n_features = len(feature_names)
    dataset_gb = (n_frames * n_features * 8) / (1024**3)
    msm_gb = (opts.n_clusters * n_features * 8) / (1024**3)
    msm_gb += (opts.n_states * opts.n_states * 8) / (1024**3)
    fes_gb = (100 * 100 * 8) / (1024**3) if opts.enable_fes else 0.0

    return dataset_gb + msm_gb + fes_gb


def create_build_summary(result: BuildResult) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "success": result.metadata.success if result.metadata else False,
        "n_frames": result.n_frames,
        "n_shards": result.n_shards,
        "n_features": len(result.feature_names),
        "has_msm": bool(result.flags.get("has_msm")),
        "has_fes": bool(result.flags.get("has_fes")),
        "has_tram": bool(result.flags.get("has_tram")),
    }

    if result.metadata:
        summary.update(
            {
                "run_id": result.metadata.run_id,
                "duration": result.metadata.duration_seconds,
                "hostname": result.metadata.hostname,
            }
        )

    return summary
