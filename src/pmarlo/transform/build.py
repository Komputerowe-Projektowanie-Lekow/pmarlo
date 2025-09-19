from __future__ import annotations

import base64
import json
import logging
import math
import os
import tempfile
from dataclasses import asdict, dataclass, field, is_dataclass, replace
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from ..analysis.fes import ensure_fes_inputs_whitened
from ..analysis.msm import ensure_msm_inputs_whitened
from ..markov_state_model._msm_utils import build_simple_msm
from ..utils.seed import set_global_seed
from .apply import apply_transform_plan
from .plan import TransformPlan, TransformStep
from .progress import ProgressCB
from .runner import apply_plan as _apply_plan

logger = logging.getLogger("pmarlo")


# --- Shard selection helpers -------------------------------------------------


@lru_cache(maxsize=512)
def _load_shard_metadata_cached(path_str: str) -> Dict[str, Any]:
    try:
        return json.loads(Path(path_str).read_text())
    except Exception:
        return {}


def _get_shard_metadata(path: Path) -> Dict[str, Any]:
    return _load_shard_metadata_cached(str(Path(path)))


def _is_demux_shard(path: Path, meta: Optional[Dict[str, Any]] = None) -> bool:
    data = meta if meta is not None else _get_shard_metadata(path)
    if isinstance(data, dict):
        source = data.get("source")
        if isinstance(source, dict):
            kind = str(source.get("kind", "")).lower()
            if kind:
                return kind == "demux"
            for key in ("traj", "path", "file", "source_path"):
                raw = source.get(key)
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


def _collect_demux_temperatures(meta: Dict[str, Any]) -> List[float]:
    temps: List[float] = []
    if not isinstance(meta, dict):
        return temps

    candidates = [meta.get("temperature")]
    source = meta.get("source")
    if isinstance(source, dict):
        candidates.extend([source.get("temperature_K"), source.get("temperature")])

    for candidate in candidates:
        coerced = _coerce_float(candidate)
        if coerced is None:
            continue
        if all(abs(coerced - existing) > 1e-9 for existing in temps):
            temps.append(coerced)
    return temps


def _temperature_matches_target(
    temperatures: Sequence[float], target: float, tolerance: float
) -> bool:
    tol = tolerance if tolerance >= 0.0 else 0.0
    return any(abs(temp - target) <= tol for temp in temperatures)


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
        filtered: List[Path] = []
        for shard_path in shards:
            meta = _get_shard_metadata(shard_path)
            if not _is_demux_shard(shard_path, meta):
                continue
            if demux_temperature is not None:
                temps = _collect_demux_temperatures(meta)
                if not temps:
                    continue
                if not _temperature_matches_target(
                    temps, float(demux_temperature), tol
                ):
                    continue
            filtered.append(shard_path)
        shards = filtered

    elif mode == "first":
        limit = max_shards if max_shards else 10
        shards = shards[:limit]
    elif mode == "last":
        limit = max_shards if max_shards else 10
        shards = shards[-limit:]
    elif mode == "random":
        import random

        shuffled = list(shards)
        random.shuffle(shuffled)
        limit = max_shards if max_shards else 10
        shards = shuffled[:limit]
    elif mode == "all":
        pass
    else:
        raise ValueError(f"Unknown selection mode: {mode}")

    if max_shards and mode in {"demux", "all"} and len(shards) > max_shards:
        shards = shards[:max_shards]

    return shards


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
        temps = _collect_demux_temperatures(meta)
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

    def __post_init__(self) -> None:
        if self.lag_candidates is not None:
            object.__setattr__(
                self, "lag_candidates", tuple(int(x) for x in self.lag_candidates)
            )
        if self.temperature is not None:
            object.__setattr__(self, "fes_temperature", float(self.temperature))

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

    def to_json(self) -> str:
        def _serialize_array(arr: Optional[np.ndarray]) -> Optional[Dict[str, Any]]:
            if arr is None:
                return None
            return {
                "dtype": str(arr.dtype),
                "shape": list(arr.shape),
                "data": base64.b64encode(arr.tobytes()).decode("ascii"),
            }

        def _serialize_generic(obj: Any) -> Any:
            if obj is None:
                return None
            if hasattr(obj, "to_dict"):
                return obj.to_dict()  # type: ignore[call-arg]
            if is_dataclass(obj):
                return asdict(obj)
            if hasattr(obj, "__dict__"):
                return obj.__dict__
            return _sanitize_artifacts(obj)

        def _serialize_fes(obj: Any) -> Any:
            from ..markov_state_model.free_energy import FESResult

            if obj is None:
                return None
            if isinstance(obj, FESResult):
                return {
                    "F": _serialize_array(obj.F),
                    "xedges": _serialize_array(obj.xedges),
                    "yedges": _serialize_array(obj.yedges),
                    "levels_kJmol": _serialize_array(obj.levels_kJmol),
                    "metadata": _sanitize_artifacts(obj.metadata),
                }
            if isinstance(obj, dict):
                return obj
            return _serialize_generic(obj)

        applied_dict = None
        if self.applied_opts is not None:
            applied_dict = asdict(self.applied_opts)
            shards = applied_dict.get("selected_shards")
            if isinstance(shards, list):
                applied_dict["selected_shards"] = [str(s) for s in shards]

        data = {
            "transition_matrix": _serialize_array(self.transition_matrix),
            "stationary_distribution": _serialize_array(self.stationary_distribution),
            "msm": _serialize_generic(self.msm),
            "fes": _serialize_fes(self.fes),
            "tram": _serialize_generic(self.tram),
            "metadata": self.metadata.to_dict() if self.metadata else None,
            "applied_opts": applied_dict,
            "n_frames": self.n_frames,
            "n_shards": self.n_shards,
            "feature_names": self.feature_names,
            "cluster_populations": _serialize_array(self.cluster_populations),
            "artifacts": _sanitize_artifacts(self.artifacts),
            "messages": list(self.messages),
            "flags": _sanitize_artifacts(dict(self.flags)),
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
        )


# --- Build functions ---------------------------------------------------------


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

    plan_to_use = plan or opts.plan

    if applied is None:
        applied_obj = AppliedOpts.from_opts(opts, [], plan=plan_to_use)
    else:
        applied_obj = applied
        if applied_obj.original_opts is None:
            applied_obj.original_opts = opts
        if applied_obj.actual_plan is None and plan_to_use is not None:
            applied_obj.actual_plan = plan_to_use

    set_global_seed(opts.seed)

    import platform
    import socket
    from datetime import datetime

    start_dt = datetime.now()
    metadata = RunMetadata(
        run_id=_generate_run_id(),
        start_time=start_dt.isoformat(),
        hostname=socket.gethostname(),
        transform_plan=tuple(plan_to_use.steps) if plan_to_use else None,
        applied_opts=applied_obj,
        seed=opts.seed,
        temperature=opts.temperature,
        python_version=platform.python_version(),
    )

    try:
        working_dataset = dataset
        if plan_to_use is not None:
            logger.info("Applying transform plan with %d steps", len(plan_to_use.steps))
            working_dataset = _apply_plan(
                plan_to_use, working_dataset, progress_callback=progress_callback
            )
            applied_obj.actual_plan = plan_to_use

        artifacts: Dict[str, Any] = {}
        if isinstance(working_dataset, dict):
            raw_artifacts = working_dataset.get("__artifacts__")
            if isinstance(raw_artifacts, dict):
                artifacts = _sanitize_artifacts(raw_artifacts)

        # Record provenance notes for learned CVs, if present
        try:
            if "mlcv_deeptica" in artifacts:
                # Minimal note to satisfy downstream gating/tests
                note = {"method": "deeptica"}
                # Include selected high-signal fields when available
                try:
                    summ = artifacts.get("mlcv_deeptica", {})
                    if isinstance(summ, dict):
                        for key in ("lag", "n_out", "pairs_total", "model_prefix"):
                            if key in summ and summ[key] is not None:
                                note[key] = summ[key]
                except Exception:
                    pass
                if applied_obj.notes is None:
                    applied_obj.notes = {}
                applied_obj.notes["mlcv"] = note
        except Exception:
            # Notes are best-effort and must not break the build
            pass

        # Also record CV bin edges for the first two learned CVs
        try:
            if isinstance(working_dataset, dict) and "X" in working_dataset:
                X_arr = np.asarray(working_dataset.get("X"), dtype=float)
                if X_arr.ndim == 2 and X_arr.shape[1] >= 2 and X_arr.shape[0] > 0:
                    cv1 = X_arr[:, 0]
                    cv2 = X_arr[:, 1]
                    n1 = 32
                    n2 = 32
                    try:
                        if isinstance(applied_obj.bins, dict):
                            n1 = int(applied_obj.bins.get("cv1", n1))
                            n2 = int(applied_obj.bins.get("cv2", n2))
                    except Exception:
                        pass

                    def _bounds(arr):
                        a_min = float(np.nanmin(arr))
                        a_max = float(np.nanmax(arr))
                        if (
                            not np.isfinite(a_min)
                            or not np.isfinite(a_max)
                            or a_max <= a_min
                        ):
                            return -1.0, 1.0
                        return a_min, a_max

                    a_min, a_max = _bounds(cv1)
                    b_min, b_max = _bounds(cv2)
                    e1 = (
                        np.linspace(a_min, a_max, int(max(2, n1)) + 1)
                        .astype(float)
                        .tolist()
                    )
                    e2 = (
                        np.linspace(b_min, b_max, int(max(2, n2)) + 1)
                        .astype(float)
                        .tolist()
                    )
                    if applied_obj.notes is None:
                        applied_obj.notes = {}
                    applied_obj.notes["cv_bin_edges"] = {"cv1": e1, "cv2": e2}
        except Exception:
            pass

        transition_matrix: Optional[np.ndarray] = None
        stationary_distribution: Optional[np.ndarray] = None
        msm_payload: Optional[Any] = None
        if opts.msm_mode != "none":
            logger.info("Building MSM...")
            msm_result = _build_msm(working_dataset, opts, applied_obj)
            if isinstance(msm_result, tuple) and len(msm_result) == 2:
                transition_matrix, stationary_distribution = msm_result
            else:
                msm_payload = msm_result

        fes_payload: Optional[Any] = None
        if opts.enable_fes:
            logger.info("Building FES...")
            fes_raw = _build_fes(working_dataset, opts, applied_obj)
            if isinstance(fes_raw, dict) and "result" in fes_raw:
                result_obj = fes_raw.get("result")
                metadata.fes = {
                    "bins": None,
                    "names": tuple(
                        x
                        for x in (fes_raw.get("cv1_name"), fes_raw.get("cv2_name"))
                        if x
                    ),
                    "temperature": opts.temperature,
                }
                fes_payload = result_obj
            else:
                metadata.fes = None
                if isinstance(fes_raw, dict) and fes_raw.get("skipped"):
                    fes_payload = None
                else:
                    fes_payload = fes_raw

        tram_payload: Optional[Any] = None
        if opts.enable_tram:
            logger.info("Building TRAM...")
            tram_payload = _build_tram(working_dataset, opts, applied_obj)

        end_dt = datetime.now()
        metadata.end_time = end_dt.isoformat()
        metadata.duration_seconds = (end_dt - start_dt).total_seconds()
        metadata.success = True

        n_frames = _count_frames(working_dataset)
        feature_names = _extract_feature_names(working_dataset)

        if not applied_obj.selected_shards and isinstance(dataset, dict):
            shards_meta = dataset.get("__shards__")
            if isinstance(shards_meta, list):
                try:
                    applied_obj.selected_shards = [
                        Path(str(item.get("id", ""))) for item in shards_meta
                    ]
                except Exception:
                    applied_obj.selected_shards = []

        n_shards = len(applied_obj.selected_shards)
        if n_shards == 0 and isinstance(dataset, dict):
            shards_meta = dataset.get("__shards__")
            if isinstance(shards_meta, list):
                n_shards = len(shards_meta)

        flags: Dict[str, Any] = {}
        if transition_matrix is not None and transition_matrix.size > 0:
            flags["has_msm"] = True
        if fes_payload is not None:
            flags["has_fes"] = True
        if tram_payload not in (None, {}):
            flags["has_tram"] = True
        if "mlcv_deeptica" in artifacts:
            summary = artifacts["mlcv_deeptica"]
            flags["mlcv_deeptica_applied"] = bool(summary.get("applied"))

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
        )

    except Exception as exc:
        logger.error("Build failed: %s", exc)
        metadata.error_message = str(exc)
        metadata.success = False
        raise


def _generate_run_id() -> str:
    import time

    return f"build_{int(time.time())}_{os.getpid()}"


def _count_frames(dataset: Any) -> int:
    try:
        if hasattr(dataset, "__len__"):
            return len(dataset)
        if hasattr(dataset, "n_frames"):
            return dataset.n_frames
        if isinstance(dataset, dict) and "X" in dataset:
            return int(np.asarray(dataset["X"]).shape[0])
        return 0
    except Exception:
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


def _build_msm(dataset: Any, opts: BuildOpts, applied: AppliedOpts) -> Any:
    try:
        dtrajs: Any = dataset
        if isinstance(dataset, dict):
            try:
                ensure_msm_inputs_whitened(dataset)
            except Exception:
                logger.debug("Failed to apply CV whitening before MSM build", exc_info=True)
        if isinstance(dataset, dict):
            dtrajs = dataset.get("dtrajs")

        # If dtrajs are missing or empty, try to create them from continuous CV data
        if not dtrajs or (isinstance(dtrajs, list) and all(d is None for d in dtrajs)):
            if isinstance(dataset, dict) and "X" in dataset:
                logger.info(
                    "No discrete trajectories found, clustering continuous CV data for MSM..."
                )
                X = dataset["X"]
                if isinstance(X, np.ndarray) and X.size > 0:
                    # Import clustering function
                    from ..markov_state_model.clustering import cluster_microstates

                    # Perform clustering to create discrete trajectories
                    clustering = cluster_microstates(
                        X,
                        n_states=opts.n_states,
                        method="kmeans",
                        random_state=opts.seed,
                    )

                    labels = clustering.labels
                    if labels is not None and labels.size > 0:
                        # Split labels back into per-shard trajectories based on shard info
                        shards_info = dataset.get("__shards__", [])
                        if shards_info:
                            dtrajs = []
                            for shard_info in shards_info:
                                start = int(shard_info.get("start", 0))
                                stop = int(shard_info.get("stop", start))
                                if stop > start:
                                    shard_labels = labels[start:stop]
                                    dtrajs.append(shard_labels.astype(np.int32))
                        else:
                            # Single trajectory case
                            dtrajs = [labels.astype(np.int32)]

                        logger.info(
                            f"Created {len(dtrajs)} discrete trajectories from clustering"
                        )
                    else:
                        logger.warning("Clustering failed to produce labels")
                        return None
                else:
                    logger.warning("No continuous CV data available for clustering")
                    return None
            else:
                logger.warning(
                    "No dtrajs or continuous data available for MSM building"
                )
                return None

        if isinstance(dtrajs, list):
            clean: List[np.ndarray] = []
            for dt in dtrajs:
                if dt is None:
                    continue
                arr = np.asarray(dt, dtype=np.int32).reshape(-1)
                if arr.size:
                    clean.append(arr)
            if not clean:
                return None
            dtrajs = clean
        return build_simple_msm(
            dtrajs,
            n_states=opts.n_states,
            lag=opts.lag_time,
            count_mode=str(opts.count_mode),
        )
    except Exception as e:
        logger.warning("MSM build failed: %s", e)
        return None


def _build_fes(dataset: Any, opts: BuildOpts, applied: AppliedOpts) -> Any:
    try:
        return default_fes_builder(dataset, opts, applied)
    except Exception as e:
        logger.warning("FES build failed: %s", e)
        return None


def _build_tram(dataset: Any, opts: BuildOpts, applied: AppliedOpts) -> Any:
    try:
        return default_tram_builder(dataset, opts, applied)
    except Exception as e:
        logger.warning("TRAM build failed: %s", e)
        return {"skipped": True, "reason": f"tram_error: {e}"}


# --- Default builders ---------------------------------------------------------


def default_fes_builder(
    dataset: Any, opts: BuildOpts, applied: AppliedOpts
) -> Any | None:
    """Build a simple free energy surface with histogram fallback."""

    if isinstance(dataset, dict):
        try:
            ensure_fes_inputs_whitened(dataset)
        except Exception:
            logger.debug("Failed to apply CV whitening before FES build", exc_info=True)

    cv_pair = _extract_cvs(dataset)
    if cv_pair is None:
        return {"skipped": True, "reason": "no_cvs"}

    cv1, cv2, names, periodic = cv_pair
    try:
        a = np.asarray(cv1, dtype=float).reshape(-1)
        b = np.asarray(cv2, dtype=float).reshape(-1)
        if a.size == 0 or b.size == 0:
            return {"skipped": True, "reason": "empty_cvs"}
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            return {"skipped": True, "reason": "non_finite_cvs"}
    except Exception as exc:
        logger.warning("Failed to coerce CVs to float: %s", exc)
        return {"skipped": True, "reason": "cv_coercion_failed"}

    try:
        from pmarlo.markov_state_model.free_energy import generate_2d_fes

        fes = generate_2d_fes(
            a,
            b,
            temperature=opts.temperature,
            periodic=periodic,
        )
        return {"result": fes, "cv1_name": names[0], "cv2_name": names[1]}
    except Exception as exc:
        logger.warning("FES generation failed: %s; using histogram fallback", exc)
        try:
            hist, xedges, yedges = np.histogram2d(a, b, bins=32)
        except Exception as exc2:
            logger.warning("Histogram fallback failed: %s", exc2)
            a_min, a_max = float(np.min(a)), float(np.max(a))
            b_min, b_max = float(np.min(b)), float(np.max(b))
            if not np.isfinite(a_min) or not np.isfinite(a_max):
                a_min, a_max = -1.0, 1.0
            if not np.isfinite(b_min) or not np.isfinite(b_max):
                b_min, b_max = -1.0, 1.0
            if a_max <= a_min:
                a_max = a_min + 1.0
            if b_max <= b_min:
                b_max = b_min + 1.0
            xedges = np.linspace(a_min, a_max, 33)
            yedges = np.linspace(b_min, b_max, 33)
            hist = np.ones((32, 32), dtype=np.float64)
        hist = np.asarray(hist, dtype=np.float64)
        with np.errstate(divide="ignore"):
            F = -np.log(hist + 1e-12)

        from pmarlo.markov_state_model.free_energy import FESResult

        fallback = FESResult(
            F=F,
            xedges=xedges,
            yedges=yedges,
            metadata={"method": "histogram", "temperature": opts.temperature},
        )
        return {"result": fallback, "cv1_name": names[0], "cv2_name": names[1]}


def default_tram_builder(
    dataset: Any, opts: BuildOpts, applied: AppliedOpts
) -> Any | None:
    logger.info("TRAM builder not yet implemented")
    return {"skipped": True, "reason": "not_implemented"}


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
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): _sanitize_artifacts(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_artifacts(v) for v in obj]
    return str(obj)


def estimate_memory_usage(dataset: Any, opts: BuildOpts) -> float:
    try:
        n_frames = _count_frames(dataset)
        n_features = len(_extract_feature_names(dataset))

        dataset_gb = (n_frames * n_features * 8) / (1024**3)
        msm_gb = (opts.n_clusters * n_features * 8) / (1024**3)
        msm_gb += (opts.n_states * opts.n_states * 8) / (1024**3)
        fes_gb = (100 * 100 * 8) / (1024**3) if opts.enable_fes else 0

        return (dataset_gb + msm_gb + fes_gb) * 1.5
    except Exception:
        return 1.0


def create_build_summary(result: BuildResult) -> Dict[str, Any]:
    summary = {
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
