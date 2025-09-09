from __future__ import annotations

import base64
import json
import logging
import os
import tempfile
from dataclasses import asdict, dataclass, field, replace
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from ..markov_state_model._msm_utils import build_simple_msm
from .progress import ProgressCB
from .apply import apply_transform_plan
from .plan import TransformPlan, TransformStep
from .runner import apply_plan as _apply_plan

logger = logging.getLogger("pmarlo")


# --- Shard selection helpers -------------------------------------------------


def select_shards(
    all_shards: Sequence[Union[str, Path]],
    *,
    mode: str = "demux",
    max_shards: Optional[int] = None,
    sort_key: Optional[Callable[[Union[str, Path]], Any]] = None,
) -> List[Path]:
    """Select a subset of shards for analysis.

    Parameters
    ----------
    all_shards
        Full list of shard paths.
    mode
        Selection mode: 'demux', 'first', 'last', 'random', or 'all'.
    max_shards
        Maximum number of shards to select.
    sort_key
        Optional function to sort shards before selection.

    Returns
    -------
    List[Path]
        Selected shard paths.
    """
    shards = [Path(s) for s in all_shards]

    if sort_key:
        shards = sorted(shards, key=sort_key)

    if mode == "demux":
        # For demux mode, prefer shards with demux metadata
        demux_shards = [s for s in shards if "demux" in s.name.lower()]
        if demux_shards:
            shards = demux_shards

    elif mode == "first":
        shards = shards[:max_shards] if max_shards else shards[:10]
    elif mode == "last":
        shards = shards[-max_shards:] if max_shards else shards[-10:]
    elif mode == "random":
        import random

        random.shuffle(shards)
        shards = shards[:max_shards] if max_shards else shards[:10]
    elif mode == "all":
        pass  # Use all shards
    else:
        raise ValueError(f"Unknown selection mode: {mode}")

    if max_shards and len(shards) > max_shards:
        shards = shards[:max_shards]

    return shards


# --- Configuration classes ---------------------------------------------------


@dataclass(frozen=True)
class BuildOpts:
    """Configuration options for the build process."""

    # Transform plan
    plan: Optional[TransformPlan] = None

    # Shard selection
    shard_selection_mode: str = "demux"
    max_shards: Optional[int] = None

    # MSM options
    n_clusters: int = 200
    n_states: int = 50
    lag_time: int = 10
    msm_mode: str = "kmeans+msm"

    # FES options
    enable_fes: bool = True
    fes_temperature: float = 300.0

    # TRAM options
    enable_tram: bool = False
    tram_lag: int = 1
    tram_n_iter: int = 100

    # Output options
    output_format: str = "json"
    save_trajectories: bool = False
    save_plots: bool = True

    # Performance options
    n_jobs: int = 1
    memory_limit_gb: Optional[float] = None
    chunk_size: int = 1000

    # Debugging options
    debug: bool = False
    verbose: bool = False

    def with_plan(self, plan: TransformPlan) -> "BuildOpts":
        """Return a new BuildOpts with the specified plan."""
        return replace(self, plan=plan)

    def with_shards(
        self, mode: str = "demux", max_shards: Optional[int] = None
    ) -> "BuildOpts":
        """Return a new BuildOpts with shard selection options."""
        return replace(self, shard_selection_mode=mode, max_shards=max_shards)

    def with_msm(
        self, n_clusters: int = 200, n_states: int = 50, lag_time: int = 10
    ) -> "BuildOpts":
        """Return a new BuildOpts with MSM options."""
        return replace(
            self, n_clusters=n_clusters, n_states=n_states, lag_time=lag_time
        )


@dataclass(frozen=True)
class AppliedOpts:
    """Applied configuration after processing BuildOpts."""

    # Original options
    original_opts: BuildOpts

    # Resolved values
    selected_shards: List[Path] = field(default_factory=list)
    actual_plan: Optional[TransformPlan] = None
    effective_n_jobs: int = 1
    effective_memory_limit: Optional[float] = None

    # Runtime metadata
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
        """Create AppliedOpts from BuildOpts and runtime information."""
        import socket
        from datetime import datetime

        return cls(
            original_opts=opts,
            selected_shards=selected_shards,
            actual_plan=plan or opts.plan,
            effective_n_jobs=opts.n_jobs,
            effective_memory_limit=opts.memory_limit_gb,
            start_time=datetime.now().isoformat(),
            hostname=socket.gethostname(),
        )


@dataclass
class RunMetadata:
    """Metadata about a build run."""

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

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunMetadata":
        """Create RunMetadata from a dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> Dict[str, Any]:
        """Convert RunMetadata to a dictionary."""
        return asdict(self)


@dataclass
class BuildResult:
    """Result of a build operation."""

    # Core results
    msm: Optional[Any] = None
    fes: Optional[Any] = None
    tram: Optional[Any] = None

    # Metadata
    metadata: Optional[RunMetadata] = None
    applied_opts: Optional[AppliedOpts] = None

    # Diagnostics
    n_frames: int = 0
    n_shards: int = 0
    feature_names: List[str] = field(default_factory=list)
    cluster_populations: Optional[np.ndarray] = None

    # Serialization support
    def to_json(self) -> str:
        """Serialize BuildResult to JSON string."""

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
                return obj.to_dict()
            if hasattr(obj, "__dict__"):
                return obj.__dict__
            return str(obj)

        data = {
            "msm": _serialize_generic(self.msm),
            "fes": _serialize_generic(self.fes),
            "tram": _serialize_generic(self.tram),
            "metadata": self.metadata.to_dict() if self.metadata else None,
            "applied_opts": _serialize_generic(self.applied_opts),
            "n_frames": self.n_frames,
            "n_shards": self.n_shards,
            "feature_names": self.feature_names,
            "cluster_populations": _serialize_array(self.cluster_populations),
        }

        return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)

    @classmethod
    def from_json(cls, text: str) -> "BuildResult":
        """Deserialize BuildResult from JSON produced by to_json."""
        from pmarlo.markov_state_model.free_energy import (  # local import to avoid cycles
            FESResult,
        )

        data = json.loads(text)
        md = RunMetadata.from_dict(data["metadata"]) if "metadata" in data else None

        def _decode_array(obj: Optional[Dict[str, Any]]) -> Optional[np.ndarray]:
            if obj is None:
                return None
            dtype = np.dtype(obj["dtype"])
            shape = tuple(obj["shape"])
            data_bytes = base64.b64decode(obj["data"])
            return np.frombuffer(data_bytes, dtype=dtype).reshape(shape)

        def _decode_fes(obj: Optional[Dict[str, Any]]) -> Optional[FESResult]:
            if obj is None:
                return None
            # Reconstruct FESResult from serialized data
            try:
                return FESResult(
                    F=_decode_array(obj.get("F")),
                    xedges=_decode_array(obj.get("xedges")),
                    yedges=_decode_array(obj.get("yedges")),
                    levels_kJmol=_decode_array(obj.get("levels_kJmol")),
                    metadata=obj.get("metadata", {}),
                )
            except Exception:
                return None

        return cls(
            msm=data.get("msm"),
            fes=_decode_fes(data.get("fes")),
            tram=data.get("tram"),
            metadata=md,
            applied_opts=data.get("applied_opts"),
            n_frames=data.get("n_frames", 0),
            n_shards=data.get("n_shards", 0),
            feature_names=data.get("feature_names", []),
            cluster_populations=_decode_array(data.get("cluster_populations")),
        )


# --- Build functions ----------------------------------------------------------


def build_result(
    dataset: Any,
    opts: Optional[BuildOpts] = None,
    *,
    progress_callback: Optional[ProgressCB] = None,
) -> BuildResult:
    """Build MSM, FES, and other analyses from a dataset.

    This is the main entry point for the build system. It applies the
    configured transform plan and builds the requested analyses.

    Parameters
    ----------
    dataset
        Input dataset (typically from aggregate operations).
    opts
        Build configuration options.
    progress_callback
        Optional progress callback function.

    Returns
    -------
    BuildResult
        Results of the build operation.
    """
    if opts is None:
        opts = BuildOpts()

    # Create applied options
    applied = AppliedOpts.from_opts(opts, [])

    # Create metadata
    metadata = RunMetadata(
        run_id=_generate_run_id(),
        start_time=applied.start_time or "",
        hostname=applied.hostname,
    )

    try:
        # Apply transform plan if specified
        if opts.plan:
            logger.info(f"Applying transform plan with {len(opts.plan.steps)} steps")
            dataset = _apply_plan(
                opts.plan, dataset, progress_callback=progress_callback
            )

        # Build MSM if requested
        msm_result = None
        if opts.msm_mode != "none":
            logger.info("Building MSM...")
            msm_result = _build_msm(dataset, opts, applied)

        # Build FES if requested
        fes_result = None
        if opts.enable_fes:
            logger.info("Building FES...")
            fes_result = _build_fes(dataset, opts, applied)

        # Build TRAM if requested
        tram_result = None
        if opts.enable_tram:
            logger.info("Building TRAM...")
            tram_result = _build_tram(dataset, opts, applied)

        # Update metadata
        from datetime import datetime

        metadata.end_time = datetime.now().isoformat()
        metadata.success = True

        # Count frames and features
        n_frames = _count_frames(dataset)
        feature_names = _extract_feature_names(dataset)

        return BuildResult(
            msm=msm_result,
            fes=fes_result,
            tram=tram_result,
            metadata=metadata,
            applied_opts=applied,
            n_frames=n_frames,
            n_shards=len(applied.selected_shards),
            feature_names=feature_names,
        )

    except Exception as e:
        logger.error(f"Build failed: {e}")
        metadata.error_message = str(e)
        metadata.success = False
        raise


def _generate_run_id() -> str:
    """Generate a unique run ID."""
    import time

    return f"build_{int(time.time())}_{os.getpid()}"


def _count_frames(dataset: Any) -> int:
    """Count the total number of frames in the dataset."""
    try:
        if hasattr(dataset, "__len__"):
            return len(dataset)
        if hasattr(dataset, "n_frames"):
            return dataset.n_frames
        return 0
    except Exception:
        return 0


def _extract_feature_names(dataset: Any) -> List[str]:
    """Extract feature names from the dataset."""
    try:
        if hasattr(dataset, "feature_names"):
            return list(dataset.feature_names)
        if hasattr(dataset, "columns"):
            return list(dataset.columns)
        return []
    except Exception:
        return []


def _build_msm(dataset: Any, opts: BuildOpts, applied: AppliedOpts) -> Any:
    """Build a Markov state model."""
    try:
        return build_simple_msm(
            dataset,
            n_clusters=opts.n_clusters,
            n_states=opts.n_states,
            lag_time=opts.lag_time,
        )
    except Exception as e:
        logger.warning(f"MSM build failed: {e}")
        return None


def _build_fes(dataset: Any, opts: BuildOpts, applied: AppliedOpts) -> Any:
    """Build a free energy surface."""
    try:
        return default_fes_builder(dataset, opts, applied)
    except Exception as e:
        logger.warning(f"FES build failed: {e}")
        return None


def _build_tram(dataset: Any, opts: BuildOpts, applied: AppliedOpts) -> Any:
    """Build TRAM analysis."""
    try:
        return default_tram_builder(dataset, opts, applied)
    except Exception as e:
        logger.warning(f"TRAM build failed: {e}")
        return None


# --- Default builders ---------------------------------------------------------


def default_fes_builder(
    dataset: Any, opts: BuildOpts, applied: AppliedOpts
) -> Any | None:
    """Default FES builder using pmarlo.markov_state_model.free_energy.generate_2d_fes.

    Returns either a payload with a "result" field or a dict indicating a
    skipped reason. Callers may treat non-dict/None as "no-op".
    """
    cv_pair = _extract_cvs(dataset)
    if cv_pair is None:
        return {"skipped": True, "reason": "no_cvs"}
    from pmarlo.markov_state_model.free_energy import generate_2d_fes

    cv1, cv2, names, periodic = cv_pair
    # Guardrails: only attempt FES if data are finite and have non-zero extent
    try:
        a = np.asarray(cv1, dtype=float).reshape(-1)
        b = np.asarray(cv2, dtype=float).reshape(-1)
        if a.size == 0 or b.size == 0:
            return {"skipped": True, "reason": "empty_cvs"}
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            return {"skipped": True, "reason": "non_finite_cvs"}
        a_range = float(np.ptp(a))
        b_range = float(np.ptp(b))
        if a_range <= 1e-12 or b_range <= 1e-12:
            return {"skipped": True, "reason": "zero_variance_cvs"}
    except Exception:
        return {"skipped": True, "reason": "cv_validation_error"}

    try:
        fes = generate_2d_fes(
            cv1,
            cv2,
            temperature=opts.fes_temperature,
            periodic=periodic,
        )
        return {"result": fes, "cv1_name": names[0], "cv2_name": names[1]}
    except Exception as e:
        logger.warning(f"FES generation failed: {e}")
        return {"skipped": True, "reason": f"fes_error: {e}"}


def default_tram_builder(
    dataset: Any, opts: BuildOpts, applied: AppliedOpts
) -> Any | None:
    """Default TRAM builder (placeholder implementation)."""
    logger.info("TRAM builder not yet implemented")
    return {"skipped": True, "reason": "not_implemented"}


def _extract_cvs(
    dataset: Any,
) -> Optional[Tuple[np.ndarray, np.ndarray, List[str], Tuple[bool, bool]]]:
    """Extract a pair of collective variables from the dataset.

    Returns (cv1, cv2, names, periodic) or None if no suitable pair found.
    """
    try:
        # Try to extract CV data from various dataset formats
        if hasattr(dataset, "get_cvs"):
            cvs = dataset.get_cvs()
            if len(cvs) >= 2:
                cv1, cv2 = cvs[0], cvs[1]
                names = [f"CV{i+1}" for i in range(2)]
                periodic = (False, False)  # Default to non-periodic
                return cv1, cv2, names, periodic

        # Try common attribute names
        for attr in ["features", "data", "X"]:
            if hasattr(dataset, attr):
                data = getattr(dataset, attr)
                if hasattr(data, "shape") and len(data.shape) >= 2 and data.shape[1] >= 2:
                    cv1, cv2 = data[:, 0], data[:, 1]
                    names = [f"Feature{i+1}" for i in range(2)]
                    periodic = (False, False)
                    return cv1, cv2, names, periodic

        # Try dictionary-like access
        if hasattr(dataset, "keys") or isinstance(dataset, dict):
            keys = list(dataset.keys()) if hasattr(dataset, "keys") else list(dataset.keys())
            if len(keys) >= 2:
                cv1 = dataset[keys[0]]
                cv2 = dataset[keys[1]]
                names = [str(keys[0]), str(keys[1])]
                periodic = (False, False)
                return cv1, cv2, names, periodic

        return None
    except Exception:
        return None


# --- Utility functions --------------------------------------------------------


def validate_build_opts(opts: BuildOpts) -> List[str]:
    """Validate build options and return a list of warnings."""
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


def estimate_memory_usage(dataset: Any, opts: BuildOpts) -> float:
    """Estimate memory usage in GB for the build operation."""
    try:
        n_frames = _count_frames(dataset)
        n_features = len(_extract_feature_names(dataset))

        # Base dataset memory (assuming float64)
        dataset_gb = (n_frames * n_features * 8) / (1024**3)

        # MSM memory (clusters and states)
        msm_gb = (opts.n_clusters * n_features * 8) / (1024**3)
        msm_gb += (opts.n_states * opts.n_states * 8) / (1024**3)

        # FES memory (assuming 100x100 grid)
        fes_gb = (100 * 100 * 8) / (1024**3) if opts.enable_fes else 0

        # Safety margin
        total_gb = (dataset_gb + msm_gb + fes_gb) * 1.5

        return total_gb
    except Exception:
        return 1.0  # Default estimate


def create_build_summary(result: BuildResult) -> Dict[str, Any]:
    """Create a summary dictionary from BuildResult."""
    summary = {
        "success": result.metadata.success if result.metadata else False,
        "n_frames": result.n_frames,
        "n_shards": result.n_shards,
        "n_features": len(result.feature_names),
        "has_msm": result.msm is not None,
        "has_fes": result.fes is not None,
        "has_tram": result.tram is not None,
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
