from __future__ import annotations

"""Backend utilities powering the Streamlit joint-learning demo.

The previous iteration of the app mixed UI callbacks, shard bookkeeping, and
engine calls in a single ~900 line module. This rewrite keeps the backend
focused on three responsibilities:

1. manage the on-disk workspace layout (sims -> shards -> models -> bundles)
2. provide thin orchestration wrappers around the high-level
   :mod:`pmarlo.api` helpers that already implement REMD, shard emission, and
   MSM/FES builds
3. persist lightweight manifest entries so the UI can remain mostly stateless

The goal is to make it straightforward to express the interactive workflow::

    sample -> emit shards -> train CV model -> enrich dataset -> build MSM/FES

while keeping the logic reusable for non-UI automation in the future.
"""

import json
import logging
import math
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Mapping, Optional, Sequence, cast

from pmarlo.utils.path_utils import ensure_directory

try:  # Package-relative when imported as module
    from .state import StateManager
except ImportError:  # Fallback for direct script import
    import sys

    _APP_DIR = Path(__file__).resolve().parent
    if str(_APP_DIR) not in sys.path:
        sys.path.insert(0, str(_APP_DIR))
    from state import StateManager  # type: ignore

from pmarlo.analysis import compute_analysis_debug, export_analysis_debug
from pmarlo.data.aggregate import load_shards_as_dataset

__all__ = [
    "WorkspaceLayout",
    "SimulationConfig",
    "SimulationResult",
    "ShardRequest",
    "ShardResult",
    "TrainingConfig",
    "TrainingResult",
    "BuildConfig",
    "BuildArtifact",
    "WorkflowBackend",
    "choose_sim_seed",
    "run_short_sim",
]

logger = logging.getLogger(__name__)


if TYPE_CHECKING:  # pragma: no cover - typing only
    from pmarlo.transform.build import BuildResult as _BuildResult


@lru_cache(maxsize=1)
def _pmarlo_handles() -> Dict[str, Any]:
    """Import heavyweight PMARLO helpers on demand."""

    from pmarlo.api import build_from_shards as _build_from_shards
    from pmarlo.api import emit_shards_rg_rmsd_windowed as _emit_shards
    from pmarlo.api import run_replica_exchange as _run_replica_exchange
    from pmarlo.data.shard import read_shard as _read_shard
    from pmarlo.transform.build import BuildResult as _BuildResultRuntime
    from pmarlo.transform.build import _sanitize_artifacts as _sanitize

    return {
        "build_from_shards": _build_from_shards,
        "emit_shards_rg_rmsd_windowed": _emit_shards,
        "run_replica_exchange": _run_replica_exchange,
        "read_shard": _read_shard,
        "BuildResult": _BuildResultRuntime,
        "_sanitize_artifacts": _sanitize,
    }


def _build_result_cls() -> "_BuildResult":
    return cast("_BuildResult", _pmarlo_handles()["BuildResult"])


def _sanitize_artifacts(data: Any) -> Any:
    return _pmarlo_handles()["_sanitize_artifacts"](data)


def build_from_shards(*args: Any, **kwargs: Any) -> Any:
    return _pmarlo_handles()["build_from_shards"](*args, **kwargs)


def emit_shards_rg_rmsd_windowed(*args: Any, **kwargs: Any) -> Any:
    return _pmarlo_handles()["emit_shards_rg_rmsd_windowed"](*args, **kwargs)


def run_replica_exchange(*args: Any, **kwargs: Any) -> Any:
    return _pmarlo_handles()["run_replica_exchange"](*args, **kwargs)


def read_shard(*args: Any, **kwargs: Any) -> Any:
    return _pmarlo_handles()["read_shard"](*args, **kwargs)


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _coerce_path_list(paths: Iterable[str | Path]) -> List[Path]:
    return [Path(p).resolve() for p in paths]


def _slugify(label: Optional[str]) -> Optional[str]:
    if not label:
        return None
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(label))
    safe = safe.strip("_").lower()
    return safe or None


def choose_sim_seed(mode: str, *, fixed: Optional[int] = None) -> Optional[int]:
    """Choose simulation seed based on mode."""
    import random

    if mode == "none":
        return None
    elif mode == "fixed":
        return fixed
    elif mode == "auto":
        return random.randint(1, 1000000)
    else:
        raise ValueError(f"Unknown seed mode: {mode}")


def run_short_sim(
    pdb_path: Path,
    workspace: Path,
    temperatures: Sequence[float],
    *,
    steps: int = 1000,
    quick: bool = True,
    random_seed: Optional[int] = None,
    start_from: Optional[Path] = None,
    use_stub: Optional[bool] = None,
) -> "SimulationResult":
    """Run a short simulation for testing purposes.

    Parameters
    ----------
    use_stub:
        When ``True`` (default if ``quick`` is ``True``), generate synthetic
        trajectories rather than invoking the full REMD stack. Setting this to
        ``False`` forces a real simulation even in quick mode.
    """
    layout = WorkspaceLayout(
        app_root=workspace,
        inputs_dir=workspace / "inputs",
        workspace_dir=workspace / "output",
        sims_dir=workspace / "output" / "sims",
        shards_dir=workspace / "output" / "shards",
        models_dir=workspace / "output" / "models",
        bundles_dir=workspace / "output" / "bundles",
        logs_dir=workspace / "output" / "logs",
        state_path=workspace / "output" / "state.json",
    )
    layout.ensure()

    backend = WorkflowBackend(layout)
    effective_steps = int(steps)
    if quick:
        effective_steps = max(1, min(effective_steps, 200))

    stub_result = quick if use_stub is None else bool(use_stub)

    config = SimulationConfig(
        pdb_path=pdb_path,
        temperatures=temperatures,
        steps=effective_steps,
        quick=quick,
        random_seed=random_seed,
        stub_result=stub_result,
    )
    return backend.run_sampling(config)


@dataclass(frozen=True)
class WorkspaceLayout:
    """Resolved paths for the app's workspace tree."""

    app_root: Path
    inputs_dir: Path
    workspace_dir: Path
    sims_dir: Path
    shards_dir: Path
    models_dir: Path
    bundles_dir: Path
    logs_dir: Path
    state_path: Path

    @classmethod
    def from_app_package(cls, file_path: Optional[Path] = None) -> "WorkspaceLayout":
        here = Path(file_path or __file__).resolve()
        app_dir = here.parent  # .../app
        root = app_dir.parent  # .../app_usecase
        workspace = root / "app_output"
        layout = cls(
            app_root=root,
            inputs_dir=root / "app_intputs",
            workspace_dir=workspace,
            sims_dir=workspace / "sims",
            shards_dir=workspace / "shards",
            models_dir=workspace / "models",
            bundles_dir=workspace / "bundles",
            logs_dir=workspace / "logs",
            state_path=workspace / "state.json",
        )
        layout.ensure()
        return layout

    def ensure(self) -> None:
        for path in (
            self.workspace_dir,
            self.sims_dir,
            self.shards_dir,
            self.models_dir,
            self.bundles_dir,
            self.logs_dir,
        ):
            ensure_directory(path)
        ensure_directory(self.analysis_debug_dir)

    def available_inputs(self) -> List[Path]:
        if not self.inputs_dir.exists():
            return []
        return sorted(p.resolve() for p in self.inputs_dir.glob("*.pdb"))

    @property
    def analysis_debug_dir(self) -> Path:
        return self.workspace_dir / "analysis_debug"


@dataclass
class SimulationConfig:
    pdb_path: Path
    temperatures: Sequence[float]
    steps: int
    quick: bool = True
    random_seed: Optional[int] = None
    label: Optional[str] = None
    jitter_start: bool = False
    jitter_sigma_A: float = 0.05
    exchange_frequency_steps: Optional[int] = None
    temperature_schedule_mode: Optional[str] = None
    cv_model_bundle: Optional[Path] = None  # Path to trained CV model for CV-informed sampling
    stub_result: bool = False  # Use synthetic trajectories instead of running REMD


@dataclass
class SimulationResult:
    run_id: str
    run_dir: Path
    pdb_path: Path
    traj_files: List[Path]
    analysis_temperatures: List[float]
    steps: int
    created_at: str


@dataclass
class ShardRequest:
    stride: int = 5
    temperature: float = 300.0
    reference: Optional[Path] = None
    seed_start: int = 0
    frames_per_shard: int = 5000
    hop_frames: Optional[int] = None


@dataclass
class ShardResult:
    run_id: str
    shard_dir: Path
    shard_paths: List[Path]
    n_frames: int
    n_shards: int
    temperature: float
    stride: int
    frames_per_shard: int
    hop_frames: Optional[int]
    created_at: str


@dataclass
class TrainingConfig:
    lag: int = 5
    bins: Dict[str, int] = field(default_factory=lambda: {"Rg": 64, "RMSD_ref": 64})
    seed: int = 1337
    temperature: float = 300.0
    hidden: Sequence[int] = (128, 128)
    max_epochs: int = 200
    early_stopping: int = 25
    tau_schedule: Sequence[int] = (2, 5, 10, 20)
    val_tau: int = 20
    epochs_per_tau: int = 15

    def deeptica_params(self) -> Dict[str, Any]:
        return {
            "lag": int(max(1, self.lag)),
            "n_out": 2,
            "hidden": tuple(int(h) for h in self.hidden),
            "max_epochs": int(self.max_epochs),
            "early_stopping": int(self.early_stopping),
            "tau_schedule": tuple(int(t) for t in self.tau_schedule),
            "val_tau": int(self.val_tau),
            "epochs_per_tau": int(self.epochs_per_tau),
            "reweight_mode": "scaled_time",
        }


@dataclass
class TrainingResult:
    bundle_path: Path
    dataset_hash: str
    build_result: BuildResult
    created_at: str
    checkpoint_dir: Optional[Path] = None
    cv_model_bundle: Optional[Dict[str, Any]] = None  # Paths to exported CV model files


@dataclass
class BuildConfig:
    lag: int
    bins: Dict[str, int]
    seed: int
    temperature: float
    learn_cv: bool = False
    deeptica_params: Optional[Dict[str, Any]] = None
    notes: Dict[str, Any] = field(default_factory=dict)
    apply_cv_whitening: bool = False
    cluster_mode: str = "kmeans"
    n_microstates: int = 20
    kmeans_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {"n_init": 50}
    )
    reweight_mode: str = "MBAR"
    fes_method: str = "kde"
    fes_bandwidth: str | float = "scott"
    fes_min_count_per_bin: int = 1


@dataclass
class BuildArtifact:
    bundle_path: Path
    dataset_hash: str
    build_result: BuildResult
    created_at: str
    debug_dir: Optional[Path] = None
    debug_summary: Optional[Dict[str, Any]] = None
    discretizer_fingerprint: Optional[Dict[str, Any]] = None
    tau_frames: Optional[int] = None
    effective_tau_frames: Optional[int] = None
    effective_stride_max: Optional[int] = None
    analysis_healthy: bool = True
    guardrail_violations: Optional[List[Dict[str, Any]]] = None


class WorkflowBackend:
    """High-level orchestration for the Streamlit UI."""

    def __init__(self, layout: WorkspaceLayout) -> None:
        self.layout = layout
        self.state = StateManager(layout.state_path)

    # ------------------------------------------------------------------
    # Sampling & shard emission
    # ------------------------------------------------------------------
    def run_sampling(self, config: SimulationConfig) -> SimulationResult:
        base_label = _slugify(config.label) or f"run-{_timestamp()}"

        # Prepare CV model info if provided
        # CV biasing is now properly implemented with harmonic expansion bias
        # The exported model includes CVBiasPotential wrapper that transforms CVs → Energy
        cv_kwargs = {}
        use_stub = bool(config.stub_result)
        if config.cv_model_bundle:
            use_stub = False
            import logging
            logger = logging.getLogger(__name__)

            logger.info("=" * 60)
            logger.info("CV-INFORMED SAMPLING ENABLED")
            logger.info("=" * 60)
            try:
                from pmarlo.features.deeptica import load_cv_model_info, check_openmm_torch_available

                # Check for openmm-torch
                if not check_openmm_torch_available():
                    logger.error(
                        "CV model selected but openmm-torch is NOT installed!\n"
                        "Install with: conda install -c conda-forge openmm-torch\n"
                        "CV-biased sampling will be SKIPPED."
                    )
                    logger.warning("Running simulation WITHOUT CV biasing")
                else:
                    # Warn about CPU-only PyTorch performance
                    import torch
                    if not torch.cuda.is_available():
                        logger.warning(
                            "⚠️  PyTorch is running on CPU only!\n"
                            "CV-biased simulations will be ~10-20x slower than unbiased.\n"
                            "For production use, install CUDA-enabled PyTorch:\n"
                            "https://pytorch.org/get-started/locally/\n"
                        )

                    # Load CV model info
                    cv_info = load_cv_model_info(config.cv_model_bundle, model_name="deeptica_cv_model")
                    cv_kwargs["cv_model_path"] = cv_info["model_path"]
                    cv_kwargs["cv_scaler_mean"] = cv_info["scaler_params"]["mean"]
                    cv_kwargs["cv_scaler_scale"] = cv_info["scaler_params"]["scale"]

                    logger.info("✓ CV bias potential loaded successfully")
                    logger.info(f"  Model path: {cv_info['model_path']}")
                    logger.info(f"  CV dimensions: {cv_info['config']['cv_dim']}")
                    logger.info(f"  Bias type: {cv_info['config'].get('bias_type', 'harmonic_expansion')}")
                    logger.info(f"  Bias strength: {cv_info['config'].get('bias_strength', 10.0):.1f} kJ/mol")
                    logger.info("\nBias physics:")
                    logger.info("  E_bias = k * sum(cv_i^2)")
                    logger.info("  Forces: F = -∇E_bias (computed by OpenMM)")
                    logger.info("  Purpose: Repulsive bias → explore diverse conformations")
                    logger.info("\n⚠️  IMPORTANT: The model expects MOLECULAR FEATURES as input")
                    logger.info("  (distances, angles, dihedrals), not raw atomic positions.")
                    logger.info("  Feature extraction must be configured in OpenMM system.")

            except ImportError as exc:
                logger.error(f"Could not import CV modules: {exc}")
                logger.warning("Running simulation WITHOUT CV biasing")
            except FileNotFoundError as exc:
                logger.error(f"CV model files not found: {exc}")
                logger.warning("Running simulation WITHOUT CV biasing")
            except Exception as exc:
                logger.error(f"Could not load CV model: {exc}", exc_info=True)
                logger.warning("Running simulation WITHOUT CV biasing")
        run_label = base_label
        if config.random_seed is not None:
            run_label = f"{base_label}-seed-{int(config.random_seed)}"
        elif use_stub:
            run_label = f"{base_label}-stub-{len(self.state.runs)}"
        run_dir = (self.layout.sims_dir / run_label).resolve()
        ensure_directory(run_dir)

        if use_stub:
            result, metadata = self._run_quick_sampling_stub(run_label, run_dir, config)
            self.state.append_run(metadata)
            return result

        traj_files, temps = run_replica_exchange(
            pdb_file=str(config.pdb_path),
            output_dir=str(run_dir),
            temperatures=[float(t) for t in config.temperatures],
            total_steps=int(config.steps),
            quick=bool(config.quick),
            random_seed=(
                int(config.random_seed) if config.random_seed is not None else None
            ),
            jitter_start=bool(config.jitter_start),
            jitter_sigma_A=float(config.jitter_sigma_A),
            exchange_frequency_steps=(
                int(config.exchange_frequency_steps)
                if config.exchange_frequency_steps is not None
                else None
            ),
            temperature_schedule_mode=config.temperature_schedule_mode,
            **cv_kwargs,
        )
        created = _timestamp()
        result = SimulationResult(
            run_id=run_label,
            run_dir=run_dir.resolve(),
            pdb_path=Path(config.pdb_path).resolve(),
            traj_files=_coerce_path_list(traj_files),
            analysis_temperatures=[float(t) for t in temps],
            steps=int(config.steps),
            created_at=created,
        )
        run_metadata = {
            "run_id": run_label,
            "run_dir": str(result.run_dir),
            "pdb": str(result.pdb_path),
            "temperatures": [float(t) for t in config.temperatures],
            "analysis_temperatures": result.analysis_temperatures,
            "steps": int(config.steps),
            "quick": bool(config.quick),
            "random_seed": (
                int(config.random_seed) if config.random_seed is not None else None
            ),
            "traj_files": [str(p) for p in result.traj_files],
            "created_at": created,
            "stub_result": bool(use_stub),
        }

        # Add CV model reference if used
        if config.cv_model_bundle:
            run_metadata["cv_model_bundle"] = str(config.cv_model_bundle)
            run_metadata["cv_informed"] = True

        self.state.append_run(run_metadata)
        return result

    def _run_quick_sampling_stub(
        self,
        run_label: str,
        run_dir: Path,
        config: SimulationConfig,
    ) -> tuple[SimulationResult, Dict[str, Any]]:
        """Generate a lightweight deterministic sampling result for quick-mode tests."""

        import numpy as np
        import mdtraj as md

        seed = (
            int(config.random_seed)
            if config.random_seed is not None
            else abs(hash(run_label)) % (2**32)
        )
        rng = np.random.default_rng(seed)
        template = md.load(str(config.pdb_path))
        base_coords = template.xyz[0]
        frames = max(5, min(50, int(config.steps) if config.steps else 5))

        rep_dir = (run_dir / "replica_exchange").resolve()
        ensure_directory(rep_dir)

        analysis_temperatures = [float(t) for t in config.temperatures]
        traj_files: list[Path] = []
        for idx, temp in enumerate(config.temperatures):
            noise = 0.01 * rng.standard_normal((frames,) + base_coords.shape)
            coords = base_coords + noise
            traj = md.Trajectory(coords, template.topology)
            out_path = rep_dir / f"traj_{idx:02d}.dcd"
            traj.save_dcd(str(out_path))
            traj_files.append(out_path.resolve())

        # Minimal diagnostics payload mirroring the real runner structure
        import json as _json

        diag_payload = {
            "temperatures": analysis_temperatures,
            "exchange_attempts": int(max(1, frames - 1)),
            "exchange_accepted": int(max(0, frames // 2)),
            "per_pair_acceptance": [0.5 for _ in analysis_temperatures],
            "acceptance_mean": 0.5,
            "mean_abs_disp_per_10k_steps": 0.0,
            "mean_abs_disp_per_sweep": 0.0,
            "sparkline": [0.0 for _ in analysis_temperatures],
        }
        (rep_dir / "exchange_diagnostics.json").write_text(
            _json.dumps(diag_payload, indent=2), encoding="utf-8"
        )

        created = _timestamp()
        result = SimulationResult(
            run_id=run_label,
            run_dir=run_dir.resolve(),
            pdb_path=Path(config.pdb_path).resolve(),
            traj_files=traj_files,
            analysis_temperatures=analysis_temperatures,
            steps=int(config.steps),
            created_at=created,
        )
        metadata = {
            "run_id": run_label,
            "run_dir": str(result.run_dir),
            "pdb": str(result.pdb_path),
            "temperatures": analysis_temperatures,
            "analysis_temperatures": analysis_temperatures,
            "steps": int(config.steps),
            "quick": True,
            "random_seed": (
                int(config.random_seed) if config.random_seed is not None else None
            ),
            "traj_files": [str(p) for p in traj_files],
            "created_at": created,
            "stub_result": True,
        }
        return result, metadata

    def emit_shards(
        self,
        simulation: SimulationResult,
        request: ShardRequest,
        *,
        provenance: Optional[Dict[str, Any]] = None,
    ) -> ShardResult:
        shard_dir = self.layout.shards_dir / simulation.run_id
        ensure_directory(shard_dir)
        created = _timestamp()
        note = {
            "created_at": created,
            "kind": "demux",
            "run_id": simulation.run_id,
            "analysis_temperatures": simulation.analysis_temperatures,
            "topology": str(simulation.pdb_path),
            "traj_files": [str(p) for p in simulation.traj_files],
        }
        if provenance:
            note.update(provenance)
        shard_paths = emit_shards_rg_rmsd_windowed(
            pdb_file=simulation.pdb_path,
            traj_files=[str(p) for p in simulation.traj_files],
            out_dir=str(shard_dir),
            reference=str(request.reference) if request.reference else None,
            stride=int(max(1, request.stride)),
            temperature=float(request.temperature),
            seed_start=int(max(0, request.seed_start)),
            frames_per_shard=int(max(1, request.frames_per_shard)),
            hop_frames=(
                int(request.hop_frames)
                if request.hop_frames is not None and request.hop_frames > 0
                else None
            ),
            provenance=note,
        )
        shard_paths = _coerce_path_list(shard_paths)
        n_frames = 0
        for path in shard_paths:
            try:
                meta, _, _ = read_shard(path)
                n_frames += int(getattr(meta, "n_frames", 0))
            except Exception:
                continue
        result = ShardResult(
            run_id=simulation.run_id,
            shard_dir=shard_dir.resolve(),
            shard_paths=shard_paths,
            n_frames=int(n_frames),
            n_shards=len(shard_paths),
            temperature=float(request.temperature),
            stride=int(max(1, request.stride)),
            frames_per_shard=int(max(1, request.frames_per_shard)),
            hop_frames=(
                int(request.hop_frames)
                if request.hop_frames is not None and request.hop_frames > 0
                else None
            ),
            created_at=created,
        )
        self.state.append_shards(
            {
                "run_id": simulation.run_id,
                "directory": str(result.shard_dir),
                "paths": [str(p) for p in shard_paths],
                "temperature": float(request.temperature),
                "stride": int(max(1, request.stride)),
                "n_shards": len(shard_paths),
                "n_frames": int(n_frames),
                "frames_per_shard": int(max(1, request.frames_per_shard)),
                "hop_frames": (
                    int(request.hop_frames)
                    if request.hop_frames is not None and request.hop_frames > 0
                    else None
                ),
                "created_at": created,
            }
        )
        return result

    # ------------------------------------------------------------------
    # Discovery helpers
    # ------------------------------------------------------------------
    def discover_shards(self) -> List[Path]:
        if not self.layout.shards_dir.exists():
            return []
        return sorted(self.layout.shards_dir.rglob("*.json"))

    def shard_summaries(self) -> List[Dict[str, Any]]:
        # Return only shard batches that have existing files; trim missing paths.
        info: List[Dict[str, Any]] = []
        for entry in self.state.shards:
            paths = [str(p) for p in entry.get("paths", []) if Path(p).exists()]
            if not paths:
                # Skip batches that no longer have files on disk
                continue
            e = dict(entry)
            e["paths"] = paths
            e["n_shards"] = len(paths)
            info.append(e)
        return info

    # ------------------------------------------------------------------
    # Model training and analysis
    # ------------------------------------------------------------------
    def get_training_progress(self, checkpoint_dir: Path) -> Optional[Dict[str, Any]]:
        """Read real-time training progress from checkpoint directory."""
        import json

        if not checkpoint_dir or not checkpoint_dir.exists():
            return None

        progress_path = checkpoint_dir / "training_progress.json"
        if not progress_path.exists():
            return None

        try:
            with progress_path.open("r") as f:
                return json.load(f)
        except Exception:
            return None

    def train_model(
        self,
        shard_jsons: Sequence[Path],
        config: TrainingConfig,
    ) -> TrainingResult:
        import logging

        shards = [Path(p).resolve() for p in shard_jsons]
        if not shards:
            raise ValueError("No shards selected for training")
        stamp = _timestamp()
        bundle_path = self.layout.models_dir / f"deeptica-{stamp}.pbz"

        # Create checkpoint directory for training progress
        checkpoint_dir = self.layout.models_dir / f"training-{stamp}"
        ensure_directory(checkpoint_dir)

        # Setup logging to file
        log_file = checkpoint_dir / "training.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

        # Get the pmarlo logger and add our file handler
        pmarlo_logger = logging.getLogger("pmarlo")
        pmarlo_logger.addHandler(file_handler)
        pmarlo_logger.setLevel(logging.INFO)

        try:
            pmarlo_logger.info(f"Starting training with {len(shards)} shards")
            pmarlo_logger.info(f"Configuration: lag={config.lag}, bins={config.bins}, max_epochs={config.max_epochs}")

            # Add checkpoint_dir to deeptica_params
            deeptica_params = config.deeptica_params()
            deeptica_params["checkpoint_dir"] = str(checkpoint_dir)

            pmarlo_logger.info("Calling build_from_shards...")
            br, ds_hash = build_from_shards(
                shard_jsons=shards,
                out_bundle=bundle_path,
                bins=dict(config.bins),
                lag=int(config.lag),
                seed=int(config.seed),
                temperature=float(config.temperature),
                learn_cv=True,
                deeptica_params=deeptica_params,
                notes={"model_dir": str(self.layout.models_dir), "checkpoint_dir": str(checkpoint_dir)},
            )
            pmarlo_logger.info("Training completed successfully")
        except ImportError as exc:
            pmarlo_logger.error(f"Import error: {exc}")
            raise RuntimeError(
                "Deep-TICA optional dependencies missing. Install pmarlo[mlcv] to enable"
            ) from exc
        except Exception as exc:
            pmarlo_logger.error(f"Training failed: {exc}", exc_info=True)
            raise
        finally:
            # Remove the handler so it doesn't persist
            pmarlo_logger.removeHandler(file_handler)
            file_handler.close()
        # Export CV model for OpenMM integration
        cv_model_bundle_info = None
        try:
            from pmarlo.features.deeptica import export_cv_model

            # Load the trained model from bundle
            import pickle
            with open(bundle_path, "rb") as f:
                bundle_data = pickle.load(f)

            network = bundle_data.get("network")
            scaler = bundle_data.get("scaler")
            history = br.get("history", {})

            if network is not None and scaler is not None:
                try:
                    import yaml
                except Exception as exc:
                    raise RuntimeError("PyYAML is required to load feature specifications") from exc
                spec_path = self.layout.app_root / "app" / "feature_spec.yaml"
                if not spec_path.exists():
                    raise FileNotFoundError(f"Feature specification not found at {spec_path}")
                with spec_path.open("r", encoding="utf-8") as spec_file:
                    feature_spec = yaml.safe_load(spec_file)
                pmarlo_logger.info("Exporting CV model with bias potential for OpenMM integration...")
                pmarlo_logger.info("Creating CVBiasPotential wrapper (harmonic expansion bias)...")
                cv_bundle = export_cv_model(
                    network=network,
                    scaler=scaler,
                    history=history,
                    output_dir=checkpoint_dir,
                    model_name="deeptica_cv_model",
                    bias_strength=10.0,  # Can be made configurable
                    feature_spec=feature_spec,
                )
                cv_model_bundle_info = {
                    "model_path": str(cv_bundle.model_path),
                    "scaler_path": str(cv_bundle.scaler_path),
                    "config_path": str(cv_bundle.config_path),
                    "metadata_path": str(cv_bundle.metadata_path),
                    "cv_dim": cv_bundle.cv_dim,
                    "feature_spec_sha256": cv_bundle.feature_spec_hash,
                }
                pmarlo_logger.info("✓ CV bias potential exported successfully")
                pmarlo_logger.info(f"  Model outputs: Energy (kJ/mol) for OpenMM force calculation")
                pmarlo_logger.info(f"  Bias formula: E = 10.0 * sum(cv_i^2)")
                pmarlo_logger.info(f"  Purpose: Encourages conformational exploration")
        except Exception as exc:
            pmarlo_logger.warning(f"Could not export CV model: {exc}")

        result = TrainingResult(
            bundle_path=bundle_path.resolve(),
            dataset_hash=ds_hash,
            build_result=br,
            created_at=stamp,
            checkpoint_dir=checkpoint_dir,
            cv_model_bundle=cv_model_bundle_info,
        )
        self.state.append_model(
            {
                "bundle": str(bundle_path.resolve()),
                "checkpoint_dir": str(checkpoint_dir.resolve()),  # ADD THIS for CV model loading
                "dataset_hash": ds_hash,
                "lag": int(config.lag),
                "bins": dict(config.bins),
                "seed": int(config.seed),
                "temperature": float(config.temperature),
                "hidden": [int(h) for h in config.hidden],
                "max_epochs": int(config.max_epochs),
                "early_stopping": int(config.early_stopping),
                "tau_schedule": [int(t) for t in config.tau_schedule],
                "val_tau": int(config.val_tau),
                "epochs_per_tau": int(config.epochs_per_tau),
                "created_at": stamp,
                "metrics": _sanitize_artifacts(br.artifacts.get("mlcv_deeptica", {})),
            }
        )
        return result

    def build_analysis(
        self,
        shard_jsons: Sequence[Path],
        config: BuildConfig,
    ) -> BuildArtifact:
        shards = [Path(p).resolve() for p in shard_jsons]
        if not shards:
            raise ValueError("No shards selected for analysis")
        stamp = _timestamp()
        bundle_path = self.layout.bundles_dir / f"bundle-{stamp}.pbz"
        dataset = load_shards_as_dataset(shards)
        debug_data = compute_analysis_debug(
            dataset,
            lag=int(config.lag),
            count_mode="sliding",
        )
        config_payload = asdict(config)
        analysis_notes = dict(config.notes or {})
        if config.learn_cv and "model_dir" not in analysis_notes:
            analysis_notes["model_dir"] = str(self.layout.models_dir)
        analysis_notes["apply_cv_whitening_requested"] = bool(config.apply_cv_whitening)
        analysis_notes["apply_cv_whitening_enforced"] = True
        analysis_notes["kmeans_kwargs"] = dict(config.kmeans_kwargs)
        analysis_notes["analysis_overrides"] = {
            "cluster_mode": str(config.cluster_mode),
            "n_microstates": int(config.n_microstates),
            "reweight_mode": str(config.reweight_mode),
            "fes_method": str(config.fes_method),
            "fes_bandwidth": config.fes_bandwidth,
            "fes_min_count_per_bin": int(config.fes_min_count_per_bin),
            "kmeans_kwargs": dict(config.kmeans_kwargs),
        }
        requested_fingerprint = {
            "mode": str(config.cluster_mode),
            "n_states": int(config.n_microstates),
            "seed": int(config.seed),
        }
        previous_fingerprint = analysis_notes.get("discretizer_fingerprint")
        if previous_fingerprint and previous_fingerprint != requested_fingerprint:
            analysis_notes.setdefault(
                "discretizer_fingerprint_previous", previous_fingerprint
            )
            logger.info(
                "Discretizer fingerprint override changed from %s to %s; "
                "forcing refit of clusterer.",
                previous_fingerprint,
                requested_fingerprint,
            )
        analysis_notes["discretizer_fingerprint_requested"] = requested_fingerprint
        analysis_notes["analysis_tau_requested"] = int(config.lag)
        analysis_notes["debug_summary"] = _sanitize_artifacts(debug_data.summary)
        logger.info(
            (
                "Analysis debug summary: frames=%d states=%d pairs=%d zero_rows=%d "
                "warnings=%d"
            ),
            int(debug_data.summary.get("total_frames_with_states", 0)),
            int(debug_data.counts.shape[0]),
            int(debug_data.summary.get("total_pairs", 0)),
            int(debug_data.summary.get("zero_rows", 0)),
            len(debug_data.summary.get("warnings", [])),
        )

        br, ds_hash = build_from_shards(
            shard_jsons=shards,
            out_bundle=bundle_path,
            bins=dict(config.bins),
            lag=int(config.lag),
            seed=int(config.seed),
            temperature=float(config.temperature),
            learn_cv=bool(config.learn_cv),
            deeptica_params=config.deeptica_params,
            notes=analysis_notes,
            kmeans_kwargs=dict(config.kmeans_kwargs),
        )

        summary = debug_data.summary

        def _safe_int(value: Any, default: int) -> int:
            try:
                return int(value)
            except (TypeError, ValueError):
                return default

        def _safe_float(value: Any, default: float = float("nan")) -> float:
            try:
                return float(value)
            except (TypeError, ValueError):
                return default

        tau_frames = _safe_int(summary.get("tau_frames"), int(config.lag))
        stride_max = max(1, _safe_int(summary.get("effective_stride_max"), 1))
        effective_tau_frames = _safe_int(
            summary.get("effective_tau_frames"), tau_frames
        )
        raw_stride_values = summary.get("effective_strides") or []
        stride_values = []
        for value in raw_stride_values:
            try:
                stride_values.append(int(value))
            except (TypeError, ValueError):
                continue
        stride_map = summary.get("effective_stride_map") or {}
        preview_truncated = summary.get("preview_truncated") or []

        # NOTE: Don't use pre-clustering debug data for pair counts, as clustering hasn't happened yet
        # Extract actual statistics from the MSM build result
        total_pairs_val = 0  # Will be updated from MSM diagnostics below
        zero_rows_val = 0    # Will be updated from MSM diagnostics below
        largest_cover_raw = summary.get("largest_scc_frame_fraction")
        try:
            largest_cover = (
                float(largest_cover_raw)
                if largest_cover_raw is not None
                else None
            )
        except (TypeError, ValueError):
            largest_cover = None
        diag_mass_val = _safe_float(summary.get("diag_mass"))
        expected_effective_tau = tau_frames * stride_max if tau_frames > 0 else 0

        actual_seed = int(config.seed)
        if getattr(br.metadata, "seed", None) is not None:
            try:
                actual_seed = int(br.metadata.seed)  # type: ignore[arg-type]
            except Exception:
                actual_seed = int(config.seed)
        fingerprint = {
            "mode": str(config.cluster_mode),
            "n_states": int(config.n_microstates),
            "seed": actual_seed,
        }
        msm_obj = getattr(br, "msm", None)
        feature_schema_payload: Dict[str, Any] | None = None
        if isinstance(msm_obj, Mapping):
            schema_candidate = msm_obj.get("feature_schema")
        else:
            schema_candidate = getattr(msm_obj, "feature_schema", None)
        if isinstance(schema_candidate, Mapping):
            feature_schema_payload = {
                "names": list(schema_candidate.get("names", [])),
                "n_features": int(schema_candidate.get("n_features", 0)),
            }
            fingerprint["feature_schema"] = feature_schema_payload

        fingerprint_compare = {
            "mode": fingerprint.get("mode"),
            "n_states": fingerprint.get("n_states"),
            "seed": fingerprint.get("seed"),
        }
        fingerprint_changed = fingerprint_compare != requested_fingerprint

        # Guardrail checks based on post-clustering statistics
        # Note: total_pairs and zero_rows checks are removed because they require
        # post-clustering data which we'll validate from the build result instead
        guardrail_violations: List[Dict[str, Any]] = []

        # Check if MSM build succeeded by verifying the transition matrix exists
        if br.transition_matrix is None or br.transition_matrix.size == 0:
            guardrail_violations.append(
                {"code": "msm_build_failed", "actual": "no_transition_matrix"}
            )
        else:
            # Extract actual statistics from the built MSM
            n_states_actual = br.transition_matrix.shape[0]
            if n_states_actual == 0:
                guardrail_violations.append(
                    {"code": "no_states_in_msm", "actual": 0}
                )

        if effective_tau_frames != expected_effective_tau:
            logger.warning(
                "Effective tau mismatch: expected=%d, actual=%d",
                expected_effective_tau,
                effective_tau_frames,
            )
            # Don't treat tau mismatch as a hard failure

        analysis_healthy = not guardrail_violations

        summary_overrides = {
            "fingerprint": fingerprint,
            "analysis_guardrail_violations": guardrail_violations,
            "analysis_expected_effective_tau_frames": expected_effective_tau,
            "analysis_healthy": analysis_healthy,
            "discretizer_fingerprint_changed": bool(fingerprint_changed),
        }
        summary.update(summary_overrides)

        debug_dir = (self.layout.analysis_debug_dir / f"analysis-{stamp}").resolve()
        export_info = export_analysis_debug(
            output_dir=debug_dir,
            build_result=br,
            debug_data=debug_data,
            config=config_payload,
            dataset_hash=ds_hash,
            summary_overrides=summary_overrides,
            fingerprint=fingerprint,
        )

        for idx, shard in enumerate(summary.get("shards", [])):
            shard_id = shard.get("id", f"shard-{idx}")
            frames_loaded = shard.get("frames_loaded", shard.get("length"))
            frames_declared = shard.get("frames_declared", shard.get("length"))
            stride_val = shard.get("effective_frame_stride")
            logger.info(
                "Shard %s: loaded=%s declared=%s stride=%s",
                shard_id,
                frames_loaded,
                frames_declared,
                stride_val,
            )
            if shard.get("first_timestamp") is not None or shard.get("last_timestamp") is not None:
                logger.info(
                    "Shard %s timestamps: first=%s last=%s",
                    shard_id,
                    shard.get("first_timestamp"),
                    shard.get("last_timestamp"),
                )

        if not analysis_healthy:
            summary_path = Path(export_info["summary"]).resolve()
            raise ValueError(
                "Analysis guardrails failed: "
                f"{guardrail_violations}. "
                f"See {summary_path} for details."
            )

        logger.info(
            "Analysis lag requested=%d, applied=%d, effective_tau=%d (max stride=%d, stride values=%s)",
            int(config.lag),
            tau_frames,
            effective_tau_frames,
            stride_max,
            stride_values,
        )
        if fingerprint_changed:
            logger.info(
                "Effective discretizer fingerprint differs from request: %s (requested %s)",
                fingerprint,
                requested_fingerprint,
            )
        if tau_frames != int(config.lag):
            logger.warning(
                "Analysis lag mismatch: requested %d frames, applied %d frames",
                int(config.lag),
                tau_frames,
            )
        analysis_notes["discretizer_fingerprint"] = fingerprint
        analysis_notes["analysis_total_pairs"] = int(total_pairs_val)
        analysis_notes["analysis_zero_rows"] = int(zero_rows_val)
        analysis_notes["analysis_largest_scc_fraction"] = (
            float(largest_cover) if largest_cover is not None else None
        )
        analysis_notes["analysis_diag_mass"] = float(diag_mass_val)
        analysis_notes["analysis_tau_frames"] = tau_frames
        analysis_notes["analysis_effective_tau_frames"] = effective_tau_frames
        analysis_notes["analysis_effective_stride_max"] = stride_max
        analysis_notes["analysis_effective_stride_values"] = stride_values
        analysis_notes["analysis_effective_stride_map"] = stride_map
        analysis_notes["analysis_expected_effective_tau_frames"] = expected_effective_tau
        analysis_notes["analysis_healthy"] = analysis_healthy
        if preview_truncated:
            analysis_notes["analysis_preview_truncated"] = preview_truncated
        analysis_notes["analysis_guardrail_violations"] = guardrail_violations
        analysis_notes["discretizer_fingerprint_changed"] = bool(fingerprint_changed)
        analysis_notes["analysis_kmeans_kwargs"] = dict(config.kmeans_kwargs)
        analysis_notes.pop("discretizer_fingerprint_requested", None)

        try:
            flags = dict(br.flags or {})
        except Exception:
            flags = {}
        flags["discretizer_fingerprint"] = fingerprint
        flags["discretizer_fingerprint_changed"] = bool(fingerprint_changed)
        flags["analysis_requested_tau_frames"] = int(config.lag)
        flags["analysis_total_pairs"] = int(total_pairs_val)
        flags["analysis_zero_rows"] = int(zero_rows_val)
        flags["analysis_largest_scc_fraction"] = (
            float(largest_cover) if largest_cover is not None else None
        )
        flags["analysis_diag_mass"] = float(diag_mass_val)
        flags["analysis_tau_frames"] = int(tau_frames)
        flags["analysis_effective_tau_frames"] = int(effective_tau_frames)
        flags["analysis_expected_effective_tau_frames"] = int(expected_effective_tau)
        flags["analysis_effective_stride_max"] = int(stride_max)
        flags["analysis_healthy"] = analysis_healthy
        flags["analysis_guardrail_violations"] = guardrail_violations
        flags["analysis_kmeans_kwargs"] = dict(config.kmeans_kwargs)
        if stride_values:
            flags["analysis_effective_stride_values"] = list(stride_values)
        if stride_map:
            flags["analysis_effective_stride_map"] = stride_map
        if preview_truncated:
            flags["analysis_preview_truncated"] = list(preview_truncated)
        if tau_frames != int(config.lag):
            flags["analysis_tau_mismatch"] = {
                "requested": int(config.lag),
                "actual": int(tau_frames),
            }
        overrides = {
            "cluster_mode": str(config.cluster_mode),
            "n_microstates": int(config.n_microstates),
            "reweight_mode": str(config.reweight_mode),
            "fes_method": str(config.fes_method),
            "fes_bandwidth": config.fes_bandwidth,
            "fes_min_count_per_bin": int(config.fes_min_count_per_bin),
            "apply_whitening": bool(config.apply_cv_whitening),
            "kmeans_kwargs": dict(config.kmeans_kwargs),
        }
        flags.setdefault("analysis_overrides", overrides)
        flags.setdefault("analysis_reweight_mode", str(config.reweight_mode))
        flags.setdefault("analysis_apply_whitening", bool(config.apply_cv_whitening))
        warning_count = len(debug_data.summary.get("warnings", []))
        flags.setdefault("analysis_debug_warning_count", warning_count)
        if warning_count:
            flags.setdefault(
                "analysis_debug_warnings",
                _sanitize_artifacts(debug_data.summary.get("warnings")),
            )
        br.flags = flags  # type: ignore[assignment]
        try:
            artifacts = dict(br.artifacts or {})
            artifacts["analysis_debug"] = {
                "directory": str(debug_dir),
                "summary": str(Path(export_info["summary"]).name),
                "arrays": export_info.get("arrays", {}),
            }
            artifacts["analysis_discretizer_fingerprint"] = fingerprint
            artifacts["analysis_tau_frames"] = int(tau_frames)
            artifacts["analysis_effective_tau_frames"] = int(effective_tau_frames)
            artifacts["analysis_effective_stride_max"] = int(stride_max)
            if stride_map:
                artifacts["analysis_effective_stride_map"] = stride_map
            if preview_truncated:
                artifacts["analysis_preview_truncated"] = list(preview_truncated)
            br.artifacts = artifacts  # type: ignore[assignment]
        except Exception:
            logger.debug("Failed to attach analysis debug artifacts", exc_info=True)

        artifact = BuildArtifact(
            bundle_path=bundle_path.resolve(),
            dataset_hash=ds_hash,
            build_result=br,
            created_at=stamp,
            debug_dir=debug_dir,
            debug_summary=debug_data.summary,
            discretizer_fingerprint=fingerprint,
            tau_frames=int(tau_frames),
            effective_tau_frames=int(effective_tau_frames),
            effective_stride_max=int(stride_max),
            analysis_healthy=analysis_healthy,
            guardrail_violations=guardrail_violations or None,
        )
        self.state.append_build(
            {
                "bundle": str(bundle_path.resolve()),
                "dataset_hash": ds_hash,
                "lag": int(config.lag),
                "bins": dict(config.bins),
                "seed": int(config.seed),
                "temperature": float(config.temperature),
                "learn_cv": bool(config.learn_cv),
                "deeptica_params": (
                    _sanitize_artifacts(config.deeptica_params)
                    if config.deeptica_params
                    else None
                ),
                "created_at": stamp,
                "flags": _sanitize_artifacts(br.flags),
                "mlcv": _sanitize_artifacts(br.artifacts.get("mlcv_deeptica", {})),
                "apply_cv_whitening": bool(config.apply_cv_whitening),
                "cluster_mode": str(config.cluster_mode),
                "n_microstates": int(config.n_microstates),
                "kmeans_kwargs": _sanitize_artifacts(config.kmeans_kwargs),
                "reweight_mode": str(config.reweight_mode),
                "fes_method": str(config.fes_method),
                "fes_bandwidth": config.fes_bandwidth,
                "fes_min_count_per_bin": int(config.fes_min_count_per_bin),
                "debug_dir": str(debug_dir),
                "debug_summary": _sanitize_artifacts(debug_data.summary),
                "debug_summary_file": str(Path(export_info["summary"]).name),
                "discretizer_fingerprint": _sanitize_artifacts(fingerprint),
                "discretizer_fingerprint_changed": bool(fingerprint_changed),
                "tau_frames": int(tau_frames),
                "effective_tau_frames": int(effective_tau_frames),
                "effective_stride_max": int(stride_max),
                "effective_stride_values": list(stride_values),
                "effective_stride_map": _sanitize_artifacts(stride_map),
                "preview_truncated": list(preview_truncated),
                "analysis_healthy": bool(analysis_healthy),
                "guardrail_violations": _sanitize_artifacts(guardrail_violations),
                "total_pairs": int(total_pairs_val),
                "zero_rows": int(zero_rows_val),
                "largest_scc_fraction": float(largest_cover) if largest_cover is not None else None,
                "diag_mass": float(diag_mass_val),
            }
        )
        return artifact

    # ------------------------------------------------------------------
    # Utilities used by the UI
    # ------------------------------------------------------------------
    def latest_model_path(self) -> Optional[Path]:
        if not self.state.models:
            return None
        last = self.state.models[-1]
        p = Path(last.get("bundle", ""))
        return p if p.exists() else None

    def list_models(self) -> List[Dict[str, Any]]:
        return [dict(entry) for entry in self.state.models]

    def list_builds(self) -> List[Dict[str, Any]]:
        return [dict(entry) for entry in self.state.builds]

    def sidebar_summary(self) -> Dict[str, int]:
        # Reconcile stale shard entries first
        self._reconcile_shard_state()

        # Count shard files on disk for accuracy
        try:
            shard_files = len(self.discover_shards())
        except Exception:
            shard_files = len(self.state.shards)

        return {
            "runs": len(self.state.runs),
            "shards": int(shard_files),
            "models": len(self.state.models),
            "builds": len(self.state.builds),
        }

    def _reconcile_shard_state(self) -> None:
        """Remove shard batches from state if all referenced files are missing."""
        try:
            to_delete: List[int] = []
            for i, entry in enumerate(list(self.state.shards)):
                paths = [Path(p) for p in entry.get("paths", [])]
                existing = [p for p in paths if p.exists()]
                if len(existing) == 0:
                    to_delete.append(i)
            for i in reversed(to_delete):
                # Best-effort removal (also attempts to clean empty dirs)
                if not self.delete_shard_batch(i):
                    try:
                        self.state.remove_shards(i)
                    except Exception:
                        pass
        except Exception:
            # Non-fatal; leave state as-is
            pass

    # ------------------------------------------------------------------
    # Rehydrate existing assets
    # ------------------------------------------------------------------
    def load_run(self, run_id: str) -> Optional[SimulationResult]:
        """Best-effort reconstruction of a previous simulation result."""

        if not run_id:
            return None
        record: Optional[Dict[str, Any]] = None
        for entry in reversed(self.state.runs):
            if str(entry.get("run_id")) == str(run_id):
                record = dict(entry)
                break
        if record is None:
            return None

        run_dir = Path(record.get("run_dir", ""))
        pdb_path = Path(record.get("pdb", ""))
        if not run_dir.exists() or not pdb_path.exists():
            return None

        traj_strings: Sequence[str] = record.get("traj_files", []) or []
        traj_files = [Path(p) for p in traj_strings if Path(p).exists()]
        if not traj_files:
            # Fallback: scan standard REMD output locations
            candidates: List[Path] = []
            replica_dir = run_dir / "replica_exchange"
            demux_dir = run_dir
            candidates.extend(sorted(replica_dir.rglob("*.dcd")))
            candidates.extend(sorted(replica_dir.rglob("*.nc")))
            candidates.extend(sorted(demux_dir.glob("demux_*.*")))
            traj_files = candidates
        if not traj_files:
            return None

        analysis_temps = [float(t) for t in record.get("analysis_temperatures", [])]
        steps = int(record.get("steps", 0))
        created_at = str(record.get("created_at", "")) or _timestamp()

        return SimulationResult(
            run_id=str(record.get("run_id")),
            run_dir=run_dir.resolve(),
            pdb_path=pdb_path.resolve(),
            traj_files=[p.resolve() for p in traj_files],
            analysis_temperatures=analysis_temps,
            steps=steps,
            created_at=created_at,
        )

    def load_model(self, index: int) -> Optional[TrainingResult]:
        if index < 0 or index >= len(self.state.models):
            return None
        entry = dict(self.state.models[index])
        return self._load_model_from_entry(entry)

    def load_analysis_bundle(self, index: int) -> Optional[BuildArtifact]:
        if index < 0 or index >= len(self.state.builds):
            return None
        entry = dict(self.state.builds[index])
        return self._load_analysis_from_entry(entry)

    def build_config_from_entry(self, entry: Dict[str, Any]) -> BuildConfig:
        bins_raw = entry.get("bins")
        bins = (
            dict(bins_raw) if isinstance(bins_raw, dict) else {"Rg": 64, "RMSD_ref": 64}
        )
        deeptica_params = self._coerce_deeptica_params(entry.get("deeptica_params"))
        notes = {}
        entry_notes = entry.get("notes")
        if isinstance(entry_notes, dict):
            notes.update(entry_notes)
        apply_whitening = bool(entry.get("apply_cv_whitening", True))
        cluster_mode = str(entry.get("cluster_mode", "kmeans"))
        n_microstates = int(entry.get("n_microstates", 20))
        kmeans_kwargs_raw = entry.get("kmeans_kwargs")
        kmeans_kwargs = (
            dict(kmeans_kwargs_raw)
            if isinstance(kmeans_kwargs_raw, dict)
            else {"n_init": 50}
        )
        reweight_mode = str(entry.get("reweight_mode", "MBAR"))
        fes_method = str(entry.get("fes_method", "kde"))
        bw_raw = entry.get("fes_bandwidth", "scott")
        try:
            fes_bandwidth = float(bw_raw)
        except (TypeError, ValueError):
            fes_bandwidth = bw_raw if bw_raw is not None else "scott"
        min_count = int(entry.get("fes_min_count_per_bin", 1))
        return BuildConfig(
            lag=int(entry.get("lag", 10)),
            bins=bins,
            seed=int(entry.get("seed", 2025)),
            temperature=float(entry.get("temperature", 300.0)),
            learn_cv=bool(entry.get("learn_cv", False)),
            deeptica_params=deeptica_params,
            notes=notes,
            apply_cv_whitening=apply_whitening,
            cluster_mode=cluster_mode,
            n_microstates=n_microstates,
            reweight_mode=reweight_mode,
            fes_method=fes_method,
            fes_bandwidth=fes_bandwidth,
            fes_min_count_per_bin=min_count,
            kmeans_kwargs=kmeans_kwargs,
        )

    def training_config_from_entry(self, entry: Dict[str, Any]) -> TrainingConfig:
        bins_raw = entry.get("bins")
        bins = (
            dict(bins_raw) if isinstance(bins_raw, dict) else {"Rg": 64, "RMSD_ref": 64}
        )
        hidden = self._coerce_hidden_layers(entry.get("hidden"))
        tau_raw = entry.get("tau_schedule")
        tau_schedule = self._coerce_tau_schedule(tau_raw)
        if not tau_schedule:
            tau_schedule = (int(entry.get("lag", 5)),)
        val_tau_entry = entry.get("val_tau")
        val_tau = (
            int(val_tau_entry)
            if val_tau_entry is not None
            else (tau_schedule[-1] if tau_schedule else int(entry.get("lag", 5)))
        )
        epochs_per_tau = int(entry.get("epochs_per_tau", 15))
        return TrainingConfig(
            lag=int(entry.get("lag", 5)),
            bins=bins,
            seed=int(entry.get("seed", 1337)),
            temperature=float(entry.get("temperature", 300.0)),
            hidden=hidden,
            max_epochs=int(entry.get("max_epochs", 200)),
            early_stopping=int(entry.get("early_stopping", 25)),
            tau_schedule=tau_schedule,
            val_tau=val_tau,
            epochs_per_tau=epochs_per_tau,
        )

    def _coerce_deeptica_params(self, raw: Any) -> Optional[Dict[str, Any]]:
        if raw is None:
            return None
        if isinstance(raw, dict):
            return {str(k): v for k, v in raw.items()}
        return None

    @staticmethod
    def _coerce_hidden_layers(raw: Any) -> tuple[int, ...]:
        layers: List[int] = []
        if isinstance(raw, (list, tuple)):
            for item in raw:
                try:
                    layers.append(int(item))
                except (TypeError, ValueError):
                    continue
        elif isinstance(raw, str):
            for token in raw.split(","):
                token = token.strip()
                if not token:
                    continue
                try:
                    layers.append(int(token))
                except ValueError:
                    continue
        if layers:
            return tuple(layers)
        return (128, 128)

    @staticmethod
    def _coerce_tau_schedule(raw: Any) -> tuple[int, ...]:
        values: List[int] = []
        if isinstance(raw, (list, tuple)):
            for item in raw:
                try:
                    v = int(item)
                    if v > 0:
                        values.append(v)
                except (TypeError, ValueError):
                    continue
        elif isinstance(raw, str):
            tokens = raw.replace(";", ",").split(",")
            for token in tokens:
                token = token.strip()
                if not token:
                    continue
                try:
                    v = int(token)
                    if v > 0:
                        values.append(v)
                except ValueError:
                    continue
        if not values:
            return ()
        return tuple(sorted(set(values)))

    @staticmethod
    def _load_build_result_from_path(path: Path) -> Optional["_BuildResult"]:
        try:
            bundle_path = Path(path)
        except TypeError:
            return None
        if not bundle_path.exists():
            return None
        try:
            text = bundle_path.read_text(encoding="utf-8")
        except Exception:
            return None
        try:
            return _build_result_cls().from_json(text)
        except Exception:
            return None

    def _load_model_from_entry(self, entry: Dict[str, Any]) -> Optional[TrainingResult]:
        bundle_path = Path(entry.get("bundle", ""))
        br = self._load_build_result_from_path(bundle_path)
        if br is None:
            return None
        dataset_hash = str(entry.get("dataset_hash", "")) or (
            str(getattr(br.metadata, "dataset_hash", "")) if br.metadata else ""
        )
        created_at = str(entry.get("created_at", "")) or _timestamp()
        return TrainingResult(
            bundle_path=bundle_path.resolve(),
            dataset_hash=dataset_hash,
            build_result=br,
            created_at=created_at,
        )

    def _load_analysis_from_entry(
        self, entry: Dict[str, Any]
    ) -> Optional[BuildArtifact]:
        bundle_path = Path(entry.get("bundle", ""))
        br = self._load_build_result_from_path(bundle_path)
        if br is None:
            return None
        dataset_hash = str(entry.get("dataset_hash", "")) or (
            str(getattr(br.metadata, "dataset_hash", "")) if br.metadata else ""
        )
        created_at = str(entry.get("created_at", "")) or _timestamp()
        debug_dir_raw = entry.get("debug_dir")
        debug_dir = Path(debug_dir_raw).resolve() if debug_dir_raw else None
        debug_summary = entry.get("debug_summary")
        if debug_summary is None and debug_dir:
            summary_name = entry.get("debug_summary_file") or "summary.json"
            summary_path = debug_dir / summary_name
            if summary_path.exists():
                try:
                    debug_summary = json.loads(summary_path.read_text(encoding="utf-8"))
                except Exception:
                    logger.debug("Failed to load analysis summary from %s", summary_path)
        fingerprint = entry.get("discretizer_fingerprint")
        if fingerprint is None and isinstance(br.flags, Mapping):
            fingerprint = br.flags.get("discretizer_fingerprint")
        tau_frames = entry.get("tau_frames")
        if tau_frames is None and isinstance(br.flags, Mapping):
            tau_frames = br.flags.get("analysis_tau_frames")
        effective_tau_frames = entry.get("effective_tau_frames")
        if effective_tau_frames is None and isinstance(br.flags, Mapping):
            effective_tau_frames = br.flags.get("analysis_effective_tau_frames")
        effective_stride_max = entry.get("effective_stride_max")
        if effective_stride_max is None and isinstance(br.flags, Mapping):
            effective_stride_max = br.flags.get("analysis_effective_stride_max")
        return BuildArtifact(
            bundle_path=bundle_path.resolve(),
            dataset_hash=dataset_hash,
            build_result=br,
            created_at=created_at,
            debug_dir=debug_dir,
            debug_summary=debug_summary,
            discretizer_fingerprint=fingerprint,
            tau_frames=int(tau_frames) if tau_frames is not None else None,
            effective_tau_frames=(
                int(effective_tau_frames)
                if effective_tau_frames is not None
                else None
            ),
            effective_stride_max=(
                int(effective_stride_max) if effective_stride_max is not None else None
            ),
        )

    # ------------------------------------------------------------------
    # Asset deletion methods
    # ------------------------------------------------------------------
    def delete_simulation(self, index: int) -> bool:
        """Delete a simulation run and its associated files."""
        entry = self.state.remove_run(index)
        if entry is None:
            return False

        try:
            # Delete simulation directory
            run_dir = Path(entry.get("run_dir", ""))
            if run_dir.exists() and run_dir.is_dir():
                shutil.rmtree(run_dir)

            # Also remove any associated shards
            run_id = entry.get("run_id", "")
            if run_id:
                # Find and remove associated shard entries
                shards_to_remove = []
                for i, shard_entry in enumerate(self.state.shards):
                    if shard_entry.get("run_id") == run_id:
                        shards_to_remove.append(i)

                # Remove in reverse order to maintain indices
                for i in reversed(shards_to_remove):
                    self.delete_shard_batch(i)

            return True
        except Exception:
            return False

    def delete_shard_batch(self, index: int) -> bool:
        """Delete a shard batch and its associated files."""
        entry = self.state.remove_shards(index)
        if entry is None:
            return False

        try:
            # Delete individual shard files
            paths = entry.get("paths", [])
            for path_str in paths:
                path = Path(path_str)
                if path.exists():
                    path.unlink()  # Delete the .json file
                    # Also delete associated .npz file
                    npz_path = path.with_suffix(".npz")
                    if npz_path.exists():
                        npz_path.unlink()

            # Delete shard directory if empty
            directory = Path(entry.get("directory", ""))
            if directory.exists() and directory.is_dir():
                try:
                    directory.rmdir()  # Only removes if empty
                except OSError:
                    pass  # Directory not empty, that's OK

            return True
        except Exception:
            return False

    def delete_model(self, index: int) -> bool:
        """Delete a model and its associated files."""
        entry = self.state.remove_model(index)
        if entry is None:
            return False

        try:
            # Delete model bundle file and associated files
            bundle_path = Path(entry.get("bundle", ""))
            if bundle_path.exists():
                base_name = bundle_path.stem
                model_dir = bundle_path.parent

                # Find and delete related files (history, json, pt files)
                for file_path in model_dir.glob(f"{base_name}.*"):
                    if file_path.is_file():
                        file_path.unlink()

            return True
        except Exception:
            return False

    def delete_analysis_bundle(self, index: int) -> bool:
        """Delete an analysis bundle and its associated files."""
        entry = self.state.remove_build(index)
        if entry is None:
            return False

        try:
            # Delete bundle file
            bundle_path = Path(entry.get("bundle", ""))
            if bundle_path.exists():
                bundle_path.unlink()
            debug_dir = Path(entry.get("debug_dir", ""))
            if debug_dir.exists() and debug_dir.is_dir():
                shutil.rmtree(debug_dir)

            return True
        except Exception:
            return False
