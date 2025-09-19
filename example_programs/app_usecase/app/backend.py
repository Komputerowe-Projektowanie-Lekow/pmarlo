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

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import shutil
from typing import Any, Dict, Iterable, List, Optional, Sequence

from pmarlo.api import (
    build_from_shards,
    emit_shards_rg_rmsd_windowed,
    run_replica_exchange,
)
from pmarlo.data.shard import read_shard
from pmarlo.transform.build import BuildResult, _sanitize_artifacts

try:  # Package-relative when imported as module
    from .state import StateManager
except ImportError:  # Fallback for direct script import
    import sys

    _APP_DIR = Path(__file__).resolve().parent
    if str(_APP_DIR) not in sys.path:
        sys.path.insert(0, str(_APP_DIR))
    from state import StateManager  # type: ignore

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
) -> "SimulationResult":
    """Run a short simulation for testing purposes."""
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
    config = SimulationConfig(
        pdb_path=pdb_path,
        temperatures=temperatures,
        steps=steps,
        quick=quick,
        random_seed=random_seed,
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
        for path in (self.workspace_dir, self.sims_dir, self.shards_dir, self.models_dir, self.bundles_dir, self.logs_dir):
            path.mkdir(parents=True, exist_ok=True)

    def available_inputs(self) -> List[Path]:
        if not self.inputs_dir.exists():
            return []
        return sorted(p.resolve() for p in self.inputs_dir.glob("*.pdb"))


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

    def deeptica_params(self) -> Dict[str, Any]:
        return {
            "lag": int(max(1, self.lag)),
            "n_out": 2,
            "hidden": tuple(int(h) for h in self.hidden),
            "max_epochs": int(self.max_epochs),
            "early_stopping": int(self.early_stopping),
            "reweight_mode": "scaled_time",
        }


@dataclass
class TrainingResult:
    bundle_path: Path
    dataset_hash: str
    build_result: BuildResult
    created_at: str


@dataclass
class BuildConfig:
    lag: int
    bins: Dict[str, int]
    seed: int
    temperature: float
    learn_cv: bool = False
    deeptica_params: Optional[Dict[str, Any]] = None
    notes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BuildArtifact:
    bundle_path: Path
    dataset_hash: str
    build_result: BuildResult
    created_at: str


class WorkflowBackend:
    """High-level orchestration for the Streamlit UI."""

    def __init__(self, layout: WorkspaceLayout) -> None:
        self.layout = layout
        self.state = StateManager(layout.state_path)

    # ------------------------------------------------------------------
    # Sampling & shard emission
    # ------------------------------------------------------------------
    def run_sampling(self, config: SimulationConfig) -> SimulationResult:
        run_label = _slugify(config.label) or f"run-{_timestamp()}"
        run_dir = self.layout.sims_dir / run_label
        run_dir.mkdir(parents=True, exist_ok=True)
        traj_files, temps = run_replica_exchange(
            pdb_file=str(config.pdb_path),
            output_dir=str(run_dir),
            temperatures=[float(t) for t in config.temperatures],
            total_steps=int(config.steps),
            quick=bool(config.quick),
            random_seed=int(config.random_seed) if config.random_seed is not None else None,
            jitter_start=bool(config.jitter_start),
            jitter_sigma_A=float(config.jitter_sigma_A),
            exchange_frequency_steps=(
                int(config.exchange_frequency_steps)
                if config.exchange_frequency_steps is not None
                else None
            ),
            temperature_schedule_mode=config.temperature_schedule_mode,
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
        self.state.append_run(
            {
                "run_id": run_label,
                "run_dir": str(result.run_dir),
                "pdb": str(result.pdb_path),
                "temperatures": [float(t) for t in config.temperatures],
                "analysis_temperatures": result.analysis_temperatures,
                "steps": int(config.steps),
                "quick": bool(config.quick),
                "random_seed": int(config.random_seed)
                if config.random_seed is not None
                else None,
                "traj_files": [str(p) for p in result.traj_files],
                "created_at": created,
            }
        )
        return result

    def emit_shards(
        self,
        simulation: SimulationResult,
        request: ShardRequest,
        *,
        provenance: Optional[Dict[str, Any]] = None,
    ) -> ShardResult:
        shard_dir = self.layout.shards_dir / simulation.run_id
        shard_dir.mkdir(parents=True, exist_ok=True)
        note = {
            "run_id": simulation.run_id,
            "analysis_temperatures": simulation.analysis_temperatures,
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
        created = _timestamp()
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
    def train_model(
        self,
        shard_jsons: Sequence[Path],
        config: TrainingConfig,
    ) -> TrainingResult:
        shards = [Path(p).resolve() for p in shard_jsons]
        if not shards:
            raise ValueError("No shards selected for training")
        stamp = _timestamp()
        bundle_path = self.layout.models_dir / f"deeptica-{stamp}.pbz"
        try:
            br, ds_hash = build_from_shards(
                shard_jsons=shards,
                out_bundle=bundle_path,
                bins=dict(config.bins),
                lag=int(config.lag),
                seed=int(config.seed),
                temperature=float(config.temperature),
                learn_cv=True,
                deeptica_params=config.deeptica_params(),
                notes={"model_dir": str(self.layout.models_dir)},
            )
        except ImportError as exc:
            raise RuntimeError(
                "Deep-TICA optional dependencies missing. Install pmarlo[mlcv] to enable"
            ) from exc
        except Exception:
            raise
        result = TrainingResult(
            bundle_path=bundle_path.resolve(),
            dataset_hash=ds_hash,
            build_result=br,
            created_at=stamp,
        )
        self.state.append_model(
            {
                "bundle": str(bundle_path.resolve()),
                "dataset_hash": ds_hash,
                "lag": int(config.lag),
                "bins": dict(config.bins),
                "seed": int(config.seed),
                "temperature": float(config.temperature),
                "hidden": [int(h) for h in config.hidden],
                "max_epochs": int(config.max_epochs),
                "early_stopping": int(config.early_stopping),
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
        analysis_notes = dict(config.notes or {})
        if config.learn_cv and "model_dir" not in analysis_notes:
            analysis_notes["model_dir"] = str(self.layout.models_dir)

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
        )
        artifact = BuildArtifact(
            bundle_path=bundle_path.resolve(),
            dataset_hash=ds_hash,
            build_result=br,
            created_at=stamp,
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
                "deeptica_params": _sanitize_artifacts(config.deeptica_params) if config.deeptica_params else None,
                "created_at": stamp,
                "flags": _sanitize_artifacts(br.flags),
                "mlcv": _sanitize_artifacts(br.artifacts.get("mlcv_deeptica", {})),
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
        bins = dict(bins_raw) if isinstance(bins_raw, dict) else {"Rg": 64, "RMSD_ref": 64}
        deeptica_params = self._coerce_deeptica_params(entry.get("deeptica_params"))
        notes = {}
        entry_notes = entry.get("notes")
        if isinstance(entry_notes, dict):
            notes.update(entry_notes)
        return BuildConfig(
            lag=int(entry.get("lag", 10)),
            bins=bins,
            seed=int(entry.get("seed", 2025)),
            temperature=float(entry.get("temperature", 300.0)),
            learn_cv=bool(entry.get("learn_cv", False)),
            deeptica_params=deeptica_params,
            notes=notes,
        )

    def training_config_from_entry(self, entry: Dict[str, Any]) -> TrainingConfig:
        bins_raw = entry.get("bins")
        bins = dict(bins_raw) if isinstance(bins_raw, dict) else {"Rg": 64, "RMSD_ref": 64}
        hidden = self._coerce_hidden_layers(entry.get("hidden"))
        return TrainingConfig(
            lag=int(entry.get("lag", 5)),
            bins=bins,
            seed=int(entry.get("seed", 1337)),
            temperature=float(entry.get("temperature", 300.0)),
            hidden=hidden,
            max_epochs=int(entry.get("max_epochs", 200)),
            early_stopping=int(entry.get("early_stopping", 25)),
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
    def _load_build_result_from_path(path: Path) -> Optional[BuildResult]:
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
            return BuildResult.from_json(text)
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

    def _load_analysis_from_entry(self, entry: Dict[str, Any]) -> Optional[BuildArtifact]:
        bundle_path = Path(entry.get("bundle", ""))
        br = self._load_build_result_from_path(bundle_path)
        if br is None:
            return None
        dataset_hash = str(entry.get("dataset_hash", "")) or (
            str(getattr(br.metadata, "dataset_hash", "")) if br.metadata else ""
        )
        created_at = str(entry.get("created_at", "")) or _timestamp()
        return BuildArtifact(
            bundle_path=bundle_path.resolve(),
            dataset_hash=dataset_hash,
            build_result=br,
            created_at=created_at,
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
                    npz_path = path.with_suffix('.npz')
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

            return True
        except Exception:
            return False
