from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Sequence, Tuple, Mapping

from pmarlo.conformations.representative_picker import (
    TrajectoryFrameLocator,
    TrajectorySegment,
)

from pmarlo.analysis.debug_export import AnalysisDebugData

@dataclass
class ShardRequest:
    stride: int = 5
    temperature: float = 300.0
    reference: Optional[Path] = None
    seed_start: int = 0
    frames_per_shard: int = 5000
    hop_frames: Optional[int] = None
    feature_profile: str = "cv_analysis"  # "cv_analysis" or "molecular_cv_biasing" or "molecular_custom"

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
    feature_profile: str = "cv_analysis"  # Feature profile used for extraction

@dataclass
class SimulationResult:
    run_id: str
    run_dir: Path
    pdb_path: Path
    traj_files: List[Path]
    analysis_temperatures: List[float]
    steps: int
    created_at: str
    restart_pdb_path: Optional[Path] = None
    restart_inputs_entry: Optional[Path] = None

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
    cv_model_bundle: Optional[Path] = None
    stub_result: bool = False
    save_restart_pdb: bool = False
    restart_temperature: Optional[float] = None
    start_from_pdb: Optional[Path] = None
    start_from_checkpoint: Optional[Path] = None
    save_state_frequency: Optional[int] = None
    resume_context: Optional[Dict[str, Any]] = None
    force_run_id: Optional[str] = None
    single_temperature_mode: bool = False  # Use single-T MD instead of REMD

    def snapshot(self) -> Dict[str, Any]:
        """Serialize configuration to a JSON-friendly dict."""
        def _path_str(value: Optional[Path]) -> Optional[str]:
            if value is None:
                return None
            return str(Path(value).expanduser().resolve())

        return {
            "pdb_path": _path_str(self.pdb_path),
            "temperatures": [float(t) for t in self.temperatures],
            "steps": int(self.steps),
            "quick": bool(self.quick),
            "random_seed": int(self.random_seed)
            if self.random_seed is not None
            else None,
            "label": self.label,
            "jitter_start": bool(self.jitter_start),
            "jitter_sigma_A": float(self.jitter_sigma_A),
            "exchange_frequency_steps": int(self.exchange_frequency_steps)
            if self.exchange_frequency_steps is not None
            else None,
            "temperature_schedule_mode": self.temperature_schedule_mode,
            "cv_model_bundle": _path_str(self.cv_model_bundle),
            "stub_result": bool(self.stub_result),
            "save_restart_pdb": bool(self.save_restart_pdb),
            "restart_temperature": float(self.restart_temperature)
            if self.restart_temperature is not None
            else None,
            "start_from_pdb": _path_str(self.start_from_pdb),
            "start_from_checkpoint": _path_str(self.start_from_checkpoint),
            "save_state_frequency": int(self.save_state_frequency)
            if self.save_state_frequency is not None
            else None,
            "resume_context": dict(self.resume_context)
            if isinstance(self.resume_context, Mapping)
            else self.resume_context,
            "single_temperature_mode": bool(self.single_temperature_mode),
            "force_run_id": self.force_run_id,
        }

    @classmethod
    def from_snapshot(cls, snapshot: Mapping[str, Any]) -> "SimulationConfig":
        """Reconstruct configuration from a serialized snapshot."""
        def _path(value: Any) -> Optional[Path]:
            if value in (None, ""):
                return None
            return Path(str(value)).expanduser()

        temps = snapshot.get("temperatures") or []
        return cls(
            pdb_path=_path(snapshot["pdb_path"]) or Path(snapshot["pdb_path"]),
            temperatures=[float(t) for t in temps],
            steps=int(snapshot.get("steps", 0)),
            quick=bool(snapshot.get("quick", True)),
            random_seed=(
                int(snapshot["random_seed"])
                if snapshot.get("random_seed") is not None
                else None
            ),
            label=snapshot.get("label"),
            jitter_start=bool(snapshot.get("jitter_start", False)),
            jitter_sigma_A=float(snapshot.get("jitter_sigma_A", 0.05)),
            exchange_frequency_steps=(
                int(snapshot["exchange_frequency_steps"])
                if snapshot.get("exchange_frequency_steps") is not None
                else None
            ),
            temperature_schedule_mode=snapshot.get("temperature_schedule_mode"),
            cv_model_bundle=_path(snapshot.get("cv_model_bundle")),
            stub_result=bool(snapshot.get("stub_result", False)),
            save_restart_pdb=bool(snapshot.get("save_restart_pdb", False)),
            restart_temperature=(
                float(snapshot["restart_temperature"])
                if snapshot.get("restart_temperature") is not None
                else None
            ),
            start_from_pdb=_path(snapshot.get("start_from_pdb")),
            start_from_checkpoint=_path(snapshot.get("start_from_checkpoint")),
            save_state_frequency=(
                int(snapshot["save_state_frequency"])
                if snapshot.get("save_state_frequency") is not None
                else None
            ),
            resume_context=(
                dict(snapshot["resume_context"])
                if isinstance(snapshot.get("resume_context"), Mapping)
                else snapshot.get("resume_context")
            ),
            single_temperature_mode=bool(snapshot.get("single_temperature_mode", False)),
            force_run_id=snapshot.get("force_run_id"),
        )

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
    gradient_clip_val: float = 1.0
    learning_rate: float = 3e-4  # Default from DEEPTICA_DEFAULT_LEARNING_RATE
    weight_decay: float = 0.0  # Default from DEEPTICA_DEFAULT_WEIGHT_DECAY

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
            "gradient_clip_val": float(self.gradient_clip_val),
            "learning_rate": float(self.learning_rate),
            "weight_decay": float(self.weight_decay),
        }

@dataclass
class TrainingResult:
    bundle_path: Path
    dataset_hash: str
    build_result: "_BuildResult"
    created_at: str
    checkpoint_dir: Optional[Path] = None
    cv_model_bundle: Optional[Dict[str, Any]] = None

@dataclass
class BuildConfig:
    lag: int
    bins: Dict[str, int]
    seed: int
    temperature: float
    fes_grid_strategy: str = "adaptive"
    fes_bins: tuple[int, int] | None = None
    learn_cv: bool = False
    deeptica_params: Optional[Dict[str, Any]] = None
    notes: Dict[str, Any] = field(default_factory=dict)
    apply_cv_whitening: bool = False
    cluster_mode: str = "kmeans"
    n_microstates: int = 20
    kmeans_kwargs: Dict[str, Any] = field(default_factory=dict)
    reweight_mode: str = "MBAR"
    fes_method: str = "kde"
    fes_bandwidth: str | float = "scott"
    fes_min_count_per_bin: int = 1
    require_fully_connected_msm: bool = True  # Set to False to allow disconnected MSMs

@dataclass
class BuildArtifact:
    bundle_path: Path
    dataset_hash: str
    build_result: "_BuildResult"
    created_at: str
    debug_dir: Optional[Path] = None
    debug_summary: Optional[Dict[str, Any]] = None
    discretizer_fingerprint: Optional[Dict[str, Any]] = None
    tau_frames: Optional[int] = None
    effective_tau_frames: Optional[int] = None
    effective_stride_max: Optional[int] = None
    analysis_msm_n_states: Optional[int] = None
    analysis_healthy: bool = True
    guardrail_violations: Optional[List[Dict[str, Any]]] = None

@dataclass
class ConformationsConfig:
    """Configuration for TPT conformations analysis."""
    lag: int = 10
    n_clusters: int = 30
    cluster_mode: str = "kmeans"
    cluster_seed: Optional[int] = 42
    kmeans_n_init: int = 50
    kmeans_kwargs: Dict[str, Any] = field(default_factory=dict)
    n_components: int = 10
    tica_dim: Optional[int] = None
    n_metastable: int = 10
    temperature: float = 300.0
    auto_detect_states: bool = True
    source_states: Optional[List[int]] = None
    sink_states: Optional[List[int]] = None
    n_paths: int = 10
    pathway_fraction: float = 0.99
    compute_kis: bool = True
    k_slow: int = 3
    uncertainty_analysis: bool = True
    bootstrap_samples: int = 50
    n_representatives: int = 5
    topology_pdb: Optional[Path] = None
    cv_method: str = "tica"
    deeptica_projection_path: Optional[Path] = None
    deeptica_metadata_path: Optional[Path] = None
    committor_thresholds: Tuple[float, float] = (0.05, 0.95)

@dataclass
class ConformationsResult:
    """Result from TPT conformations analysis."""
    output_dir: Path
    tpt_summary: Dict[str, Any]
    metastable_states: Dict[str, Any]
    transition_states: List[Dict[str, Any]]
    pathways: List[List[int]]
    representative_pdbs: List[Path]
    plots: Dict[str, Path]
    created_at: str
    config: ConformationsConfig
    error: Optional[str] = None
    tpt_converged: bool = True
    tpt_pathway_iterations: Optional[int] = None
    tpt_pathway_max_iterations: Optional[int] = None

@dataclass(frozen=True)
class _AnalysisMSMStats:
    total_pairs: int
    zero_rows: int
    largest_scc_fraction: float | None
    diag_mass: float
