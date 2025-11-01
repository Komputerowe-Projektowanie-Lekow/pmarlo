from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Sequence, Tuple

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
