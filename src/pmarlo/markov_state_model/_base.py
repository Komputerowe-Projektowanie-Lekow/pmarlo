from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import mdtraj as md
import numpy as np
import pandas as pd

logger = logging.getLogger("pmarlo")


@dataclass
class CKTestResult:
    """Results of a Chapman–Kolmogorov test."""

    mse: Dict[int, float] = field(default_factory=dict)
    mode: str = "micro"
    insufficient_data: bool = False
    thresholds: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mse": {int(k): float(v) for k, v in self.mse.items()},
            "mode": self.mode,
            "insufficient_data": self.insufficient_data,
            "thresholds": self.thresholds,
        }


class MSMBase:
    """Base class providing core attributes and bookkeeping for MSM analysis."""

    def __init__(
        self,
        trajectory_files: Optional[Union[str, List[str]]] = None,
        topology_file: Optional[str] = None,
        temperatures: Optional[List[float]] = None,
        output_dir: str = "output/msm_analysis",
        random_state: Optional[int] = 42,
    ) -> None:
        # IO and configuration
        self.trajectory_files: List[str] = (
            trajectory_files
            if isinstance(trajectory_files, list)
            else [trajectory_files] if trajectory_files else []
        )
        self.topology_file: Optional[str] = topology_file
        self.temperatures: List[float] = temperatures or [300.0]
        self.output_dir: Path = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Trajectory and feature data
        self.trajectories: List[md.Trajectory] = []
        self.dtrajs: List[np.ndarray] = []
        self.features: Optional[np.ndarray] = None
        self.cluster_centers: Optional[np.ndarray] = None
        self.n_states: int = 0
        self.random_state: Optional[int] = (
            int(random_state) if random_state is not None else None
        )

        # MSM data
        self.transition_matrix: Optional[np.ndarray] = None
        self.count_matrix: Optional[np.ndarray] = None
        self.stationary_distribution: Optional[np.ndarray] = None
        self.free_energies: Optional[np.ndarray] = None
        self.lag_time: int = 20
        self.frame_stride: Optional[int] = None

        # Feature processing settings
        self.feature_stride: int = 1
        self.tica_lag: int = 0
        self.tica_components: Optional[int] = None
        self.effective_frames: int = 0
        self.raw_frames: int = 0

        # Metadata
        self.time_per_frame_ps: Optional[float] = None
        self.demux_metadata: Optional[Any] = None
        self.total_frames: Optional[int] = None

        # Estimation controls
        self.estimator_backend: str = "deeptime"
        self.count_mode: str = "sliding"

        # TRAM
        self.tram_weights: Optional[np.ndarray] = None
        self.multi_temp_counts: Dict[float, Dict[Tuple[int, int], float]] = {}

        # Analysis results containers
        self.implied_timescales: Optional[Any] = None
        self.state_table: Optional[pd.DataFrame] = None
        self.fes_data: Optional[Dict[str, Any]] = None

        logger.info(
            f"Enhanced MSM initialized for {len(self.trajectory_files)} trajectories"
        )

    def _update_total_frames(self) -> None:
        try:
            self.total_frames = int(sum(int(t.n_frames) for t in self.trajectories))
        except Exception:
            self.total_frames = None
