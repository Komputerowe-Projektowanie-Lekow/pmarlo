"""Fallback EnhancedMSM implementation used when scikit-learn is unavailable."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Union

import numpy as np

logger = logging.getLogger("pmarlo")


class EnhancedMSMStub:
    """Minimal stub used when the optional machine-learning stack is missing."""

    def __init__(
        self,
        *,
        trajectory_files: Union[str, List[str]] | None = None,
        topology_file: str | None = None,
        temperatures: Optional[List[float]] = None,
        output_dir: str | None = None,
        **_: object,
    ) -> None:
        self.output_dir = output_dir
        self.trajectories: list[object] = []
        self._effective_frames = 0
        self._feature_stride = 1
        self._tica_lag = 0

    @property
    def effective_frames(self) -> int:
        return int(self._effective_frames)

    def compute_features(
        self,
        feature_type: str = "",
        n_features: Optional[int] = None,
        feature_stride: int = 1,
        tica_lag: int = 0,
        tica_components: Optional[int] = None,
        **_: object,
    ) -> None:
        stride = int(feature_stride or 1)
        if stride <= 0:
            stride = 1
        total_frames = 0
        for traj in self.trajectories:
            n_frames = getattr(traj, "n_frames", None)
            if n_frames is None and hasattr(traj, "xyz"):
                n_frames = np.asarray(traj.xyz).shape[0]
            total_frames += int(n_frames or 0)
        processed = total_frames // stride
        self._feature_stride = stride
        self._tica_lag = int(max(0, tica_lag))
        self._effective_frames = max(0, processed - self._tica_lag)

    def build_msm(self, *, lag_time: int, **_: object) -> None:
        lag = int(max(0, lag_time))
        if self.effective_frames < lag:
            msg = f"effective frames after lag {lag}: {self.effective_frames}"
            logger.info(msg)
            raise ValueError(msg)

    def compute_features_from_traj(
        self,
        *,
        feature_stride: int | None = None,
        tica_lag: int = 0,
        tica_components: int | None = None,
    ) -> None:
        stride_value = 1 if feature_stride is None else int(feature_stride)
        self.compute_features(
            feature_stride=stride_value,
            tica_lag=tica_lag,
            tica_components=tica_components,
        )

    def load_trajectories(
        self,
        *,
        stride: int = 1,
        atom_selection: str | Sequence[int] | None = None,
        chunk_size: int = 1000,
    ) -> None:
        return None

    def cluster_features(
        self,
        *,
        n_states: int | Literal["auto"] = 100,
    ) -> None:
        return None

    def compute_implied_timescales(self) -> None:
        return None

    def generate_free_energy_surface(
        self,
        cv1_name: str = "CV1",
        cv2_name: str = "CV2",
        bins: int = 100,
        temperature: float = 300.0,
        **_: object,
    ) -> Dict[str, Any]:
        return {
            "cv1": cv1_name,
            "cv2": cv2_name,
            "bins": int(bins),
            "temperature": float(temperature),
            "surface": [],
        }

    def create_state_table(self) -> None:
        return None

    def extract_representative_structures(self) -> None:
        return None

    def save_analysis_results(self) -> None:
        return None

    def plot_free_energy_surface(self, *, save_file: str) -> None:
        return None

    def plot_implied_timescales(self, *, save_file: str) -> None:
        return None

    def plot_implied_rates(self, *, save_file: str) -> None:
        return None

    def plot_free_energy_profile(self, *, save_file: str) -> None:
        return None

    def plot_ck_test(
        self,
        save_file: str = "ck_plot",
        n_macrostates: int = 3,
        factors: Optional[List[int]] = None,
        **_: object,
    ) -> Optional[Path]:
        return None


def run_complete_msm_analysis(
    trajectory_files: Union[str, List[str]],
    topology_file: str,
    output_dir: str = "output/msm_analysis",
    n_states: int | Literal["auto"] = 100,
    lag_time: int = 20,
    feature_type: str = "phi_psi",
    temperatures: Optional[List[float]] = None,
    stride: int = 1,
    atom_selection: str | Sequence[int] | None = None,
    chunk_size: int = 1000,
) -> "EnhancedMSMStub":
    raise ImportError(
        "EnhancedMSM full pipeline requires the optional scikit-learn dependency"
    )


__all__ = ["EnhancedMSMStub", "run_complete_msm_analysis"]
