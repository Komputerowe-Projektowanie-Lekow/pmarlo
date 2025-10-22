from __future__ import annotations

from pathlib import Path
from typing import Any, List

import numpy as np
import pytest

from pmarlo.markov_state_model._clustering import ClusteringMixin
from pmarlo.markov_state_model._estimation import EstimationMixin
from pmarlo.markov_state_model._fes import FESMixin


class _DummyTrajectory:
    def __init__(self, n_frames: int) -> None:
        self.n_frames = n_frames


class _DummyClustering(ClusteringMixin):
    def __init__(self) -> None:
        self.features = np.random.rand(5, 2)
        self.random_state = 0
        self.trajectories: List[Any] = [_DummyTrajectory(5)]
        self.dtrajs: List[np.ndarray] = []
        self.cluster_centers = None


class _DummyEstimator(EstimationMixin):
    def __init__(self, output_dir: Path) -> None:
        self.features = np.random.rand(5, 2)
        self.dtrajs = [np.zeros(5, dtype=int)]
        self.n_states = 0
        self.count_mode = "sliding"
        self.effective_frames = 10
        self.output_dir = output_dir
        self.tica_lag = 0
        self.stationary_distribution = None
        self.free_energies = None
        self.lag_time = 0
        self.count_matrix = None
        self.transition_matrix = None

    # Minimal protocol implementations
    def _maybe_apply_tica(
        self, n_components: int, lag: int
    ) -> None:  # pragma: no cover - stub
        return None

    def _build_tram_msm(self, lag_time: int) -> None:  # pragma: no cover - stub
        raise AssertionError("TRAM path should not be invoked in this test")

    def _build_standard_msm(
        self, lag_time: int, count_mode: str = "sliding"
    ) -> None:  # pragma: no cover - stub
        raise AssertionError("Guard should prevent standard MSM build")

    def _validate_and_cap_lag(
        self, lag_time: int
    ) -> tuple[int, int]:  # pragma: no cover - stub
        return lag_time, 10

    def _initialize_empty_msm(self) -> None:  # pragma: no cover - stub
        return None

    def _ensure_deeptime_backend(self) -> None:  # pragma: no cover - stub
        raise AssertionError("MSM guard should abort before backend validation")

    def _count_transitions_deeptime(
        self, *, lag: int, count_mode: str
    ) -> np.ndarray:  # pragma: no cover - stub
        raise AssertionError("Deeptime counting should not occur")

    def _finalize_transition_and_stationary(
        self, counts: np.ndarray
    ) -> None:  # pragma: no cover - stub
        return None

    def _compute_free_energies(
        self, temperature: float = 300.0
    ) -> None:  # pragma: no cover - stub
        return None


class _DummyFES(FESMixin):
    def __init__(self) -> None:
        self.features = np.random.rand(5, 2)
        self.stationary_distribution = np.ones(1, dtype=float)
        self.lag_time = 1
        self.dtrajs: List[np.ndarray] = [np.zeros(5, dtype=int)]
        self.trajectories: List[Any] = []
        self.fes_data = None
        self.n_states = 0

    # The following helpers should never be reached once the guard triggers
    def _extract_collective_variables(
        self, cv1_name: str, cv2_name: str
    ) -> tuple[np.ndarray, np.ndarray]:  # pragma: no cover - stub
        raise AssertionError("FES guard should trigger before extracting CVs")

    def _map_stationary_to_frame_weights(self) -> np.ndarray:  # pragma: no cover - stub
        raise AssertionError("FES guard should trigger before mapping weights")

    def _choose_bins(
        self, total_frames: int, user_bins: int
    ) -> int:  # pragma: no cover - stub
        return user_bins

    def _align_data_lengths(
        self,
        cv1_data: np.ndarray,
        cv2_data: np.ndarray,
        frame_weights_array: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:  # pragma: no cover - stub
        raise AssertionError("FES guard should trigger before aligning data")

    def _compute_weighted_histogram(
        self,
        cv1_data: np.ndarray,
        cv2_data: np.ndarray,
        frame_weights_array: np.ndarray,
        bins: int,
        ranges: list[tuple[float, float]] | None = None,
        smooth_sigma: float | None = None,
        periodic: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:  # pragma: no cover - stub
        raise AssertionError("FES guard should trigger before histogram computation")

    def _histogram_to_free_energy(
        self, H: np.ndarray, temperature: float
    ) -> np.ndarray:  # pragma: no cover - stub
        raise AssertionError("FES guard should trigger before free energy conversion")

    def _store_fes_result(
        self,
        F: np.ndarray,
        xedges: np.ndarray,
        yedges: np.ndarray,
        cv1_name: str,
        cv2_name: str,
        temperature: float,
    ) -> None:  # pragma: no cover - stub
        raise AssertionError("FES guard should trigger before storing results")


def test_cluster_features_rejects_non_positive_states():
    dummy = _DummyClustering()
    with pytest.raises(ValueError, match="microstate"):
        dummy.cluster_features(n_states=0)


def test_build_msm_requires_defined_states(tmp_path: Path):
    dummy = _DummyEstimator(output_dir=tmp_path)
    with pytest.raises(ValueError, match="microstate"):
        dummy.build_msm(lag_time=1)


def test_fes_generation_requires_states():
    dummy = _DummyFES()
    with pytest.raises(ValueError, match="microstate"):
        dummy.generate_free_energy_surface()
