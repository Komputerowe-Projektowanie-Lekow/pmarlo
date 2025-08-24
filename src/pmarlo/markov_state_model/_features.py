from __future__ import annotations

from typing import List, Optional

import mdtraj as md
import numpy as np


class FeaturesMixin:
    def compute_features(
        self,
        feature_type: str = "phi_psi",
        n_features: Optional[int] = None,
        feature_stride: int = 1,
        tica_lag: int = 0,
        tica_components: Optional[int] = None,
    ) -> None: ...

    def _compute_features_for_traj(
        self, traj: md.Trajectory, feature_type: str, n_features: Optional[int]
    ) -> np.ndarray: ...

    def _compute_phi_psi_features(self, traj: md.Trajectory) -> np.ndarray: ...

    def _compute_phi_psi_plus_distance_features(
        self, traj: md.Trajectory, n_distance_features: Optional[int]
    ) -> np.ndarray: ...

    def _compute_distance_features(
        self, traj: md.Trajectory, n_features: Optional[int]
    ) -> np.ndarray: ...

    def _compute_contact_features(self, traj: md.Trajectory) -> np.ndarray: ...

    def _combine_all_features(self, feature_blocks: List[np.ndarray]) -> np.ndarray: ...
