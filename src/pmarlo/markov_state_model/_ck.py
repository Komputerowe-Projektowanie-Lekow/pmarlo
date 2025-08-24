from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from scipy.sparse.csgraph import connected_components

from ._base import CKTestResult


class CKMixin:
    def compute_ck_test_micro(
        self,
        factors: Optional[List[int]] = None,
        max_states: int = 50,
        min_transitions: int = 5,
    ) -> CKTestResult: ...

    def compute_ck_test_macrostates(
        self,
        n_macrostates: int = 3,
        factors: Optional[List[int]] = None,
        min_transitions: int = 5,
    ) -> CKTestResult: ...

    def select_lag_time_ck(
        self, tau_candidates: List[int], factor: int = 2, mse_epsilon: float = 0.05
    ) -> int: ...

    def _normalize_ck_factors(self, factors: Optional[List[int]]) -> List[int]: ...

    def _largest_connected_states(
        self, C: np.ndarray, max_states: int
    ) -> np.ndarray: ...

    def _count_micro_T(
        self, dtrajs: List[np.ndarray], nS: int, lag: int
    ) -> Tuple[np.ndarray, np.ndarray]: ...

    def _count_macro_T_and_counts(
        self, macro_trajs: List[np.ndarray], nM: int, lag: int
    ) -> Tuple[np.ndarray, np.ndarray]: ...

    def _ck_mse_for_factor(
        self,
        T1: np.ndarray,
        macro_trajs: List[np.ndarray],
        nM: int,
        base_lag: int,
        factor: int,
    ) -> Tuple[Optional[float], np.ndarray]: ...

    def _micro_eigen_gap(self, k: int = 2) -> Optional[float]: ...
