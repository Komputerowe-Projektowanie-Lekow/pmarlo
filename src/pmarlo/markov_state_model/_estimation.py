from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.sparse import csc_matrix, issparse, save_npz

from pmarlo.states.msm_bridge import _row_normalize, _stationary_from_T
from pmarlo.utils.msm_utils import ensure_connected_counts


class EstimationMixin:
    def build_msm(self, lag_time: int = 20, method: str = "standard") -> None: ...

    def _build_standard_msm(
        self, lag_time: int, count_mode: str = "sliding"
    ) -> None: ...

    def _validate_and_cap_lag(self, lag_time: int) -> tuple[int, int]: ...

    def _count_transitions_deeptime(
        self, *, lag: int, count_mode: str
    ) -> np.ndarray: ...

    def _count_transitions_locally(
        self, *, lag: int, count_mode: str
    ) -> np.ndarray: ...

    def _finalize_transition_and_stationary(self, counts: np.ndarray) -> None: ...

    def _create_matrix_intelligent(
        self, shape: Tuple[int, int], use_sparse: bool | None = None
    ): ...

    def _matrix_add_count(self, matrix, i: int, j: int, count: float): ...

    def _matrix_normalize_rows(self, matrix): ...

    def _save_matrix_intelligent(
        self, matrix, filename_base: str, prefix: str = "msm_analysis"
    ) -> None: ...
