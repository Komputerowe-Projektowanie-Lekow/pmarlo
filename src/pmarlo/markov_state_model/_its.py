from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .utils import safe_timescales


class ITSMixin:
    def compute_implied_timescales(
        self,
        lag_times: Optional[List[int]] = None,
        n_timescales: int = 5,
        *,
        n_samples: int = 100,
        ci: float = 0.95,
        dirichlet_alpha: float = 1e-3,
        plateau_m: int | None = None,
        plateau_epsilon: float = 0.1,
    ) -> None: ...

    def _its_default_lag_times(self, lag_times: Optional[List[int]]) -> List[int]: ...

    def _validate_its_inputs(
        self, lag_times: List[int], n_timescales: int
    ) -> Optional[tuple[List[int], int]]: ...

    def _counts_for_lag(self, lag: int, alpha: float): ...

    def _bayesian_transition_samples(
        self, counts: np.ndarray, n_samples: int
    ) -> np.ndarray: ...

    def _summarize_its_stats(
        self,
        lag: int,
        matrices_arr: np.ndarray,
        n_timescales: int,
        q_low: float,
        q_high: float,
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]: ...

    def sample_bayesian_timescales(
        self, n_samples: int = 200, count_mode: str = "effective"
    ) -> Optional[Dict[str, Any]]: ...

    def _bmsm_build_counts(self, count_mode: str) -> Any: ...

    def _bmsm_fit_samples(self, count_model: Any, n_samples: int) -> Any: ...

    def _bmsm_collect_timescales(self, samples_model: Any) -> List[np.ndarray]: ...

    def _bmsm_collect_populations(self, samples_model: Any) -> List[np.ndarray]: ...

    def _bmsm_finalize_output(self, ts_list, pi_list) -> Dict[str, Any]: ...
