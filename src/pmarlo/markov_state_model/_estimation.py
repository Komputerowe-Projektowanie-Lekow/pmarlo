from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Protocol, Tuple

import numpy as np
from scipy.sparse import csc_matrix, issparse, save_npz

from pmarlo.states.msm_bridge import _row_normalize, _stationary_from_T
from pmarlo.utils.msm_utils import ensure_connected_counts


class _HasEstimationAttrs(Protocol):
    features: Optional[np.ndarray]
    dtrajs: List[np.ndarray]
    n_states: int
    count_mode: str
    effective_frames: int
    output_dir: Path
    tica_lag: int
    stationary_distribution: Optional[np.ndarray]
    free_energies: Optional[np.ndarray]
    lag_time: int
    count_matrix: Optional[np.ndarray]
    transition_matrix: Optional[np.ndarray]

    def _maybe_apply_tica(self, n_components: int, lag: int) -> None: ...

    def _build_tram_msm(self, lag_time: int) -> None: ...

    # Internal helpers used by EstimationMixin
    def _build_standard_msm(
        self, lag_time: int, count_mode: str = "sliding"
    ) -> None: ...

    def _validate_and_cap_lag(self, lag_time: int) -> tuple[int, int]: ...

    def _initialize_empty_msm(self) -> None: ...

    def _should_use_deeptime(self) -> bool: ...

    def _count_transitions_deeptime(
        self, *, lag: int, count_mode: str
    ) -> np.ndarray: ...

    def _count_transitions_locally(
        self, *, lag: int, count_mode: str
    ) -> np.ndarray: ...

    def _finalize_transition_and_stationary(self, counts: np.ndarray) -> None: ...

    def _compute_free_energies(self, temperature: float = 300.0) -> None: ...


class EstimationMixin:
    def build_msm(
        self: _HasEstimationAttrs, lag_time: int = 20, method: str = "standard"
    ) -> None:
        lag_time = int(max(1, lag_time))
        import logging as _logging

        _logging.getLogger("pmarlo").info(
            f"Building MSM with lag time {lag_time} using {method} method..."
        )

        self.lag_time = lag_time
        try:
            if self.features is not None and not hasattr(self, "tica_components_"):
                if self.features.shape[1] > 20:
                    _logging.getLogger("pmarlo").info(
                        "Applying default 3-component TICA prior to MSM to reduce noise"
                    )
                    # _maybe_apply_tica is provided by FeaturesMixin
                    self._maybe_apply_tica(3, getattr(self, "tica_lag", 0) or lag_time)
        except Exception:
            pass

        if method == "standard":
            self._build_standard_msm(lag_time, count_mode=self.count_mode)
        elif method == "tram":
            # Provided by TRAMMixin
            self._build_tram_msm(lag_time)
        else:
            raise ValueError(f"Unknown MSM method: {method}")

        # Compute free energies
        self._compute_free_energies()

        _logging.getLogger("pmarlo").info("MSM construction completed")

    def _build_standard_msm(
        self: _HasEstimationAttrs, lag_time: int, count_mode: str = "sliding"
    ) -> None:
        if getattr(self, "effective_frames", 0) and lag_time >= self.effective_frames:
            raise ValueError(
                f"lag_time {lag_time} exceeds available effective frames {self.effective_frames}"
            )

        lag, max_valid_lag = self._validate_and_cap_lag(lag_time)
        if max_valid_lag < 1:
            self._initialize_empty_msm()
            return

        use_deeptime = self._should_use_deeptime()
        if use_deeptime:
            counts = self._count_transitions_deeptime(lag=lag, count_mode=count_mode)
        else:
            counts = self._count_transitions_locally(lag=lag, count_mode=count_mode)

        self._finalize_transition_and_stationary(counts)

    def _validate_and_cap_lag(
        self: _HasEstimationAttrs, lag_time: int
    ) -> tuple[int, int]:
        lag = int(max(1, lag_time))
        max_valid_lag = min(len(dt) for dt in self.dtrajs) - 1 if self.dtrajs else 0
        if lag > max_valid_lag and max_valid_lag > 0:
            import logging as _logging

            _logging.getLogger("pmarlo").warning(
                "Lag %s exceeds max feasible %s; capping", lag, max_valid_lag
            )
            lag = max_valid_lag
        return lag, max_valid_lag

    def _count_transitions_deeptime(
        self: _HasEstimationAttrs, *, lag: int, count_mode: str
    ) -> np.ndarray:
        from deeptime.markov import TransitionCountEstimator  # type: ignore

        tce = TransitionCountEstimator(
            lagtime=lag,
            count_mode="sliding" if count_mode == "strided" else str(count_mode),
            sparse=False,
        )
        count_model = tce.fit(self.dtrajs).fetch_model()
        return np.asarray(count_model.count_matrix, dtype=float)

    def _count_transitions_locally(
        self: _HasEstimationAttrs, *, lag: int, count_mode: str
    ) -> np.ndarray:
        """Count transitions using vectorized numpy operations."""
        counts = np.zeros((self.n_states, self.n_states), dtype=float)
        step = lag if count_mode == "strided" else 1
        for dtraj in self.dtrajs:
            arr = np.asarray(dtraj, dtype=int)
            if arr.size <= lag:
                continue
            i_states = arr[:-lag:step]
            j_states = arr[lag::step]
            valid = (
                (i_states >= 0)
                & (j_states >= 0)
                & (i_states < self.n_states)
                & (j_states < self.n_states)
            )
            if not np.any(valid):
                continue
            np.add.at(counts, (i_states[valid], j_states[valid]), 1.0)
        return counts

    def _finalize_transition_and_stationary(
        self: _HasEstimationAttrs, counts: np.ndarray
    ) -> None:
        res = ensure_connected_counts(counts)
        cm = np.zeros((self.n_states, self.n_states), dtype=float)
        if res.counts.size:
            cm[np.ix_(res.active, res.active)] = res.counts
            T_active = _row_normalize(res.counts)
            pi_active = _stationary_from_T(T_active)
            T_full = np.eye(self.n_states, dtype=float)
            T_full[np.ix_(res.active, res.active)] = T_active
            pi_full = np.zeros((self.n_states,), dtype=float)
            pi_full[res.active] = pi_active
        else:
            T_full = np.eye(self.n_states, dtype=float)
            pi_full = np.zeros((self.n_states,), dtype=float)
        self.count_matrix = cm
        self.transition_matrix = T_full
        self.stationary_distribution = pi_full

    # Utilities shared with export
    def _initialize_empty_msm(self: _HasEstimationAttrs) -> None:
        self.count_matrix = np.zeros((self.n_states, self.n_states), dtype=float)
        self.transition_matrix = np.eye(self.n_states, dtype=float)
        self.stationary_distribution = np.zeros((self.n_states,), dtype=float)

    def _should_use_deeptime(self: _HasEstimationAttrs) -> bool:
        if getattr(self, "estimator_backend", "deeptime") != "deeptime":
            return False
        try:  # pragma: no cover
            from deeptime.markov import TransitionCountEstimator  # type: ignore

            _ = TransitionCountEstimator
            return True
        except Exception:  # pragma: no cover
            return False

    def _compute_free_energies(self: _HasEstimationAttrs, temperature: float = 300.0):
        from scipy import constants

        if self.stationary_distribution is None:
            raise ValueError("Stationary distribution must be computed first")

        kT = constants.k * temperature * constants.Avogadro / 1000.0  # kJ/mol
        pi_safe = np.maximum(self.stationary_distribution, 1e-12)
        self.free_energies = -kT * np.log(pi_safe)
        self.free_energies -= np.min(self.free_energies)

    def _create_matrix_intelligent(
        self: _HasEstimationAttrs,
        shape: Tuple[int, int],
        use_sparse: bool | None = None,
    ):
        n_states = shape[0]
        if use_sparse is None:
            use_sparse = n_states > 100
        if use_sparse:
            return csc_matrix(shape, dtype=np.float64)
        else:
            return np.zeros(shape, dtype=np.float64)

    def _matrix_add_count(
        self: _HasEstimationAttrs, matrix, i: int, j: int, count: float
    ):
        if issparse(matrix):
            matrix[i, j] += count
        else:
            matrix[i, j] += count

    def _matrix_normalize_rows(self: _HasEstimationAttrs, matrix):
        if issparse(matrix):
            row_sums = np.array(matrix.sum(axis=1)).flatten()
            row_sums[row_sums == 0] = 1
            diag = csc_matrix(
                (1.0 / row_sums, (range(len(row_sums)), range(len(row_sums)))),
                shape=(len(row_sums), len(row_sums)),
            )
            return diag @ matrix
        else:
            row_sums = matrix.sum(axis=1)
            row_sums[row_sums == 0] = 1
            return matrix / row_sums[:, np.newaxis]

    def _save_matrix_intelligent(
        self: _HasEstimationAttrs,
        matrix,
        filename_base: str,
        prefix: str = "msm_analysis",
    ) -> None:
        if matrix is None:
            return
        import numpy as _np
        from scipy.sparse import issparse as _issparse

        _np.save(
            self.output_dir / f"{prefix}_{filename_base}.npy",
            matrix.toarray() if _issparse(matrix) else matrix,
        )
        if matrix.size > 10000:
            if _issparse(matrix):
                save_npz(self.output_dir / f"{prefix}_{filename_base}.npz", matrix)
            else:
                sparsity = _np.count_nonzero(matrix) / matrix.size
                if sparsity < 0.05:
                    sparse_matrix = csc_matrix(matrix)
                    save_npz(
                        self.output_dir / f"{prefix}_{filename_base}.npz",
                        sparse_matrix,
                    )
