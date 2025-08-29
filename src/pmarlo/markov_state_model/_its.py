from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, Tuple

import numpy as np

from .utils import safe_timescales


class _SupportsITS(Protocol):
    random_state: Optional[int]
    dtrajs: List[np.ndarray]
    count_mode: str
    lag_time: int
    n_states: int
    time_per_frame_ps: Optional[float]
    implied_timescales: Any

    # Helper methods used within ITSMixin
    def _its_log_start(self, _logging) -> None: ...

    def _its_default_lag_times(self, lag_times: Optional[List[int]]) -> List[int]: ...

    def _validate_its_inputs(
        self, lag_times: List[int], n_timescales: int
    ) -> Optional[Tuple[List[int], int]]: ...

    def _its_seed_rng_if_available(self) -> None: ...

    def _its_compute_for_all_lags(
        self,
        *,
        lag_times: List[int],
        n_timescales: int,
        n_samples: int,
        ci: float,
        dirichlet_alpha: float,
        _logging,
    ) -> Tuple[
        List[List[float]],
        List[List[List[float]]],
        List[List[float]],
        List[List[List[float]]],
        List[List[float]],
        List[List[List[float]]],
    ]: ...

    def _its_build_result(
        self,
        lag_times: List[int],
        eval_means: List[List[float]],
        eval_ci: List[List[List[float]]],
        ts_means: List[List[float]],
        ts_ci: List[List[List[float]]],
        rate_means: List[List[float]],
        rate_ci: List[List[List[float]]],
    ) -> Any: ...

    def _its_optionally_attach_plateau(
        self,
        result: Any,
        lag_times: List[int],
        ts_means: List[List[float]],
        *,
        plateau_m: Optional[int],
        plateau_epsilon: float,
    ) -> None: ...

    def _its_fill_missing_timescales(
        self,
        result: Any,
        lag_times: List[int],
        n_timescales: int,
        dirichlet_alpha: float,
    ) -> None: ...

    def _its_compute_for_single_lag(
        self,
        *,
        lag: int,
        n_timescales: int,
        n_samples: int,
        dirichlet_alpha: float,
        q_low: float,
        q_high: float,
        _logging,
    ) -> Tuple[
        List[float],
        List[List[float]],
        List[float],
        List[List[float]],
        List[float],
        List[List[float]],
    ]: ...

    def _counts_for_lag(self, lag: int, dirichlet_alpha: float) -> Any: ...

    def _its_empty_counts_fallback(self, n_timescales: int) -> Tuple[
        List[float],
        List[List[float]],
        List[float],
        List[List[float]],
        List[float],
        List[List[float]],
    ]: ...

    def _bayesian_transition_samples(
        self, counts: np.ndarray, n_samples: int
    ) -> np.ndarray: ...

    def _its_sampling_failed_fallback(
        self, lag: int, counts: np.ndarray, n_timescales: int
    ) -> Tuple[
        List[float],
        List[List[float]],
        List[float],
        List[List[float]],
        List[float],
        List[List[float]],
    ]: ...

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

    def _deterministic_its_from_empty(
        self, n_timescales: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]: ...

    def _deterministic_its_from_counts(
        self, lag: int, counts: np.ndarray, n_timescales: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]: ...

    def _counts_backend_matrix_or_fallback(self, *args: Any, **kwargs: Any) -> Any: ...

    def _counts_from_deeptime_backend(
        self, count_model: Any
    ) -> Optional[np.ndarray]: ...

    def _counts_from_dtrajs_fallback(
        self, *args: Any, **kwargs: Any
    ) -> Optional[np.ndarray]: ...

    def _infer_number_of_states(self) -> int: ...

    def _bmsm_build_counts(self, count_mode: str) -> Any: ...

    def _bmsm_fit_samples(self, count_model: Any, n_samples: int) -> Any: ...

    def _bmsm_collect_timescales(self, samples_model: Any) -> List[np.ndarray]: ...

    def _bmsm_collect_populations(self, samples_model: Any) -> List[np.ndarray]: ...

    def _bmsm_finalize_output(
        self, ts_list: List[np.ndarray], pi_list: List[np.ndarray]
    ) -> Dict[str, Any]: ...


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
    ) -> None:
        import logging as _logging

        self._its_log_start(_logging)

        lag_times = self._its_default_lag_times(lag_times)

        validated = self._validate_its_inputs(lag_times, n_timescales)
        if validated is None:
            return
        lag_times, _max_valid_lag = validated

        self._its_seed_rng_if_available()

        (
            eval_means,
            eval_ci,
            ts_means,
            ts_ci,
            rate_means,
            rate_ci,
        ) = self._its_compute_for_all_lags(
            lag_times=lag_times,
            n_timescales=n_timescales,
            n_samples=n_samples,
            ci=ci,
            dirichlet_alpha=dirichlet_alpha,
            _logging=_logging,
        )

        result = self._its_build_result(
            lag_times, eval_means, eval_ci, ts_means, ts_ci, rate_means, rate_ci
        )

        self._its_optionally_attach_plateau(
            result,
            lag_times,
            ts_means,
            plateau_m=plateau_m,
            plateau_epsilon=plateau_epsilon,
        )

        self._its_fill_missing_timescales(
            result, lag_times, n_timescales, dirichlet_alpha
        )
        self.implied_timescales = result

    # ---- helpers for compute_implied_timescales (split to reduce complexity) ----
    def _its_log_start(self, _logging) -> None:
        _logging.getLogger("pmarlo").info(
            "Computing implied timescales with Bayesian estimation"
        )

    # Host attributes provided by the concrete class
    random_state: Optional[int]
    dtrajs: List[np.ndarray]
    count_mode: str
    lag_time: int
    n_states: int
    time_per_frame_ps: Optional[float]
    implied_timescales: Any

    def _its_seed_rng_if_available(self) -> None:
        if getattr(self, "random_state", None) is None:
            return
        np.random.seed(self.random_state)
        try:
            import random as _random

            _random.seed(self.random_state)
        except Exception:
            pass

    def _its_alpha_tail_bounds(self, ci: float) -> Tuple[float, float]:
        alpha_tail = 50.0 * (1.0 - ci)
        return alpha_tail, 100.0 - alpha_tail

    def _its_compute_for_all_lags(
        self,
        *,
        lag_times: List[int],
        n_timescales: int,
        n_samples: int,
        ci: float,
        dirichlet_alpha: float,
        _logging,
    ) -> Tuple[
        List[List[float]],
        List[List[List[float]]],
        List[List[float]],
        List[List[List[float]]],
        List[List[float]],
        List[List[List[float]]],
    ]:
        eval_means: List[List[float]] = []
        eval_ci: List[List[List[float]]] = []
        ts_means: List[List[float]] = []
        ts_ci: List[List[List[float]]] = []
        rate_means: List[List[float]] = []
        rate_ci: List[List[List[float]]] = []

        q_low, q_high = self._its_alpha_tail_bounds(ci)

        for lag in lag_times:
            (
                em,
                eci,
                tsm,
                tsci,
                rm,
                rci,
            ) = self._its_compute_for_single_lag(
                lag=lag,
                n_timescales=n_timescales,
                n_samples=n_samples,
                dirichlet_alpha=dirichlet_alpha,
                q_low=q_low,
                q_high=q_high,
                _logging=_logging,
            )
            eval_means.append(em)
            eval_ci.append(eci)
            ts_means.append(tsm)
            ts_ci.append(tsci)
            rate_means.append(rm)
            rate_ci.append(rci)

        return eval_means, eval_ci, ts_means, ts_ci, rate_means, rate_ci

    def _its_compute_for_single_lag(
        self,
        *,
        lag: int,
        n_timescales: int,
        n_samples: int,
        dirichlet_alpha: float,
        q_low: float,
        q_high: float,
        _logging,
    ) -> Tuple[
        List[float],
        List[List[float]],
        List[float],
        List[List[float]],
        List[float],
        List[List[float]],
    ]:
        res = self._counts_for_lag(lag, dirichlet_alpha)
        if res is None:
            return self._its_empty_counts_fallback(n_timescales)

        matrices_arr = self._bayesian_transition_samples(res.counts, n_samples)
        if matrices_arr.size == 0:
            return self._its_sampling_failed_fallback(lag, res.counts, n_timescales)

        (
            eval_med,
            eval_lo,
            eval_hi,
            ts_med,
            ts_lo,
            ts_hi,
            rate_med,
            rate_lo,
            rate_hi,
        ) = self._summarize_its_stats(lag, matrices_arr, n_timescales, q_low, q_high)

        if np.all(rate_med < 1e-12):
            _logging.getLogger("pmarlo").warning(
                "Rates collapsed to near zero at lag %s; data may be insufficient",
                lag,
            )

        return (
            eval_med.tolist(),
            np.stack([eval_lo, eval_hi], axis=1).tolist(),
            ts_med.tolist(),
            np.stack([ts_lo, ts_hi], axis=1).tolist(),
            rate_med.tolist(),
            np.stack([rate_lo, rate_hi], axis=1).tolist(),
        )

    def _its_empty_counts_fallback(self, n_timescales: int) -> Tuple[
        List[float],
        List[List[float]],
        List[float],
        List[List[float]],
        List[float],
        List[List[float]],
    ]:
        det_eval, det_ts, det_rate = self._deterministic_its_from_empty(n_timescales)
        nan_ci = [[np.nan, np.nan]] * n_timescales
        return (
            det_eval.tolist(),
            nan_ci,
            det_ts.tolist(),
            nan_ci,
            det_rate.tolist(),
            nan_ci,
        )

    def _its_sampling_failed_fallback(
        self, lag: int, counts: np.ndarray, n_timescales: int
    ) -> Tuple[
        List[float],
        List[List[float]],
        List[float],
        List[List[float]],
        List[float],
        List[List[float]],
    ]:
        det_eval, det_ts, det_rate = self._deterministic_its_from_counts(
            lag, counts, n_timescales
        )
        nan_ci = [[np.nan, np.nan]] * n_timescales
        return (
            det_eval.tolist(),
            nan_ci,
            det_ts.tolist(),
            nan_ci,
            det_rate.tolist(),
            nan_ci,
        )

    def _its_build_result(
        self,
        lag_times: List[int],
        eval_means: List[List[float]],
        eval_ci: List[List[List[float]]],
        ts_means: List[List[float]],
        ts_ci: List[List[List[float]]],
        rate_means: List[List[float]],
        rate_ci: List[List[List[float]]],
    ):
        from pmarlo.results import ITSResult

        return ITSResult(
            lag_times=np.asarray(lag_times, dtype=int),
            eigenvalues=np.asarray(eval_means, dtype=float),
            eigenvalues_ci=np.asarray(eval_ci, dtype=float),
            timescales=np.asarray(ts_means, dtype=float),
            timescales_ci=np.asarray(ts_ci, dtype=float),
            rates=np.asarray(rate_means, dtype=float),
            rates_ci=np.asarray(rate_ci, dtype=float),
        )

    def _its_optionally_attach_plateau(
        self,
        result,
        lag_times: List[int],
        ts_means: List[List[float]],
        *,
        plateau_m: Optional[int],
        plateau_epsilon: float,
    ) -> None:
        try:
            if plateau_m is not None and plateau_m >= 1 and len(lag_times) > 1:
                window = self._detect_timescale_plateau(
                    np.asarray(lag_times, dtype=float),
                    np.asarray(ts_means, dtype=float),
                    int(plateau_m),
                    float(plateau_epsilon),
                )
                if window is not None:
                    dt_ps = float(getattr(self, "time_per_frame_ps", 1.0) or 1.0)
                    start_ps = float(window[0] * dt_ps)
                    end_ps = float(window[1] * dt_ps)
                    result.recommended_lag_window = (start_ps, end_ps)
        except Exception:
            pass

    def _its_fill_missing_timescales(
        self,
        result,
        lag_times: List[int],
        n_timescales: int,
        dirichlet_alpha: float,
    ) -> None:
        if not lag_times:
            return
        if not np.isnan(result.timescales).any():
            return
        for i, lag in enumerate(lag_times):
            if not np.isnan(result.timescales[i]).all():
                continue
            res = self._counts_for_lag(int(lag), dirichlet_alpha)
            if res is None:
                continue
            det_eval, det_ts, det_rate = self._deterministic_its_from_counts(
                int(lag), res.counts, n_timescales
            )
            result.eigenvalues[i] = det_eval
            result.timescales[i] = det_ts
            result.rates[i] = det_rate

    def _its_default_lag_times(self, lag_times: Optional[List[int]]) -> List[int]:
        if lag_times is None:
            return [1, 2, 3, 5, 8, 10, 15, 20, 30, 50, 75, 100, 150, 200]
        return [int(max(1, v)) for v in lag_times]

    def _validate_its_inputs(
        self, lag_times: List[int], n_timescales: int
    ) -> Optional[tuple[List[int], int]]:
        import logging as _logging

        if not getattr(self, "dtrajs", None):
            _logging.getLogger("pmarlo").warning(
                "No trajectories available for implied timescales"
            )
            from pmarlo.results import ITSResult

            empty = ITSResult(
                lag_times=np.array([], dtype=int),
                eigenvalues=np.empty((0, n_timescales)),
                eigenvalues_ci=np.empty((0, n_timescales, 2)),
                timescales=np.empty((0, n_timescales)),
                timescales_ci=np.empty((0, n_timescales, 2)),
                rates=np.empty((0, n_timescales)),
                rates_ci=np.empty((0, n_timescales, 2)),
            )
            self.implied_timescales = empty
            return None

        max_valid_lag = min(len(dt) for dt in self.dtrajs) - 1
        if max_valid_lag < 1:
            _logging.getLogger("pmarlo").warning(
                "Trajectories too short for implied timescales"
            )
            from pmarlo.results import ITSResult

            empty = ITSResult(
                lag_times=np.array([], dtype=int),
                eigenvalues=np.empty((0, n_timescales)),
                eigenvalues_ci=np.empty((0, n_timescales, 2)),
                timescales=np.empty((0, n_timescales)),
                timescales_ci=np.empty((0, n_timescales, 2)),
                rates=np.empty((0, n_timescales)),
                rates_ci=np.empty((0, n_timescales, 2)),
            )
            self.implied_timescales = empty
            return None

        original_lag_times = list(lag_times)
        if any(lag_val > max_valid_lag for lag_val in original_lag_times):
            _logging.getLogger("pmarlo").warning(
                "Capping lag times above max_valid_lag=%s", max_valid_lag
            )
        lag_times = [lt for lt in original_lag_times if 1 <= lt <= max_valid_lag]
        if not lag_times:
            _logging.getLogger("pmarlo").warning("No valid lag times after capping")
            from pmarlo.results import ITSResult

            empty = ITSResult(
                lag_times=np.array([], dtype=int),
                eigenvalues=np.empty((0, n_timescales)),
                eigenvalues_ci=np.empty((0, n_timescales, 2)),
                timescales=np.empty((0, n_timescales)),
                timescales_ci=np.empty((0, n_timescales, 2)),
                rates=np.empty((0, n_timescales)),
                rates_ci=np.empty((0, n_timescales, 2)),
            )
            self.implied_timescales = empty
            return None

        max_lag = max(lag_times)
        eff = getattr(self, "effective_frames", None)
        if eff is not None and eff > 0 and max_lag >= eff:
            raise ValueError(
                f"Maximum lag {max_lag} exceeds available effective frames {eff}"
            )
        return lag_times, max_valid_lag

    def _counts_for_lag(self, lag: int, alpha: float):
        from pmarlo.utils.msm_utils import ensure_connected_counts

        C = self._counts_backend_matrix_or_fallback(lag)
        if C is None:
            return None

        res = ensure_connected_counts(C, alpha=alpha)
        if getattr(res, "counts", np.array([])).size == 0:
            return None
        return res

    # ---- helpers for _counts_for_lag ----
    def _counts_backend_matrix_or_fallback(self, lag: int) -> Optional[np.ndarray]:
        C = self._counts_from_deeptime_backend(lag)
        if C is not None:
            return C
        return self._counts_from_dtrajs_fallback(lag)

    def _counts_from_deeptime_backend(self, lag: int) -> Optional[np.ndarray]:
        try:  # pragma: no cover
            from deeptime.markov import TransitionCountEstimator  # type: ignore

            tce = TransitionCountEstimator(
                lagtime=int(max(1, lag)), count_mode=self.count_mode, sparse=False
            )
            model = tce.fit(self.dtrajs).fetch_model()
            return np.asarray(model.count_matrix, dtype=float)
        except Exception:
            return None

    def _infer_number_of_states(self) -> Optional[int]:
        n_states = int(getattr(self, "n_states", 0) or 0)
        if n_states > 0:
            return n_states
        try:
            return int(max(int(np.max(seq)) for seq in self.dtrajs) + 1)
        except Exception:
            return None

    def _counts_from_dtrajs_fallback(self, lag: int) -> Optional[np.ndarray]:
        n_states_opt = self._infer_number_of_states()
        if n_states_opt is None or n_states_opt <= 0:
            return None
        n_states = int(n_states_opt)
        C = np.zeros((n_states, n_states), dtype=float)
        step = int(lag) if str(self.count_mode) == "strided" else 1
        for seq in self.dtrajs:
            seq = np.asarray(seq, dtype=int)
            if seq.size <= lag:
                continue
            upper = seq.size - int(lag)
            for i in range(0, upper, step):
                a = int(seq[i])
                b = int(seq[i + int(lag)])
                if 0 <= a < n_states and 0 <= b < n_states:
                    C[a, b] += 1.0
        return C

    def _bayesian_transition_samples(
        self, counts: np.ndarray, n_samples: int
    ) -> np.ndarray:
        C_rev = 0.5 * (counts + counts.T)
        row_sums = C_rev.sum(axis=1, keepdims=True)
        matrices: list[np.ndarray] = []
        for _ in range(n_samples):
            T_sample = np.zeros_like(C_rev)
            for i in range(C_rev.shape[0]):
                T_sample[i] = np.random.dirichlet(C_rev[i])
            C_sample = T_sample * row_sums
            C_sym = 0.5 * (C_sample + C_sample.T)
            T = C_sym / C_sym.sum(axis=1, keepdims=True)
            matrices.append(T)
        return np.asarray(matrices, dtype=float)

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
    ]:
        eig_samples: list[np.ndarray] = []
        for T in matrices_arr:
            # Compute stationary distribution robustly as the leading left eigenvector
            try:
                evals, evecs = np.linalg.eig(T.T)
                idx1 = int(np.argmax(np.real(evals)))
                v = np.real(evecs[:, idx1]).astype(float)
                v = np.where(v < 0, -v, v)
                if np.allclose(v, 0):
                    raise ValueError
                pi = v / np.sum(v)
            except Exception:
                pi = np.full((T.shape[0],), 1.0 / max(1, T.shape[0]))
            sym = np.diag(np.sqrt(pi)) @ T @ np.diag(1.0 / np.sqrt(pi))
            sym = 0.5 * (sym + sym.T)
            eigenvals = np.linalg.eigvalsh(sym)
            eigenvals = np.sort(eigenvals)[::-1]
            eig = eigenvals[1 : n_timescales + 1]
            # Robustify against tiny negative rounding; keep within (0,1)
            eig = np.clip(np.abs(eig), 1e-12, 1 - 1e-12)
            eig_samples.append(eig)
        eig_arr = np.asarray(eig_samples, dtype=float)

        import warnings as _warnings

        with _warnings.catch_warnings():
            _warnings.filterwarnings("ignore", category=RuntimeWarning)

            eval_med = np.nanmedian(eig_arr, axis=0)
            eval_lo = np.nanpercentile(eig_arr, q_low, axis=0)
            eval_hi = np.nanpercentile(eig_arr, q_high, axis=0)

            ts_arr = safe_timescales(lag, eig_arr)
            rate_arr = np.reciprocal(
                ts_arr, where=np.isfinite(ts_arr), out=np.full_like(ts_arr, np.nan)
            )

            ts_med = np.nanmedian(ts_arr, axis=0)
            ts_lo = np.nanpercentile(ts_arr, q_low, axis=0)
            ts_hi = np.nanpercentile(ts_arr, q_high, axis=0)

            rate_med = np.nanmedian(rate_arr, axis=0)
            rate_lo = np.nanpercentile(rate_arr, q_low, axis=0)
            rate_hi = np.nanpercentile(rate_arr, q_high, axis=0)

        # Ensure consistent output length n_timescales by padding with NaN when needed
        def _pad1d(vec: np.ndarray, length: int) -> np.ndarray:
            out = np.full((length,), np.nan, dtype=float)
            k = min(vec.shape[0], length)
            out[:k] = vec[:k]
            return out

        def _pad_ci(lo: np.ndarray, hi: np.ndarray, length: int) -> np.ndarray:
            out = np.full((length, 2), np.nan, dtype=float)
            k = min(lo.shape[0], hi.shape[0], length)
            out[:k, 0] = lo[:k]
            out[:k, 1] = hi[:k]
            return out

        eval_med = _pad1d(eval_med, int(n_timescales))
        eval_lo_hi = _pad_ci(eval_lo, eval_hi, int(n_timescales))
        eval_lo = eval_lo_hi[:, 0]
        eval_hi = eval_lo_hi[:, 1]

        ts_med = _pad1d(ts_med, int(n_timescales))
        ts_lo_hi = _pad_ci(ts_lo, ts_hi, int(n_timescales))
        ts_lo = ts_lo_hi[:, 0]
        ts_hi = ts_lo_hi[:, 1]

        rate_med = _pad1d(rate_med, int(n_timescales))
        rate_lo_hi = _pad_ci(rate_lo, rate_hi, int(n_timescales))
        rate_lo = rate_lo_hi[:, 0]
        rate_hi = rate_lo_hi[:, 1]

        return (
            eval_med,
            eval_lo,
            eval_hi,
            ts_med,
            ts_lo,
            ts_hi,
            rate_med,
            rate_lo,
            rate_hi,
        )

    def sample_bayesian_timescales(
        self: "_SupportsITS", n_samples: int = 200, count_mode: str = "effective"
    ) -> Optional[Dict[str, Any]]:
        try:
            count_model = self._bmsm_build_counts(count_mode)
            samples_model = self._bmsm_fit_samples(count_model, n_samples)
            ts_list = self._bmsm_collect_timescales(samples_model)
            pi_list = self._bmsm_collect_populations(samples_model)
            if not ts_list and not pi_list:
                return None
            return self._bmsm_finalize_output(ts_list, pi_list)
        except Exception:
            return None

    def _bmsm_build_counts(self: "_SupportsITS", count_mode: str) -> Any:
        from deeptime.markov import TransitionCountEstimator  # type: ignore

        tce = TransitionCountEstimator(
            lagtime=int(max(1, self.lag_time)),
            count_mode=str(count_mode),
            sparse=False,
        )
        return tce.fit(self.dtrajs).fetch_model()

    def _bmsm_fit_samples(self, count_model: Any, n_samples: int) -> Any:
        from deeptime.markov.msm import BayesianMSM  # type: ignore

        bmsm = BayesianMSM(reversible=True, n_samples=int(max(1, n_samples)))
        return bmsm.fit(count_model).fetch_model()

    def _bmsm_collect_timescales(self, samples_model: Any) -> List[np.ndarray]:
        ts_list: List[np.ndarray] = []
        for sm in getattr(samples_model, "samples", []):  # type: ignore[attr-defined]
            T = np.asarray(getattr(sm, "transition_matrix", None), dtype=float)
            if T.size == 0:
                continue
            evals = np.sort(np.real(np.linalg.eigvals(T)))[::-1]
            ts = safe_timescales(self.lag_time, evals[1 : min(6, len(evals))])
            ts = ts[np.isfinite(ts)]
            if ts.size:
                ts_list.append(ts)
        return ts_list

    def _bmsm_collect_populations(self, samples_model: Any) -> List[np.ndarray]:
        pi_list: List[np.ndarray] = []
        for sm in getattr(samples_model, "samples", []):  # type: ignore[attr-defined]
            if (
                hasattr(sm, "stationary_distribution")
                and sm.stationary_distribution is not None
            ):
                pi_list.append(np.asarray(sm.stationary_distribution, dtype=float))
        return pi_list

    def _bmsm_finalize_output(self, ts_list, pi_list) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if ts_list:
            maxlen = max(arr.shape[0] for arr in ts_list)
            ts_pad = [
                np.pad(a, (0, maxlen - a.shape[0]), constant_values=np.nan)
                for a in ts_list
            ]
            out["timescales_samples"] = np.vstack(ts_pad)
        if pi_list:
            maxn = max(a.shape[0] for a in pi_list)
            pi_pad = [
                (
                    a
                    if a.shape[0] == maxn
                    else np.pad(a, (0, maxn - a.shape[0]), constant_values=np.nan)
                )
                for a in pi_list
            ]
            out["population_samples"] = np.vstack(pi_pad)
        return out

    def _deterministic_its_from_empty(
        self, n_timescales: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        evals = np.full((n_timescales,), np.nan)
        ts = np.full((n_timescales,), np.nan)
        rates = np.full((n_timescales,), np.nan)
        return evals, ts, rates

    def _deterministic_its_from_counts(
        self, lag: int, counts: np.ndarray, n_timescales: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        C_rev = 0.5 * (counts + counts.T)
        row = C_rev.sum(axis=1, keepdims=True)
        with np.errstate(invalid="ignore"):
            T = C_rev / np.where(row == 0, 1.0, row)
        # Similarity transform to symmetric
        pi = np.maximum(T.sum(axis=1), 1e-15)
        pi = pi / np.sum(pi)
        sym = np.diag(np.sqrt(pi)) @ T @ np.diag(1.0 / np.sqrt(pi))
        sym = 0.5 * (sym + sym.T)
        w = np.sort(np.linalg.eigvalsh(sym))[::-1]
        eig = w[1 : n_timescales + 1]
        eig = np.clip(np.abs(eig), 1e-12, 1 - 1e-12)
        evals = np.zeros((n_timescales,), dtype=float)
        evals[: eig.shape[0]] = eig
        ts_arr = safe_timescales(lag, evals)
        rates = np.reciprocal(
            ts_arr, where=np.isfinite(ts_arr), out=np.full_like(ts_arr, np.nan)
        )
        return evals, ts_arr, rates

    def _detect_timescale_plateau(
        self,
        lag_times: np.ndarray,
        timescales_mean: np.ndarray,
        m: int,
        epsilon: float,
    ) -> Optional[tuple[float, float]]:
        """Detect a plateau in the slowest timescale series.

        Plateau = longest contiguous interval [i, j] with length >= m such that
        range(ts[i:j+1]) <= epsilon * mean(ts[i:j+1]). Uses the slowest
        (first) timescale column.
        """
        if timescales_mean.size == 0 or lag_times.size == 0:
            return None
        ts = np.asarray(timescales_mean, dtype=float)[:, 0]
        lags = np.asarray(lag_times, dtype=float)
        n = ts.shape[0]
        if n == 0:
            return None
        best_len = 0
        best_interval: Optional[tuple[float, float]] = None
        for i in range(n):
            for j in range(i + max(1, int(m)) - 1, n):
                seg = ts[i : j + 1]
                if not np.all(np.isfinite(seg)):
                    break
                mean_val = float(np.nanmean(seg))
                if mean_val <= 0 or np.isnan(mean_val):
                    continue
                if float(np.nanmax(seg) - np.nanmin(seg)) <= float(epsilon) * mean_val:
                    length = j - i + 1
                    if length > best_len:
                        best_len = length
                        best_interval = (float(lags[i]), float(lags[j]))
        return best_interval
