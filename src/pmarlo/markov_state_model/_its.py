from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, Tuple

import numpy as np

from pmarlo import constants as const
from pmarlo.utils.validation import all_finite, any_finite

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

    def _deterministic_its_from_counts(
        self, lag: int, counts: np.ndarray, n_timescales: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]: ...
    def _counts_from_deeptime_backend(self, count_model: Any) -> np.ndarray: ...

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
        dirichlet_alpha: float = const.NUMERIC_DIRICHLET_ALPHA,
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

        self._its_fill_missing_timescales(
            result, lag_times, n_timescales, dirichlet_alpha
        )

        self._its_optionally_attach_plateau(
            result,
            lag_times,
            result.timescales.tolist(),
            plateau_m=plateau_m,
            plateau_epsilon=plateau_epsilon,
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
        import random as _random

        _random.seed(self.random_state)

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
        from deeptime.markov.msm import BayesianMSM  # type: ignore

        lag_int = int(max(1, lag))
        n_samples_int = int(max(1, n_samples))

        estimator = BayesianMSM(lagtime=lag_int, n_samples=n_samples_int)
        ignore_counting_mode = False
        count_mode = getattr(self, "count_mode", None)
        if count_mode is not None:
            mode_text = str(count_mode)
            if "effective" not in mode_text.lower():
                _logging.getLogger("pmarlo").warning(
                    "Bayesian MSM confidence sampling expects effective counts; "
                    "continuing with count_mode='%s' may yield correlated samples.",
                    mode_text,
                )
                ignore_counting_mode = True

        posterior = estimator.fit(
            self.dtrajs,
            lagtime=lag_int,
            count_mode=count_mode,
            ignore_counting_mode=ignore_counting_mode,
        ).fetch_model()

        samples = getattr(posterior, "samples", None)
        if not samples:
            raise RuntimeError(f"BayesianMSM returned no samples for lag {lag_int}")

        matrices = []
        for sample in samples:
            matrix = np.asarray(getattr(sample, "transition_matrix", None), dtype=float)
            if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
                raise ValueError(
                    "BayesianMSM sample transition matrix must be square; "
                    f"received shape {matrix.shape}"
                )
            matrices.append(matrix)

        matrices_arr = np.stack(matrices, axis=0)

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
        ) = self._summarize_its_stats(
            lag_int, matrices_arr, n_timescales, q_low, q_high
        )

        if np.all(rate_med < const.NUMERIC_MIN_POSITIVE):
            _logging.getLogger("pmarlo").warning(
                "Rates collapsed to near zero at lag %s; data may be insufficient",
                lag_int,
            )

        return (
            eval_med.tolist(),
            np.stack([eval_lo, eval_hi], axis=1).tolist(),
            ts_med.tolist(),
            np.stack([ts_lo, ts_hi], axis=1).tolist(),
            rate_med.tolist(),
            np.stack([rate_lo, rate_hi], axis=1).tolist(),
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
        from pmarlo.markov_state_model.results import ITSResult

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

    def _its_fill_missing_timescales(
        self,
        result,
        lag_times: List[int],
        n_timescales: int,
        dirichlet_alpha: float,
    ) -> None:
        if not lag_times:
            return
        if all_finite(result.timescales):
            return
        for i, lag in enumerate(lag_times):
            if any_finite(result.timescales[i]):
                continue
            res = self._counts_for_lag(int(lag), dirichlet_alpha)
            det_eval, det_ts, det_rate = self._deterministic_its_from_counts(
                int(lag), res.counts, n_timescales
            )
            result.eigenvalues[i] = det_eval
            result.timescales[i] = det_ts
            result.rates[i] = det_rate

    def _its_default_lag_times(self, lag_times: Optional[List[int]]) -> List[int]:
        if lag_times is None:
            # Extended lag sweep for more comprehensive ITS analysis
            # Includes powers of 2 (20, 40, 80, 160, 320, 640, 1280) for physically meaningful times
            return [
                1,
                2,
                3,
                5,
                8,
                10,
                15,
                20,
                30,
                40,
                50,
                75,
                80,
                100,
                150,
                160,
                200,
                320,
                640,
                1280,
            ]
        return [int(max(1, v)) for v in lag_times]

    def _validate_its_inputs(
        self, lag_times: List[int], n_timescales: int
    ) -> Optional[tuple[List[int], int]]:
        import logging as _logging

        if not getattr(self, "dtrajs", None):
            _logging.getLogger("pmarlo").warning(
                "No trajectories available for implied timescales"
            )
            from pmarlo.markov_state_model.results import ITSResult

            empty = ITSResult(
                lag_times=np.array([], dtype=int),
                eigenvalues=np.empty((0, n_timescales), dtype=float),
                eigenvalues_ci=np.empty((0, n_timescales, 2), dtype=float),
                timescales=np.empty((0, n_timescales), dtype=float),
                timescales_ci=np.empty((0, n_timescales, 2), dtype=float),
                rates=np.empty((0, n_timescales), dtype=float),
                rates_ci=np.empty((0, n_timescales, 2), dtype=float),
            )
            self.implied_timescales = empty
            return None

        max_valid_lag = min(len(dt) for dt in self.dtrajs) - 1
        if max_valid_lag < 1:
            _logging.getLogger("pmarlo").warning(
                "Trajectories too short for implied timescales"
            )
            from pmarlo.markov_state_model.results import ITSResult

            empty = ITSResult(
                lag_times=np.array([], dtype=int),
                eigenvalues=np.empty((0, n_timescales), dtype=float),
                eigenvalues_ci=np.empty((0, n_timescales, 2), dtype=float),
                timescales=np.empty((0, n_timescales), dtype=float),
                timescales_ci=np.empty((0, n_timescales, 2), dtype=float),
                rates=np.empty((0, n_timescales), dtype=float),
                rates_ci=np.empty((0, n_timescales, 2), dtype=float),
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
            from pmarlo.markov_state_model.results import ITSResult

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
        from ._msm_utils import ensure_connected_counts

        C = self._counts_from_deeptime_backend(lag)
        res = ensure_connected_counts(C, alpha=alpha)
        if getattr(res, "counts", np.array([])).size == 0:
            raise RuntimeError(f"Transition counts empty for lag {lag}")
        return res

    def _counts_from_deeptime_backend(self, lag: int) -> np.ndarray:
        from deeptime.markov import TransitionCountEstimator  # type: ignore

        tce = TransitionCountEstimator(
            lagtime=int(max(1, lag)), count_mode=self.count_mode, sparse=False
        )
        model = tce.fit(self.dtrajs).fetch_model()
        return np.asarray(model.count_matrix, dtype=float)

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
        from deeptime.markov.tools.analysis import eigenvalues as _dt_eigenvalues
        from deeptime.markov.tools.analysis import (
            stationary_distribution as _dt_stationary_distribution,
        )

        eig_samples: list[np.ndarray] = []
        ts_samples: list[np.ndarray] = []
        for T in matrices_arr:
            arr = np.asarray(T, dtype=float)
            if arr.size == 0:
                eig_samples.append(np.empty((0,), dtype=float))
                ts_samples.append(np.empty((0,), dtype=float))
                continue

            n_eval = int(max(0, n_timescales))

            pi = np.asarray(
                _dt_stationary_distribution(arr, check_inputs=False),
                dtype=float,
            )
            if pi.size == 0 or not np.all(np.isfinite(pi)):
                raise ValueError("Invalid stationary distribution in ITS analysis")

            k_request = n_eval + 1 if n_eval > 0 else 1
            if k_request >= arr.shape[0]:
                evals = np.asarray(_dt_eigenvalues(arr, k=None), dtype=complex)
            else:
                evals = np.asarray(_dt_eigenvalues(arr, k=k_request), dtype=complex)

            order = np.argsort(-np.real(evals))
            evals = np.asarray(evals[order], dtype=complex)
            eig = np.real(evals[1 : 1 + n_eval]) if n_eval > 0 else np.empty((0,))
            eig = np.clip(
                np.abs(eig),
                const.NUMERIC_MIN_POSITIVE,
                1.0 - const.NUMERIC_MIN_POSITIVE,
            )
            eig_samples.append(eig.astype(float))

            if n_eval > 0:
                ts = np.asarray(safe_timescales(int(max(1, lag)), eig), dtype=float)
            else:
                ts = np.empty((0,), dtype=float)
            ts_samples.append(np.asarray(ts, dtype=float))

        eig_arr = np.asarray(eig_samples, dtype=float)
        ts_arr = np.asarray(ts_samples, dtype=float)

        import warnings as _warnings

        with _warnings.catch_warnings():
            _warnings.filterwarnings("ignore", category=RuntimeWarning)

            eval_med = np.nanmedian(eig_arr, axis=0)
            eval_lo = np.nanpercentile(eig_arr, q_low, axis=0)
            eval_hi = np.nanpercentile(eig_arr, q_high, axis=0)
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
        count_model = self._bmsm_build_counts(count_mode)
        samples_model = self._bmsm_fit_samples(count_model, n_samples)
        ts_list = self._bmsm_collect_timescales(samples_model)
        pi_list = self._bmsm_collect_populations(samples_model)
        if not ts_list and not pi_list:
            return None
        return self._bmsm_finalize_output(ts_list, pi_list)

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

    def _deterministic_its_from_counts(
        self, lag: int, counts: np.ndarray, n_timescales: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        from deeptime.markov.tools.analysis import eigenvalues as _dt_eigenvalues
        from deeptime.markov.tools.analysis import timescales as _dt_timescales

        C_rev = 0.5 * (counts + counts.T)
        row = C_rev.sum(axis=1, keepdims=True)
        with np.errstate(invalid="ignore"):
            T = C_rev / np.where(row == 0, 1.0, row)
        # Similarity transform to symmetric
        pi = np.maximum(T.sum(axis=1), const.NUMERIC_MIN_POSITIVE_STRICT)
        pi = pi / np.sum(pi)
        n_states = T.shape[0]
        k_request = n_timescales + 1 if n_timescales > 0 else None
        if k_request is not None and k_request > n_states:
            k_eval = None
        else:
            k_eval = k_request
        eigvals = _dt_eigenvalues(T, k=k_eval, reversible=True, mu=pi)
        eigvals = np.asarray(eigvals, dtype=complex)
        eigvals = eigvals[np.argsort(-np.real(eigvals))]
        slow = (
            np.real(eigvals[1 : 1 + n_timescales])
            if n_timescales > 0
            else np.empty((0,), dtype=float)
        )
        slow = np.clip(
            np.abs(slow), const.NUMERIC_MIN_POSITIVE, 1.0 - const.NUMERIC_MIN_POSITIVE
        )
        evals = np.zeros((n_timescales,), dtype=float)
        evals[: slow.shape[0]] = slow

        if n_timescales > 0:
            if k_request is None:
                k_times = None
            else:
                k_times = min(n_states, n_timescales + 1)
            ts_raw = _dt_timescales(
                T,
                tau=int(max(1, lag)),
                k=k_times,
                reversible=True,
                mu=pi,
            )
            ts_raw = np.asarray(ts_raw, dtype=float)
            ts_arr = ts_raw[1 : 1 + n_timescales]
        else:
            ts_arr = np.empty((0,), dtype=float)
        if ts_arr.shape[0] < n_timescales:
            ts_arr = np.pad(
                ts_arr,
                (0, n_timescales - ts_arr.shape[0]),
                mode="constant",
                constant_values=np.nan,
            )
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
                if not all_finite(seg):
                    break
                mean_val = float(np.nanmean(seg))
                if mean_val <= 0 or not all_finite(mean_val):
                    continue
                if float(np.nanmax(seg) - np.nanmin(seg)) <= float(epsilon) * mean_val:
                    length = j - i + 1
                    if length > best_len:
                        best_len = length
                        best_interval = (float(lags[i]), float(lags[j]))
        return best_interval
