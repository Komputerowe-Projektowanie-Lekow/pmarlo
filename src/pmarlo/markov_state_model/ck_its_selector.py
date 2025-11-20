from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from deeptime.markov import TransitionCountEstimator
from deeptime.markov.msm import MaximumLikelihoodMSM
from scipy.sparse.csgraph import connected_components

from pmarlo import constants as const
from pmarlo.utils.msm_utils import (
    _row_normalize,
    _stationary_from_T,
    pcca_like_macrostates,
)

logger = logging.getLogger("pmarlo")


@dataclass
class LagEvaluationResult:
    """Results from evaluating a single candidate lag time."""

    lag: int
    ck_error: float
    coverage_fraction: float
    median_count: int
    n_macrostates: int
    n_microstates: int
    passed_sanity: bool
    failure_reason: Optional[str] = None
    timescales: Optional[np.ndarray] = None
    eigenvalue_gap: Optional[float] = None
    diag_mass: Optional[float] = None


def _max_supported_lag(dtrajs: Sequence[np.ndarray]) -> int:
    """Return the largest lag that still yields at least one transition."""

    max_length = 0
    for traj in dtrajs:
        if traj is None:
            continue
        length = int(getattr(traj, "size", 0))
        if length > max_length:
            max_length = length
    return max(0, max_length - 1)


def _filter_tau_candidates(
    dtrajs: Sequence[np.ndarray],
    tau_candidates: Sequence[int],
) -> tuple[list[int], list[int], int]:
    """Filter tau candidates that exceed the available trajectory length."""

    max_supported_lag = _max_supported_lag(dtrajs)
    valid: list[int] = []
    ignored: list[int] = []
    for tau in tau_candidates:
        if tau <= max_supported_lag:
            valid.append(int(tau))
        else:
            ignored.append(int(tau))
    return valid, ignored, max_supported_lag


def _count_transitions(
    dtrajs: Sequence[np.ndarray], n_states: int, lag: int
) -> np.ndarray:
    """Count transitions at a given lag time."""
    C = np.zeros((n_states, n_states), dtype=float)
    for traj in dtrajs:
        if traj.size <= lag:
            continue
        for i in range(traj.size - lag):
            a = int(traj[i])
            b = int(traj[i + lag])
            if 0 <= a < n_states and 0 <= b < n_states:
                C[a, b] += 1.0
    return C


def _compute_coverage_fraction(C: np.ndarray) -> float:
    """Compute the fraction of states in the largest connected component."""
    if C.size == 0:
        return 0.0

    adj = ((C + C.T) > 0).astype(int)
    n_components, labels = connected_components(adj, directed=False, return_labels=True)

    if n_components == 0:
        return 0.0

    # Find largest component
    counts = np.bincount(labels)
    largest_size = int(np.max(counts))
    total_size = int(C.shape[0])

    return float(largest_size) / float(total_size) if total_size > 0 else 0.0


def _compute_median_count(C: np.ndarray) -> int:
    """Compute median microstate count from transition count matrix."""
    if C.size == 0:
        return 0
    state_counts = C.sum(axis=0) + C.sum(axis=1)
    return (
        int(np.median(state_counts[state_counts > 0]))
        if np.any(state_counts > 0)
        else 0
    )


def _auto_determine_macrostates(
    T: np.ndarray, min_macro: int = 3, max_macro: int = 6
) -> int:
    """Automatically determine number of macrostates based on eigenvalue gap.

    Uses PCCA+ approach: find the largest eigenvalue gap in the range [min_macro, max_macro].
    """
    if T.size == 0 or T.shape[0] < min_macro:
        return min_macro

    try:
        evals = np.linalg.eigvals(T)
        evals = np.sort(np.real(evals))[::-1]

        if len(evals) < min_macro + 1:
            return min_macro

        # Compute gaps for candidate macrostate counts
        max_gap = 0.0
        best_m = min_macro

        for m in range(min_macro, min(max_macro + 1, len(evals))):
            if m >= len(evals):
                break
            gap = float(evals[m - 1] - evals[m])
            if gap > max_gap:
                max_gap = gap
                best_m = m

        logger.debug(
            "[CK-ITS] Auto-determined %d macrostates (gap=%.4f)", best_m, max_gap
        )
        return best_m

    except Exception as e:
        logger.warning(
            "[CK-ITS] Failed to determine macrostates via eigenvalues: %s", e
        )
        return min_macro


def _compute_predicted_macro_kinetics(
    T_tau: np.ndarray,
    pi: np.ndarray,
    chi: np.ndarray,
    k: int,
) -> np.ndarray:
    """Compute predicted macrostate transition matrix at horizon k.

    T_macro = (χ^T diag(π) T_τ^k χ)(χ^T diag(π) χ)^(-1)
    """
    n_micro = T_tau.shape[0]
    n_macro = chi.shape[1]

    # Power of microstate transition matrix
    T_k = np.linalg.matrix_power(T_tau, k)

    # Weighted projection: χ^T diag(π)
    chi_T_pi = chi.T @ np.diag(pi)

    # Numerator: χ^T diag(π) T^k χ
    numerator = chi_T_pi @ T_k @ chi

    # Denominator: χ^T diag(π) χ
    denominator = chi_T_pi @ chi

    # Regularize to avoid singular matrix
    denominator += np.eye(n_macro) * const.NUMERIC_MIN_POSITIVE

    # Solve for T_macro
    T_macro_pred = numerator @ np.linalg.inv(denominator)

    return T_macro_pred


def _compute_observed_macro_kinetics(
    dtrajs: Sequence[np.ndarray],
    macro_labels: np.ndarray,
    n_macro: int,
    lag: int,
) -> np.ndarray:
    """Compute observed macrostate transition matrix at lag time."""
    # Map microstate trajectories to macrostate trajectories
    macro_trajs = [macro_labels[traj] for traj in dtrajs]

    # Count transitions
    C_macro = _count_transitions(macro_trajs, n_macro, lag)

    # Normalize to transition matrix
    T_macro_obs = _row_normalize(C_macro)

    return T_macro_obs


def _compute_ck_error(T_pred: np.ndarray, T_obs: np.ndarray) -> float:
    """Compute relative CK error between predicted and observed matrices.

    relerr = ||T_pred - T_obs||_1 / ||T_obs||_1
    """
    if T_pred.shape != T_obs.shape:
        raise ValueError(f"Matrix shape mismatch: {T_pred.shape} vs {T_obs.shape}")

    diff = T_pred - T_obs
    l1_diff = float(np.sum(np.abs(diff)))
    l1_obs = float(np.sum(np.abs(T_obs)))

    if l1_obs < const.NUMERIC_MIN_POSITIVE:
        return float("inf")

    return l1_diff / l1_obs


def _check_sanity_criteria(
    coverage: float,
    median_count: int,
    coverage_threshold: float,
    min_median_count: int,
) -> Tuple[bool, Optional[str]]:
    """Check if sanity criteria are met."""
    if coverage < coverage_threshold:
        return False, f"Coverage {coverage:.2%} < {coverage_threshold:.2%}"

    if median_count < min_median_count:
        return False, f"Median count {median_count} < {min_median_count}"

    return True, None


def _compute_microstate_ck_error(
    T_tau: np.ndarray,
    dtrajs: Sequence[np.ndarray],
    n_states: int,
    lag: int,
    horizons: List[int],
) -> float:
    """Compute CK error at microstate level (fallback when PCCA+ fails).

    Uses direct microstate transition matrices without coarse-graining.
    """
    max_ck_error = 0.0
    for k in horizons:
        # Predicted: T^k
        T_pred = np.linalg.matrix_power(T_tau, k)

        # Observed at lag k*tau
        C_obs = _count_transitions(dtrajs, n_states, lag * k)
        T_obs = _row_normalize(C_obs)

        # Compute error
        ck_error_k = _compute_ck_error(T_pred, T_obs)
        max_ck_error = max(max_ck_error, ck_error_k)

        logger.debug(
            "[CK-ITS] Microstate CK: lag %d, horizon k=%d, error=%.4f",
            lag,
            k,
            ck_error_k,
        )

    return max_ck_error


def _evaluate_single_lag(
    dtrajs: Sequence[np.ndarray],
    lag: int,
    horizons: List[int],
    n_states: int,
    coverage_threshold: float,
    min_median_count: int,
    diag_mass_threshold: float,
) -> LagEvaluationResult:
    """Evaluate a single candidate lag time."""
    logger.info("[CK-ITS] Evaluating lag=%d", lag)

    try:
        # Build transition count matrix
        C_tau = _count_transitions(dtrajs, n_states, lag)

        # Check coverage
        coverage = _compute_coverage_fraction(C_tau)
        median_count = _compute_median_count(C_tau)

        # Check sanity criteria
        passed_sanity, failure_reason = _check_sanity_criteria(
            coverage, median_count, coverage_threshold, min_median_count
        )

        if not passed_sanity:
            logger.warning("[CK-ITS] Lag %d failed sanity: %s", lag, failure_reason)
            return LagEvaluationResult(
                lag=lag,
                ck_error=float("inf"),
                coverage_fraction=coverage,
                median_count=median_count,
                n_macrostates=0,
                n_microstates=n_states,
                passed_sanity=False,
                failure_reason=failure_reason,
            )

        use_microstate_ck = False

        # Normalize to transition matrix
        T_tau = _row_normalize(C_tau)

        # Compute stationary distribution
        pi = _stationary_from_T(T_tau)

        # Try macrostate CK test first (preferred)
        max_ck_error = None
        n_macro = 0
        eigenvalue_gap = None
        use_microstate_ck = False

        # Auto-determine number of macrostates
        n_macro_candidate = _auto_determine_macrostates(T_tau, min_macro=2, max_macro=6)

        # Attempt PCCA+ decomposition
        macro_labels = None
        try:
            macro_labels = pcca_like_macrostates(T_tau, n_macrostates=n_macro_candidate)
        except Exception as pcca_err:
            logger.warning("[CK-ITS] PCCA+ exception for lag %d: %s", lag, pcca_err)

        if macro_labels is not None:
            # Macrostate CK test (preferred)
            logger.debug("[CK-ITS] Using macrostate CK test for lag %d", lag)
            n_macro = n_macro_candidate

            # Build fuzzy membership matrix χ
            # For crisp assignments, χ is one-hot encoded
            n_micro = T_tau.shape[0]
            chi = np.zeros((n_micro, n_macro), dtype=float)
            for i, label in enumerate(macro_labels):
                chi[i, int(label)] = 1.0

            # Compute CK error across horizons
            max_ck_error = 0.0
            for k in horizons:
                # Predicted macrostate kinetics
                T_pred = _compute_predicted_macro_kinetics(T_tau, pi, chi, k)

                # Observed macrostate kinetics at horizon k
                T_obs = _compute_observed_macro_kinetics(
                    dtrajs, macro_labels, n_macro, lag * k
                )

                # Compute error
                ck_error_k = _compute_ck_error(T_pred, T_obs)
                max_ck_error = max(max_ck_error, ck_error_k)

                logger.debug(
                    "[CK-ITS] Macrostate CK: lag %d, horizon k=%d, error=%.4f",
                    lag,
                    k,
                    ck_error_k,
                )

            # Compute eigenvalue gap
            try:
                evals = np.linalg.eigvals(T_tau)
                evals = np.sort(np.real(evals))[::-1]
                if len(evals) > n_macro:
                    eigenvalue_gap = float(evals[n_macro - 1] - evals[n_macro])
            except Exception:
                pass
        else:
            # Fallback to microstate CK test
            logger.warning(
                "[CK-ITS] PCCA+ failed for lag %d, using microstate CK test fallback",
                lag,
            )
            use_microstate_ck = True
            max_ck_error = _compute_microstate_ck_error(
                T_tau, dtrajs, n_states, lag, horizons
            )

        # Compute timescales for ITS and diagonal mass guardrail
        diag_mass = float("nan")
        try:
            estimator = MaximumLikelihoodMSM(lagtime=lag, reversible=True)
            msm_model = estimator.fit(dtrajs).fetch_model()
            timescales = np.asarray(msm_model.timescales(), dtype=float)
            transition = np.asarray(msm_model.transition_matrix, dtype=float)
            if transition.size:
                diag_mass = float(np.trace(transition) / transition.shape[0])
        except Exception as e:
            logger.warning(
                "[CK-ITS] Failed to compute timescales for lag %d: %s", lag, e
            )
            timescales = None
        diag_failure_reason: Optional[str] = None
        diag_ok = np.isfinite(diag_mass) and diag_mass >= diag_mass_threshold
        if not diag_ok:
            diag_failure_reason = (
                f"Diagonal mass {diag_mass:.3f} < threshold {diag_mass_threshold:.3f}"
                if np.isfinite(diag_mass)
                else "Diagonal mass undefined"
            )
            logger.warning(
                "[CK-ITS] Lag %d failed diagonal-mass guardrail: %s",
                lag,
                diag_failure_reason,
            )

        mode_text = "microstate" if use_microstate_ck else "macrostate"
        logger.info(
            "[CK-ITS] Lag %d (%s): CK error=%.4f, coverage=%.2f%%, median_count=%d, diag_mass=%.3f, n_macro=%d",
            lag,
            mode_text,
            max_ck_error,
            coverage * 100,
            median_count,
            diag_mass,
            n_macro,
        )

        return LagEvaluationResult(
            lag=lag,
            ck_error=max_ck_error,
            coverage_fraction=coverage,
            median_count=median_count,
            n_macrostates=n_macro,
            n_microstates=n_states,
            passed_sanity=diag_failure_reason is None,
            failure_reason=diag_failure_reason,
            timescales=timescales,
            eigenvalue_gap=eigenvalue_gap,
            diag_mass=diag_mass,
        )

    except Exception as e:
        logger.error("[CK-ITS] Failed to evaluate lag %d: %s", lag, e, exc_info=True)
        return LagEvaluationResult(
            lag=lag,
            ck_error=float("inf"),
            coverage_fraction=0.0,
            median_count=0,
            n_macrostates=0,
            n_microstates=n_states,
            passed_sanity=False,
            failure_reason=f"Exception: {str(e)}",
        )


def select_optimal_lag_ck_its(
    dtrajs: Sequence[np.ndarray],
    tau_candidates: Optional[List[int]] = None,
    horizons: Optional[List[int]] = None,
    ck_threshold: float = 0.15,
    coverage_threshold: float = 0.98,
    min_median_count: int = 100,
    diag_mass_threshold: float = 0.6,
) -> Tuple[int, List[LagEvaluationResult]]:
    """Select optimal lag time using CK test with ITS validation.

    Selects the smallest lag τ that:
    1. Passes CK test: max CK error ≤ ck_threshold (default 10-15%)
    2. Has high coverage: giant connected component ≥ coverage_threshold
    3. Has sufficient statistics: median microstate count ≥ min_median_count
    4. Exhibits adequate diagonal mass: trace(T)/n ≥ diag_mass_threshold

    Parameters
    ----------
    dtrajs : Sequence[np.ndarray]
        List of discrete trajectory arrays.
    tau_candidates : Optional[List[int]]
        Candidate lag times to evaluate. If None, uses [25, 50, 75, 100].
    horizons : Optional[List[int]]
        CK test horizons k (multiples of base lag). If None, uses [1, 2, 3, 4, 5].
    ck_threshold : float
        Maximum acceptable CK error (default: 0.15).
    coverage_threshold : float
        Minimum coverage fraction (default: 0.98).
    min_median_count : int
        Minimum median microstate count (default: 100).
    diag_mass_threshold : float
        Minimum acceptable diagonal mass averaged across states (default: 0.6).

    Returns
    -------
    selected_lag : int
        The selected optimal lag time.
    evaluations : List[LagEvaluationResult]
        Detailed evaluation results for all candidate lags.
    """
    if not dtrajs or len(dtrajs) == 0:
        raise ValueError("No discrete trajectories provided")

    usable_dtrajs = [traj for traj in dtrajs if traj is not None and traj.size > 0]
    if not usable_dtrajs:
        raise ValueError(
            "Discrete trajectories contain no frames for CK analysis; "
            "provide trajectories with at least two time steps."
        )

    if tau_candidates is None:
        tau_candidates = [25, 50, 75, 100]
    else:
        tau_candidates = list(tau_candidates)

    if horizons is None:
        horizons = [1, 2, 3, 4, 5]

    (
        valid_tau_candidates,
        ignored_tau_candidates,
        max_supported_lag,
    ) = _filter_tau_candidates(usable_dtrajs, tau_candidates)

    if ignored_tau_candidates:
        logger.warning(
            "[CK-ITS] Ignoring %d tau candidates that exceed available length "
            "(max supported lag=%d): %s",
            len(ignored_tau_candidates),
            max_supported_lag,
            ignored_tau_candidates,
        )

    if not valid_tau_candidates:
        raise ValueError(
            "All tau candidates exceed the available trajectory length "
            f"(max supported lag {max_supported_lag}). "
            "Provide smaller lag values or shorter horizons."
        )

    # Infer number of microstates
    n_states = int(max(np.max(dt) for dt in usable_dtrajs)) + 1

    logger.info(
        "[CK-ITS] Starting lag selection: candidates=%s, horizons=%s, n_states=%d",
        valid_tau_candidates,
        horizons,
        n_states,
    )

    # Evaluate each candidate lag
    evaluations: List[LagEvaluationResult] = []
    for lag in sorted(valid_tau_candidates):
        result = _evaluate_single_lag(
            usable_dtrajs,
            lag,
            horizons,
            n_states,
            coverage_threshold,
            min_median_count,
            diag_mass_threshold,
        )
        evaluations.append(result)

    # Select smallest lag that passes all criteria
    selected_lag = None
    for result in sorted(evaluations, key=lambda r: r.lag):
        if result.passed_sanity and result.ck_error <= ck_threshold:
            selected_lag = result.lag
            logger.info(
                "[CK-ITS] Selected lag=%d (CK error=%.4f ≤ %.4f)",
                selected_lag,
                result.ck_error,
                ck_threshold,
            )
            break

    if selected_lag is None:
        # Fallback: select lag with smallest CK error among those passing sanity
        passing_sanity = [r for r in evaluations if r.passed_sanity]
        if passing_sanity:
            best = min(passing_sanity, key=lambda r: r.ck_error)
            selected_lag = best.lag
            logger.warning(
                "[CK-ITS] No lag passed CK threshold; selecting best lag=%d (CK error=%.4f)",
                selected_lag,
                best.ck_error,
            )
        else:
            # Ultimate fallback: smallest lag
            selected_lag = min(tau_candidates)
            logger.warning(
                "[CK-ITS] No lag passed sanity checks; using smallest lag=%d",
                selected_lag,
            )

    return selected_lag, evaluations
