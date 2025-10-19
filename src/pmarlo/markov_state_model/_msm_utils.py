from __future__ import annotations

import logging
from typing import Any, cast

import numpy as np
from deeptime.markov import pcca as _deeptime_pcca  # type: ignore
from deeptime.markov.tools.analysis import (
    stationary_distribution as _dt_stationary_distribution,  # type: ignore
)
from deeptime.markov.tools.estimation.dense.transition_matrix import (
    transition_matrix_non_reversible as _dt_row_normalize,  # type: ignore
)

from pmarlo import constants as const
from pmarlo.utils import msm_utils as _shared_msm_utils

logger = logging.getLogger("pmarlo")

ConnectedCountResult = _shared_msm_utils.ConnectedCountResult


def candidate_lag_ladder(
    min_lag: int = 1,
    max_lag: int = 200,
    n_candidates: int | None = None,
) -> list[int]:
    """Return the curated MSM lag ladder from :mod:`pmarlo.utils.msm_utils`."""

    return _shared_msm_utils.candidate_lag_ladder(
        min_lag=min_lag, max_lag=max_lag, n_candidates=n_candidates
    )


def ensure_connected_counts(
    C: np.ndarray,
    alpha: float = const.NUMERIC_DIRICHLET_ALPHA,
    epsilon: float = const.NUMERIC_MIN_POSITIVE,
) -> ConnectedCountResult:
    """Delegate to :func:`pmarlo.utils.msm_utils.ensure_connected_counts`."""

    return _shared_msm_utils.ensure_connected_counts(C, alpha=alpha, epsilon=epsilon)


def check_transition_matrix(
    T: np.ndarray,
    pi: np.ndarray,
    *,
    row_tol: float = const.NUMERIC_MIN_POSITIVE,
    stat_tol: float = const.NUMERIC_RELATIVE_TOLERANCE,
) -> None:
    """Delegate to :func:`pmarlo.utils.msm_utils.check_transition_matrix`."""

    _shared_msm_utils.check_transition_matrix(T, pi, row_tol=row_tol, stat_tol=stat_tol)


def _row_normalize(C: np.ndarray) -> np.ndarray[Any, Any]:
    """Row-normalize a matrix using :mod:`deeptime` utilities."""
    arr = np.asarray(C, dtype=float)
    if arr.size == 0:
        return cast(np.ndarray[Any, Any], arr.copy())
    return cast(np.ndarray[Any, Any], _dt_row_normalize(arr))


def _stationary_from_T(T: np.ndarray) -> np.ndarray:
    """Compute a stationary distribution using :mod:`deeptime`.

    Raises any numerical issues instead of attempting to silently recover.
    """
    arr = np.asarray(T, dtype=float)
    if arr.size == 0:
        return cast(np.ndarray, np.asarray([], dtype=float))

    pi_dt = _dt_stationary_distribution(arr, check_inputs=False)
    return cast(np.ndarray, np.asarray(pi_dt, dtype=float))


def _canonicalize_macro_labels(labels: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Renumber macrostate labels to ensure deterministic, consecutive ids."""
    if labels.size == 0:
        return labels.astype(int)
    pi_micro = _stationary_from_T(T)
    pops = compute_macro_populations(pi_micro, labels)
    unique = np.unique(labels)
    order = np.argsort(-pops[unique])
    mapping = {int(unique[idx]): int(i) for i, idx in enumerate(order)}
    return np.asarray([mapping[int(lbl)] for lbl in labels], dtype=int)


def compute_macro_populations(
    pi_micro: np.ndarray, micro_to_macro: np.ndarray
) -> np.ndarray:
    """Aggregate micro stationary distribution into macro populations."""
    n_macro = int(np.max(micro_to_macro)) + 1 if micro_to_macro.size else 0
    pi_macro = np.zeros((n_macro,), dtype=float)
    for m in range(n_macro):
        idx = np.where(micro_to_macro == m)[0]
        if idx.size:
            pi_macro[m] = float(np.sum(pi_micro[idx]))
    s = float(np.sum(pi_macro))
    if s > 0:
        pi_macro /= s
    return pi_macro


def lump_micro_to_macro_T(
    T_micro: np.ndarray, pi_micro: np.ndarray, micro_to_macro: np.ndarray
) -> np.ndarray:
    """Lump micro transition matrix into macro via stationary flux aggregation.

    F_AB = sum_{i in A} sum_{j in B} pi_i T_ij; then T_macro[A,B] = F_AB / sum_B F_AB.
    """
    n_macro = int(np.max(micro_to_macro)) + 1 if micro_to_macro.size else 0
    F = np.zeros((n_macro, n_macro), dtype=float)
    for i in range(T_micro.shape[0]):
        A = int(micro_to_macro[i])
        for j in range(T_micro.shape[1]):
            B = int(micro_to_macro[j])
            F[A, B] += float(pi_micro[i] * T_micro[i, j])
    rows = F.sum(axis=1)
    rows[rows == 0] = 1.0
    return cast(np.ndarray, F / rows[:, None])


def compute_macro_mfpt(T_macro: np.ndarray) -> np.ndarray:
    """Compute MFPTs between macrostates for a discrete-time Markov chain.

    For each target j, solve (I - Q) t = 1 where Q is T with row/col j removed.
    """
    n = T_macro.shape[0]
    mfpt = np.zeros((n, n), dtype=float)
    identity_matrix = np.eye(n, dtype=float)
    for j in range(n):
        # Remove j to form Q
        mask = np.ones((n,), dtype=bool)
        mask[j] = False
        Q = T_macro[np.ix_(mask, mask)]
        A = identity_matrix[: n - 1, : n - 1] - Q
        b = np.ones((n - 1,), dtype=float)
        try:
            t = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            t = np.full((n - 1,), np.nan)
        # Insert back into mfpt[:, j]
        mfpt[mask, j] = t
        mfpt[j, j] = 0.0
    return mfpt


def build_simple_msm(
    dtrajs: list[np.ndarray],
    n_states: int | None = None,
    lag: int = 20,
    count_mode: str = "sliding",
) -> tuple[np.ndarray, np.ndarray]:
    """Build MSM using deeptime estimators.

    Requires deeptime library to be installed.

    Returns a pair (transition_matrix, stationary_distribution).
    """
    if not dtrajs:
        logger.error("build_simple_msm: No dtrajs provided")
        return np.empty((0, 0), dtype=float), np.empty((0,), dtype=float)

    n_states = _infer_n_states(dtrajs, n_states)
    logger.info(f"build_simple_msm: Using {n_states} states")

    # Deeptime-based estimation (required dependency)
    T, pi = _fit_msm_deeptime(dtrajs, n_states, lag, count_mode)
    logger.info(f"build_simple_msm: Transition matrix shape: {T.shape}")
    logger.info(f"build_simple_msm: Stationary distribution shape: {pi.shape}")
    check_transition_matrix(T, pi)
    return T, pi


def _infer_n_states(dtrajs: list[np.ndarray], n_states: int | None) -> int:
    """
    Infer number of microstates from provided labels when not specified.
    """
    if n_states is not None:
        logger.debug(f"infer_n_states: States provided, using {n_states}")
        return int(n_states)
    # Start below zero so that trajectories with only negative labels
    # (often used as "unassigned") do not contribute to the count.
    max_state = -1
    for dt in dtrajs:
        if dt.size:
            m = int(np.max(dt))
            if m >= 0:
                max_state = max(max_state, m)
    n = int(max_state + 1) if max_state >= 0 else 0
    logger.debug(f"infer_n_states: Using {n} states")
    return n


def _fit_msm_deeptime(
    dtrajs: list[np.ndarray],
    n_states: int,
    lag: int,
    count_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit MSM using deeptime library (required dependency).

    Uses TransitionCountEstimator to estimate the transition matrix,
    then normalizes and computes stationary distribution.

    Parameters
    ----------
    dtrajs : list[np.ndarray]
        Discrete trajectories (state sequences).
    n_states : int
        Total number of states in the model.
    lag : int
        Lag time for counting transitions.
    count_mode : str
        Counting mode ("sliding" or "strided").

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (transition_matrix, stationary_distribution)
    """
    from deeptime.markov import TransitionCountEstimator  # type: ignore

    tce = TransitionCountEstimator(
        lagtime=int(max(1, lag)),
        count_mode=str(count_mode),
        sparse=False,
    )
    count_model = tce.fit(dtrajs).fetch_model()
    C_raw = np.asarray(count_model.count_matrix, dtype=float)
    res = ensure_connected_counts(C_raw)
    if res.counts.size == 0:
        return _expand_results(
            n_states,
            res.active,
            np.empty((0, 0), dtype=float),
            np.empty((0,), dtype=float),
        )
    T_active = _row_normalize(res.counts)
    pi_active = _stationary_from_T(T_active)
    return _expand_results(n_states, res.active, T_active, pi_active)


def _expand_results(
    n_states: int, active: np.ndarray, T_active: np.ndarray, pi_active: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Expand MSM results back to the original state space."""
    T_full = np.eye(n_states, dtype=float)
    pi_full = np.zeros((n_states,), dtype=float)
    if active.size:
        T_full[np.ix_(active, active)] = T_active
        pi_full[active] = pi_active
    return T_full, pi_full


def pcca_like_macrostates(
    T: np.ndarray, n_macrostates: int = 4, random_state: int | None = 42
) -> np.ndarray | None:
    """Compute metastable sets using deeptime's PCCA+ implementation."""
    if T.size == 0 or T.shape[0] <= n_macrostates:
        return None
    _ = random_state  # Preserved for API stability; no stochastic fall-back is used.
    try:
        model = _deeptime_pcca(np.asarray(T, dtype=float), m=int(n_macrostates))
    except ValueError as exc:
        logger.debug("PCCA+ failed to converge for provided transition matrix: %s", exc)
        return None
    chi = np.asarray(model.memberships, dtype=float)
    labels = np.argmax(chi, axis=1)
    labels = _canonicalize_macro_labels(labels.astype(int), T)
    return cast(np.ndarray, labels)
