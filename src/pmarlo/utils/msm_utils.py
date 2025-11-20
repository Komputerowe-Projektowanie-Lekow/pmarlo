from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from deeptime.markov import pcca as _deeptime_pcca  # type: ignore
from deeptime.markov.msm import MaximumLikelihoodMSM  # type: ignore
from deeptime.markov.tools.analysis import is_transition_matrix
from deeptime.markov.tools.analysis import (  # type: ignore
    stationary_distribution as _dt_stationary_distribution,
)
from deeptime.markov.tools.estimation.dense.transition_matrix import (  # type: ignore
    transition_matrix_non_reversible as _dt_row_normalize,
)
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from pmarlo import constants as const

logger = logging.getLogger("pmarlo")


def candidate_lag_ladder(
    min_lag: int = 1,
    max_lag: int = 200,
    n_candidates: int | None = None,
) -> list[int]:
    """Generate a robust set of candidate lag times for MSM ITS analysis.

    Behavior:
    - Uses a curated set of "nice" lags (1, 2, 3, 5, 8 and 10× multiples)
      commonly used for implied-timescale scans.
    - Filters to the inclusive range [min_lag, max_lag].
    - Optionally downsamples to ``n_candidates`` approximately evenly across
      the filtered list while keeping endpoints.

    Args:
        min_lag: Minimum lag value (inclusive), coerced to >= 1.
        max_lag: Maximum lag value (inclusive), coerced to >= min_lag.
        n_candidates: If provided and > 0, downsample to this many points.

    Returns:
        An increasing list of integer lag times.
    """
    lo = int(min_lag)
    hi = int(max_lag)
    if lo < 1:
        raise ValueError("min_lag must be >= 1")
    if hi < lo:
        raise ValueError("max_lag must be >= min_lag")
    if n_candidates is not None and n_candidates < 1:
        raise ValueError("n_candidates must be positive")

    # Curated ladder spanning typical analysis ranges
    # Extended to include powers of 2 (40, 80, 160, 320, 640, 1280) for comprehensive ITS analysis
    base: list[int] = [
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
        300,
        320,
        500,
        640,
        750,
        1000,
        1280,
        1500,
        2000,
    ]

    filtered: list[int] = [x for x in base if lo <= x <= hi]
    if not filtered:
        raise ValueError(f"No predefined lag values available in range [{lo}, {hi}]")

    if n_candidates is None or n_candidates >= len(filtered):
        return filtered

    logger.debug(
        "Downsampling %d lag values to %d candidates", len(filtered), n_candidates
    )

    # Downsample approximately evenly over the filtered ladder, keep endpoints
    if n_candidates == 1:
        return [filtered[0]]
    if n_candidates == 2:
        return [filtered[0], filtered[-1]]

    step = (len(filtered) - 1) / (n_candidates - 1)
    picks = sorted({int(round(i * step)) for i in range(n_candidates)})
    # Ensure endpoints are present
    picks[0] = 0
    picks[-1] = len(filtered) - 1
    return [filtered[i] for i in picks]


@dataclass(slots=True)
class ConnectedCountResult:
    """Result of :func:`ensure_connected_counts`.

    Attributes
    ----------
    counts:
        The trimmed count matrix with pseudocounts added.
    active:
        Indices of states that remained after removing disconnected rows
        and columns.
    """

    counts: np.ndarray
    active: np.ndarray

    def to_dict(self) -> dict[str, list[list[float]] | list[int]]:
        """Return a JSON serialisable representation."""
        return {"counts": self.counts.tolist(), "active": self.active.tolist()}


def ensure_connected_counts(
    C: np.ndarray,
    alpha: float = const.NUMERIC_DIRICHLET_ALPHA,
    epsilon: float = const.NUMERIC_MIN_POSITIVE,
) -> ConnectedCountResult:
    """Regularise and trim a transition count matrix.

    A small Dirichlet pseudocount ``alpha`` is added to every element of the
    matrix. States whose corresponding row *and* column sums are below
    ``epsilon`` are removed, returning the active submatrix and the indices of
    the retained states.

    Parameters
    ----------
    C:
        Square matrix of observed transition counts.
    alpha:
        Pseudocount added to each cell to avoid zeros.
    epsilon:
        Threshold below which a state is considered disconnected.

    Returns
    -------
    ConnectedCountResult
        Dataclass containing the trimmed count matrix and the mapping of
        active state indices.
    """

    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("count matrix must be square")

    totals = C.sum(axis=1) + C.sum(axis=0)
    active = np.where(totals > epsilon)[0]
    if active.size == 0:
        return ConnectedCountResult(np.empty((0, 0), dtype=float), active)

    C_active = C[np.ix_(active, active)].astype(float)
    C_active += float(alpha)
    return ConnectedCountResult(C_active, active)


def _coerce_transition_inputs(
    T: np.ndarray,
    pi: np.ndarray,
    row_tol: float,
) -> tuple[np.ndarray, np.ndarray]:
    if np.any(T < 0.0):
        raise ValueError("Negative probabilities in transition matrix")
    if not is_transition_matrix(T, tol=row_tol):
        raise ValueError("transition matrix fails stochasticity checks")
    pi_sum = float(np.sum(pi))
    if not np.isfinite(pi_sum) or pi_sum <= 0:
        raise ValueError("stationary distribution must be normalisable")
    return T, pi / pi_sum


def _check_invariance(
    pi_norm: np.ndarray,
    T: np.ndarray,
    stat_tol: float,
) -> None:
    residual = float(np.max(np.abs(pi_norm @ T - pi_norm)))
    if residual > stat_tol:
        raise ValueError(
            f"provided stationary distribution fails invariance check (max residual {residual})"
        )


def _detect_reducibility(T: np.ndarray) -> bool:
    if T.shape[0] <= 1:
        return False
    adjacency = csr_matrix((T > const.NUMERIC_MIN_RATE).astype(int))
    n_components, labels = connected_components(
        adjacency, directed=True, connection="strong"
    )
    if n_components <= 1:
        return False
    recurrent = np.ones(n_components, dtype=bool)
    for state in range(T.shape[0]):
        comp_idx = labels[state]
        if not recurrent[comp_idx]:
            continue
        mask = labels != comp_idx
        if np.any((T[state] > const.NUMERIC_MIN_RATE) & mask):
            recurrent[comp_idx] = False
    return bool(np.sum(recurrent) > 1)


def _compute_reference_stationary(T: np.ndarray) -> np.ndarray:
    try:
        return np.asarray(
            _dt_stationary_distribution(T, check_inputs=False), dtype=float
        )
    except Exception as exc:  # pragma: no cover - should rarely trigger
        raise ValueError("failed to compute stationary distribution") from exc


def _evaluate_stationary_difference(
    pi_norm: np.ndarray,
    pi_ref: np.ndarray,
    T: np.ndarray,
    stat_tol: float,
    reducible: bool,
) -> tuple[np.ndarray, bool]:
    if pi_ref.shape != pi_norm.shape:
        raise ValueError("stationary distribution size mismatch")

    diff = np.abs(pi_norm - pi_ref)
    ignore_states = np.zeros(diff.shape, dtype=bool)
    support_mask = pi_norm > stat_tol
    if support_mask.any():
        for idx_state in range(T.shape[0]):
            if pi_norm[idx_state] > stat_tol:
                continue
            incoming = T[support_mask, idx_state]
            if np.any(incoming > const.NUMERIC_MIN_RATE):
                continue
            ignore_states[idx_state] = True
    else:
        ignore_states[:] = True

    if ignore_states.any():
        diff[ignore_states] = 0.0
        reducible = True

    max_err = float(np.max(diff)) if diff.size else 0.0
    if max_err > stat_tol and not reducible:
        idx = int(np.argmax(diff))
        raise ValueError(
            f"Stationary distribution mismatch at state {idx} with error {max_err}"
        )
    return diff, reducible


def _log_transition_diagnostics(T: np.ndarray, diff: np.ndarray) -> None:
    row_err = np.abs(T.sum(axis=1) - 1.0)
    min_entry = T.min(axis=1)
    lines = ["state row_err min_T pi_diff"]
    for i in range(T.shape[0]):
        lines.append(f"{i:5d} {row_err[i]:.2e} {min_entry[i]:.2e} {diff[i]:.2e}")
    logger.debug("MSM diagnostics:\n%s", "\n".join(lines))


def check_transition_matrix(
    T: np.ndarray,
    pi: np.ndarray,
    *,
    row_tol: float = const.NUMERIC_MIN_POSITIVE,
    stat_tol: float = const.NUMERIC_RELATIVE_TOLERANCE,
) -> None:
    """Validate a transition matrix and stationary distribution."""

    if T.ndim != 2 or T.shape[0] != T.shape[1]:
        raise ValueError("transition matrix must be square")
    if pi.shape != (T.shape[0],):
        raise ValueError("stationary distribution size mismatch")
    if T.size == 0:
        return

    T = np.asarray(T, dtype=float)
    pi = np.asarray(pi, dtype=float)

    T, pi_norm = _coerce_transition_inputs(T, pi, row_tol)
    _check_invariance(pi_norm, T, stat_tol)

    is_reducible = _detect_reducibility(T)
    pi_ref = _compute_reference_stationary(T)
    diff, is_reducible = _evaluate_stationary_difference(
        pi_norm, pi_ref, T, stat_tol, is_reducible
    )
    _log_transition_diagnostics(T, diff)


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


def _infer_n_states(dtrajs: list[np.ndarray], n_states: int | None) -> int:
    """Infer number of microstates from provided labels when not specified."""
    if n_states is not None:
        logger.debug("infer_n_states: States provided, using %d", n_states)
        return int(n_states)
    max_state = -1
    for dt in dtrajs:
        if dt.size:
            m = int(np.max(dt))
            if m >= 0:
                max_state = max(max_state, m)
    n = int(max_state + 1) if max_state >= 0 else 0
    logger.debug("infer_n_states: Using %d states", n)
    return n


def _expand_results(
    n_states: int, active: np.ndarray, T_active: np.ndarray, pi_active: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Expand MSM results back to the original state space."""

    # BUGFIX: the estimator may return active indices beyond ``n_states`` when
    # the user under-specifies ``n_states``. Ensure the expanded matrices are
    # large enough to embed the active subset instead of raising IndexError.
    required_states = int(np.max(active)) + 1 if active.size else 0
    full_size = max(int(n_states), required_states)

    T_full = np.eye(full_size, dtype=float)
    pi_full = np.zeros((full_size,), dtype=float)
    if active.size:
        T_full[np.ix_(active, active)] = T_active
        pi_full[active] = pi_active
    return T_full, pi_full


def _fit_msm_deeptime(
    dtrajs: list[np.ndarray],
    n_states: int,
    lag: int,
    count_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit MSM using deeptime library (required dependency).

    Uses TransitionCountEstimator to estimate transition counts and
    MaximumLikelihoodMSM with reversible=True to obtain a reversible
    transition matrix and stationary distribution.
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
    ml = MaximumLikelihoodMSM(
        lagtime=int(max(1, lag)),
        reversible=True,
    )
    msm_model = ml.fit(res.counts).fetch_model()
    T_active = np.asarray(msm_model.transition_matrix, dtype=float)
    pi_active = np.asarray(msm_model.stationary_distribution, dtype=float)
    return _expand_results(n_states, res.active, T_active, pi_active)


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
    logger.info("build_simple_msm: Using %d states", n_states)

    T, pi = _fit_msm_deeptime(dtrajs, n_states, lag, count_mode)
    logger.info("build_simple_msm: Transition matrix shape: %s", T.shape)
    logger.info("build_simple_msm: Stationary distribution shape: %s", pi.shape)
    check_transition_matrix(T, pi)
    return T, pi


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


def select_lag_from_its(
    lag_times: np.ndarray,
    timescales: np.ndarray,
    *,
    min_lag_idx: int = 3,
    plateau_threshold: float = 0.15,
) -> int:
    """Select optimal lag time from implied timescales by detecting plateau."""
    if lag_times.size == 0 or timescales.size == 0:
        logger.warning("select_lag_from_its: Empty inputs, returning default lag=10")
        return 10

    ts_slow = timescales[:, 0] if timescales.ndim > 1 else timescales

    valid_mask = np.isfinite(ts_slow) & (ts_slow > 0)
    if not np.any(valid_mask):
        logger.warning(
            "select_lag_from_its: No valid timescales, returning default lag=10"
        )
        return 10

    start_idx = max(1, min_lag_idx)
    if start_idx >= len(ts_slow):
        start_idx = max(1, len(ts_slow) // 4)

    for idx in range(start_idx, len(ts_slow)):
        if not valid_mask[idx] or not valid_mask[idx - 1]:
            continue

        rel_change = abs((ts_slow[idx] - ts_slow[idx - 1]) / ts_slow[idx - 1])

        if rel_change < plateau_threshold:
            if idx + 1 < len(ts_slow) and valid_mask[idx + 1]:
                rel_change_next = abs((ts_slow[idx + 1] - ts_slow[idx]) / ts_slow[idx])
                if rel_change_next < plateau_threshold * 1.5:
                    selected_lag = int(lag_times[idx])
                    logger.info(
                        "select_lag_from_its: Plateau detected at lag=%d (rel_change=%.3f)",
                        selected_lag,
                        rel_change,
                    )
                    return selected_lag
            else:
                selected_lag = int(lag_times[idx])
                logger.info(
                    "select_lag_from_its: Plateau detected at lag=%d (rel_change=%.3f)",
                    selected_lag,
                    rel_change,
                )
                return selected_lag

    half_idx = len(ts_slow) // 2
    valid_latter = valid_mask[half_idx:]
    if np.any(valid_latter):
        latter_ts = ts_slow[half_idx:]
        max_idx_relative = np.nanargmax(np.where(valid_latter, latter_ts, -np.inf))
        selected_lag = int(lag_times[half_idx + max_idx_relative])
        logger.info(
            "select_lag_from_its: No clear plateau, selecting lag=%d with max timescale",
            selected_lag,
        )
        return selected_lag

    median_idx = len(lag_times) // 2
    selected_lag = int(lag_times[median_idx])
    logger.warning(
        "select_lag_from_its: No plateau detected, using median lag=%d", selected_lag
    )
    return selected_lag
