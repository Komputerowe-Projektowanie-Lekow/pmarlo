from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

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
        50,
        75,
        100,
        150,
        200,
        300,
        500,
        750,
        1000,
        1500,
        2000,
    ]

    filtered: list[int] = [x for x in base if lo <= x <= hi]
    if not filtered:
        logger.warning("No predefined lags in range [%s, %s]", lo, hi)
        return [lo] if lo == hi else [lo, hi]

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
    C: np.ndarray, alpha: float = 1e-3, epsilon: float = 1e-12
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
        return ConnectedCountResult(np.zeros((0, 0), dtype=float), active)

    C_active = C[np.ix_(active, active)].astype(float)
    C_active += float(alpha)
    return ConnectedCountResult(C_active, active)


def check_transition_matrix(
    T: np.ndarray,
    pi: np.ndarray,
    *,
    row_tol: float = 1e-12,
    stat_tol: float = 1e-8,
) -> None:
    """Validate a transition matrix and stationary distribution.

    The following conditions are enforced:

    * Each row of ``T`` sums to 1 within ``row_tol``.
    * All elements of ``T`` are non-negative.
    * The provided ``pi`` is a left eigenvector of ``T`` with unit eigenvalue
      up to ``stat_tol`` in the infinity norm.

    Parameters
    ----------
    T:
        Transition matrix.
    pi:
        Stationary distribution corresponding to ``T``.
    row_tol:
        Permitted deviation from exact row stochasticity.
    stat_tol:
        Permitted deviation of ``pi`` from the left eigenvector equation.

    Raises
    ------
    ValueError
        If any of the checks fail. The error message includes the offending
        state indices to ease debugging.
    """

    if T.ndim != 2 or T.shape[0] != T.shape[1]:
        raise ValueError("transition matrix must be square")
    if pi.shape != (T.shape[0],):
        raise ValueError("stationary distribution size mismatch")
    if T.size == 0:
        return

    rowsum = T.sum(axis=1)
    row_err = np.abs(rowsum - 1.0)
    neg_idx = np.where(T < 0)
    if neg_idx[0].size:
        pairs = list(zip(neg_idx[0].tolist(), neg_idx[1].tolist()))
        vals = T[neg_idx].tolist()
        raise ValueError(f"Negative probabilities at {pairs}: {vals}")

    bad_rows = np.where(row_err > row_tol)[0]
    if bad_rows.size:
        devs = row_err[bad_rows].tolist()
        raise ValueError(f"Non-stochastic rows at indices {bad_rows.tolist()}: {devs}")

    pi_res = np.abs(pi @ T - pi)
    max_err = float(np.max(pi_res)) if pi_res.size else 0.0
    if max_err > stat_tol:
        idx = int(np.argmax(pi_res))
        raise ValueError(
            f"Stationary distribution mismatch at state {idx} with error {max_err}"
        )

    min_entry = T.min(axis=1)
    lines = ["state row_err min_T pi_res"]
    for i in range(T.shape[0]):
        lines.append(f"{i:5d} {row_err[i]:.2e} {min_entry[i]:.2e} {pi_res[i]:.2e}")
    logger.debug("MSM diagnostics:\n%s", "\n".join(lines))


def _row_normalize(C: np.ndarray) -> np.ndarray[Any, Any]:
    """Row-normalize a matrix."""
    from typing import cast

    rows = C.sum(axis=1)
    rows[rows == 0] = 1.0
    return cast(np.ndarray[Any, Any], C / rows[:, None])


def _stationary_from_T(T: np.ndarray) -> np.ndarray:
    """Compute stationary distribution from transition matrix."""
    from typing import cast

    evals, evecs = np.linalg.eig(T.T)
    idx = int(np.argmax(np.real(evals)))
    pi = np.real(evecs[:, idx])
    pi = np.abs(pi)
    s = float(np.sum(pi))
    if s > 0:
        pi /= s
    return cast(np.ndarray, pi)


def pcca_like_macrostates(
    T: np.ndarray, n_macrostates: int = 4, random_state: int | None = 42
) -> np.ndarray | None:
    """Compute metastable sets using PCCA+ with a k-means fallback.

    Parameters
    ----------
    T:
        Microstate transition matrix.
    n_macrostates:
        Desired number of macrostates.
    random_state:
        Seed for the k-means fallback. ``None`` uses NumPy's global state.

    Returns
    -------
    Optional[np.ndarray]
        Hard labels per microstate or ``None`` if the decomposition failed.
    """
    if T.size == 0 or T.shape[0] <= n_macrostates:
        return None
    # Try deeptime PCCA+ on transition matrix
    try:
        from deeptime.markov import pcca as _pcca  # type: ignore

        model = _pcca(np.asarray(T, dtype=float), n_metastable_sets=int(n_macrostates))
        # Hard assignments from membership matrix
        chi = np.asarray(model.memberships, dtype=float)
        labels = np.argmax(chi, axis=1)
        labels = _canonicalize_macro_labels(labels.astype(int), T)
        return labels
    except Exception:
        # Fallback: spectral embedding + k-means
        eigvals, eigvecs = np.linalg.eig(T.T)
        order = np.argsort(-np.real(eigvals))
        k = max(2, min(n_macrostates, T.shape[0] - 1))
        comps = np.real(eigvecs[:, order[1 : 1 + k]])
        try:
            from sklearn.cluster import MiniBatchKMeans

            km = MiniBatchKMeans(n_clusters=n_macrostates, random_state=random_state)
            labels = km.fit_predict(comps)
            labels = _canonicalize_macro_labels(labels.astype(int), T)
            return labels
        except Exception:
            return None


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
    from typing import cast

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

    Returns a pair (transition_matrix, stationary_distribution).
    """
    if not dtrajs:
        logger.error("build_simple_msm: No dtrajs provided")
        return np.zeros((0, 0), dtype=float), np.zeros((0,), dtype=float)

    n_states = _infer_n_states(dtrajs, n_states)
    logger.info(f"build_simple_msm: Using {n_states} states")

    # Deeptime-based estimation with fallback
    try:
        T, pi = _fit_msm_deeptime(dtrajs, n_states, lag, count_mode)
    except Exception as exc:  # pragma: no cover - triggered without deeptime
        logger.warning("Falling back to internal MSM estimator due to error: %s", exc)
        T, pi = _fit_msm_fallback(dtrajs, n_states, lag, count_mode)
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
    """
    This function is used to fit the MSM using the deeptime library.
    It uses the TransitionCountEstimator to estimate the transition matrix,
    and the MaximumLikelihoodMSM to fit the MSM.
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
        return _expand_results(n_states, res.active, np.zeros((0, 0)), np.zeros((0,)))
    T_active = _row_normalize(res.counts)
    pi_active = _stationary_from_T(T_active)
    return _expand_results(n_states, res.active, T_active, pi_active)


def _fit_msm_fallback(
    dtrajs: list[np.ndarray], n_states: int, lag: int, count_mode: str
) -> tuple[np.ndarray, np.ndarray]:
    """
    This function is used to fit the MSM using the fallback method.
    It uses the Dirichlet-regularized ML counts to estimate the transition matrix,
    and the stationary distribution is computed from the transition matrix.
    """
    counts = np.zeros((n_states, n_states), dtype=float)
    step = lag if count_mode == "strided" else 1
    for dtraj in dtrajs:
        if dtraj.size <= lag:
            continue
        for i in range(0, dtraj.size - lag, step):
            a = int(dtraj[i])
            b = int(dtraj[i + lag])
            if a < 0 or b < 0 or a >= n_states or b >= n_states:
                continue
            counts[a, b] += 1.0
    res = ensure_connected_counts(counts)
    if res.counts.size == 0:
        return _expand_results(n_states, res.active, np.zeros((0, 0)), np.zeros((0,)))
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
