from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np


def build_simple_msm(
    dtrajs: List[np.ndarray],
    n_states: Optional[int] = None,
    lag: int = 20,
    count_mode: str = "sliding",
) -> Tuple[np.ndarray, np.ndarray]:
    """Build MSM using deeptime estimators, with a robust fallback.

    Returns a pair (transition_matrix, stationary_distribution).
    """
    if not dtrajs:
        return np.zeros((0, 0), dtype=float), np.zeros((0,), dtype=float)

    n_states = _infer_n_states(dtrajs, n_states)

    # Try deeptime-based estimation first
    try:
        T, pi = _fit_msm_deeptime(dtrajs, n_states, lag, count_mode)
        return T, pi
    except Exception:
        # Fall back to Dirichlet-regularized ML counts
        return _fit_msm_fallback(dtrajs, n_states, lag)


def _infer_n_states(dtrajs: List[np.ndarray], n_states: Optional[int]) -> int:
    if n_states is not None:
        return int(n_states)
    max_state = 0
    for dt in dtrajs:
        if dt.size:
            max_state = max(max_state, int(np.max(dt)))
    return int(max_state + 1)


def _fit_msm_deeptime(
    dtrajs: List[np.ndarray],
    n_states: int,
    lag: int,
    count_mode: str,
) -> Tuple[np.ndarray, np.ndarray]:
    from deeptime.markov import TransitionCountEstimator  # type: ignore
    from deeptime.markov.msm import MaximumLikelihoodMSM  # type: ignore

    tce = TransitionCountEstimator(
        lagtime=int(max(1, lag)), count_mode=str(count_mode), sparse=False
    )
    count_model = tce.fit(dtrajs).fetch_model()
    ml = MaximumLikelihoodMSM(reversible=True)
    msm = ml.fit(count_model).fetch_model()

    T = np.asarray(msm.transition_matrix, dtype=float)
    pi = _stationary_from_model_or_T(msm, T)
    return T, cast(np.ndarray, pi)


def _stationary_from_model_or_T(msm: object, T: np.ndarray) -> np.ndarray:
    if (
        hasattr(msm, "stationary_distribution")
        and getattr(msm, "stationary_distribution") is not None
    ):
        return np.asarray(getattr(msm, "stationary_distribution"), dtype=float)
    return _stationary_from_T(T)


def _fit_msm_fallback(
    dtrajs: List[np.ndarray], n_states: int, lag: int
) -> Tuple[np.ndarray, np.ndarray]:
    counts: Dict[Tuple[int, int], float] = defaultdict(float)
    alpha = 2.0
    for dtraj in dtrajs:
        if dtraj.size <= lag:
            continue
        for i in range(0, dtraj.size - lag):
            a = int(dtraj[i])
            b = int(dtraj[i + lag])
            counts[(a, b)] += 1.0
    C = np.full((n_states, n_states), alpha, dtype=float)
    for (i, j), c in counts.items():
        C[i, j] += c
    T = _row_normalize(C)
    pi = _stationary_from_T(T)
    return T, pi


def _row_normalize(C: np.ndarray) -> np.ndarray[Any, Any]:
    rows = C.sum(axis=1)
    rows[rows == 0] = 1.0
    return cast(np.ndarray[Any, Any], C / rows[:, None])


def _stationary_from_T(T: np.ndarray) -> np.ndarray:
    evals, evecs = np.linalg.eig(T.T)
    idx = int(np.argmax(np.real(evals)))
    pi = np.real(evecs[:, idx])
    pi = np.abs(pi)
    s = float(np.sum(pi))
    if s > 0:
        pi /= s
    return cast(np.ndarray, pi)


def pcca_like_macrostates(
    T: np.ndarray, n_macrostates: int = 4, random_state: int = 42
) -> Optional[np.ndarray]:
    """Compute metastable sets using PCCA+ (deeptime), fallback to k-means on eigenvectors.
    Returns hard labels per microstate.
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
        return cast(np.ndarray, labels.astype(int))
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
            return cast(np.ndarray, labels.astype(int))
        except Exception:
            return None


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
