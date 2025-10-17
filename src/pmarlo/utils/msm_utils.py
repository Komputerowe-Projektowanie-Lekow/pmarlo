from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from deeptime.markov.tools.analysis import (
    is_transition_matrix,
    stationary_distribution as _dt_stationary_distribution,
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

    if np.any(T < 0.0):
        raise ValueError("Negative probabilities in transition matrix")

    if not is_transition_matrix(T, tol=row_tol):
        raise ValueError("transition matrix fails stochasticity checks")

    pi_sum = float(np.sum(pi))
    if not np.isfinite(pi_sum) or pi_sum <= 0:
        raise ValueError("stationary distribution must be normalisable")
    pi_norm = pi / pi_sum

    residual = np.max(np.abs(pi_norm @ T - pi_norm)) if T.size else 0.0
    if residual > stat_tol:
        raise ValueError(
            f"provided stationary distribution fails invariance check (max residual {residual})"
        )

    is_reducible = False
    if T.shape[0] > 1:
        adjacency = csr_matrix((T > const.NUMERIC_MIN_RATE).astype(int))
        n_components, labels = connected_components(
            adjacency, directed=True, connection="strong"
        )
        if n_components > 1:
            recurrent = np.ones(n_components, dtype=bool)
            for state in range(T.shape[0]):
                comp_idx = labels[state]
                if not recurrent[comp_idx]:
                    continue
                mask = labels != comp_idx
                if np.any((T[state] > const.NUMERIC_MIN_RATE) & mask):
                    recurrent[comp_idx] = False
            if np.sum(recurrent) > 1:
                is_reducible = True

    try:
        pi_ref = np.asarray(
            _dt_stationary_distribution(T, check_inputs=False), dtype=float
        )
    except Exception as exc:  # pragma: no cover - should rarely trigger
        raise ValueError("failed to compute stationary distribution") from exc

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
        is_reducible = True
    max_err = float(np.max(diff)) if diff.size else 0.0
    if max_err > stat_tol and not is_reducible:
        idx = int(np.argmax(diff))
        raise ValueError(
            f"Stationary distribution mismatch at state {idx} with error {max_err}"
        )

    row_err = np.abs(T.sum(axis=1) - 1.0)
    min_entry = T.min(axis=1)
    lines = ["state row_err min_T pi_diff"]
    for i in range(T.shape[0]):
        lines.append(f"{i:5d} {row_err[i]:.2e} {min_entry[i]:.2e} {diff[i]:.2e}")
    logger.debug("MSM diagnostics:\n%s", "\n".join(lines))
