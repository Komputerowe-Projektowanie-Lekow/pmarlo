from __future__ import annotations

import logging
from dataclasses import dataclass

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
