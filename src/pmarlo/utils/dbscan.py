from __future__ import annotations

"""Shared helpers for configuring DBSCAN-based clustering."""

from typing import Any, Mapping

import numpy as np
from numpy.random import Generator, default_rng
from sklearn.neighbors import NearestNeighbors

DBSCAN_SUPPORTED_KWARGS = frozenset(
    {
        "eps",
        "min_samples",
        "metric",
        "metric_params",
        "algorithm",
        "leaf_size",
        "p",
        "n_jobs",
    }
)


def _coerce_positive_float(value: Any, *, name: str) -> float:
    try:
        coerced = float(value)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise TypeError(f"{name} must be a float-compatible value") from exc
    if coerced <= 0:
        raise ValueError(f"{name} must be positive; received {coerced!r}")
    return coerced


def _coerce_positive_int(value: Any, *, name: str) -> int:
    try:
        coerced = int(value)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise TypeError(f"{name} must be an integer") from exc
    if coerced <= 0:
        raise ValueError(f"{name} must be a positive integer; received {coerced!r}")
    return coerced


def normalize_dbscan_kwargs(kwargs: Mapping[str, Any] | None) -> dict[str, Any]:
    """Validate and normalise keyword arguments for DBSCAN estimators."""

    if kwargs is None:
        base: dict[str, Any] = {}
    else:
        unsupported = set(kwargs) - DBSCAN_SUPPORTED_KWARGS
        if unsupported:
            raise TypeError(
                "Unsupported DBSCAN parameters: " + ", ".join(sorted(unsupported))
            )
        base = {str(key): value for key, value in kwargs.items()}

    normalized: dict[str, Any] = {}
    if "eps" in base:
        normalized["eps"] = _coerce_positive_float(base["eps"], name="eps")
    if "min_samples" in base:
        normalized["min_samples"] = _coerce_positive_int(
            base["min_samples"], name="min_samples"
        )
    if "leaf_size" in base:
        normalized["leaf_size"] = _coerce_positive_int(
            base["leaf_size"], name="leaf_size"
        )
    if "p" in base and base["p"] is not None:
        normalized["p"] = float(base["p"])
    if "n_jobs" in base and base["n_jobs"] is not None:
        normalized["n_jobs"] = int(base["n_jobs"])
    if "metric" in base:
        normalized["metric"] = str(base["metric"])
    if "metric_params" in base:
        normalized["metric_params"] = base["metric_params"]
    if "algorithm" in base:
        normalized["algorithm"] = str(base["algorithm"])

    normalized.setdefault("eps", 0.5)
    normalized.setdefault("min_samples", 5)

    if "leaf_size" not in normalized and "algorithm" in normalized:
        # When users request ball_tree or kd_tree we keep sklearn default (30).
        normalized["leaf_size"] = 30

    return normalized


def summarise_dbscan_kwargs(kwargs: Mapping[str, Any]) -> str:
    """Produce a short human-readable summary string for logging."""

    eps = kwargs.get("eps")
    min_samples = kwargs.get("min_samples")
    metric = kwargs.get("metric")
    summary_parts = []
    if eps is not None:
        summary_parts.append(f"eps={eps:g}")
    if min_samples is not None:
        summary_parts.append(f"min_samples={min_samples}")
    if metric:
        summary_parts.append(f"metric={metric}")
    if not summary_parts:
        return "dbscan"
    return "dbscan " + " ".join(summary_parts)


def _subsample_for_eps(
    data: np.ndarray, *, max_samples: int, rng: Generator | None
) -> np.ndarray:
    """Return either the full data or a reproducible subsample."""

    if max_samples < 2:
        raise ValueError("max_samples must be at least 2 to estimate eps")
    if data.shape[0] <= max_samples:
        return data
    if rng is None:
        rng = default_rng()
    indices = rng.choice(data.shape[0], size=max_samples, replace=False)
    return data[indices]


def estimate_dbscan_eps(
    data: np.ndarray,
    *,
    min_samples: int,
    quantile: float = 0.9,
    max_samples: int = 10_000,
    random_state: int | None = None,
) -> tuple[float, dict[str, float]]:
    """Estimate an ``eps`` radius based on nearest-neighbour statistics.

    Parameters
    ----------
    data : np.ndarray
        Feature matrix used for clustering (``n_frames`` x ``n_features``).
    min_samples : int
        ``min_samples`` parameter that will be used with DBSCAN. Determines
        which neighbour rank is used to estimate the distance threshold.
    quantile : float, default=0.9
        Percentile of the k-distance distribution to use when deriving ``eps``.
    max_samples : int, default=10_000
        Maximum number of points evaluated when computing nearest neighbours.
        Larger datasets are subsampled uniformly at random for efficiency.
    random_state : int | None, default=None
        Seed controlling the subsampling operation for deterministic behaviour.

    Returns
    -------
    tuple[float, dict[str, float]]
        The estimated ``eps`` radius and metadata describing the sample size,
        neighbour rank, and percentile that were used.
    """

    array = np.asarray(data, dtype=float)
    if array.ndim != 2:
        raise ValueError("DBSCAN EPS estimation expects a 2D feature matrix")
    if array.shape[0] < 2:
        raise ValueError("At least two frames are required to estimate eps")

    if max_samples < 2:
        raise ValueError("max_samples must be at least 2")
    rng = default_rng(random_state) if random_state is not None else None
    sample = _subsample_for_eps(
        array,
        max_samples=int(max_samples),
        rng=rng,
    )

    neighbour_rank = int(max(2, min_samples))
    neighbour_rank = min(neighbour_rank, sample.shape[0])
    if neighbour_rank == sample.shape[0]:
        neighbour_rank = max(2, sample.shape[0] - 1)

    nbrs = NearestNeighbors(n_neighbors=neighbour_rank)
    nbrs.fit(sample)
    distances, _ = nbrs.kneighbors(sample)
    kth_distances = distances[:, -1]

    percentile = float(np.clip(quantile * 100.0, 1.0, 99.0))
    eps = float(np.percentile(kth_distances, percentile))
    if not np.isfinite(eps) or eps <= 0:
        positive = kth_distances[kth_distances > 0]
        if positive.size:
            eps = float(np.mean(positive))
        else:
            eps = 0.5

    metadata = {
        "sample_size": float(sample.shape[0]),
        "neighbor_rank": float(neighbour_rank),
        "percentile": percentile,
    }
    return eps, metadata


__all__ = [
    "DBSCAN_SUPPORTED_KWARGS",
    "normalize_dbscan_kwargs",
    "summarise_dbscan_kwargs",
    "estimate_dbscan_eps",
]
