from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score

import logging

logger = logging.getLogger("pmarlo")


@dataclass
class ClusteringResult:
    """Result of microstate clustering."""

    labels: np.ndarray
    n_states: int
    rationale: str | None = None
    centers: np.ndarray | None = None


def cluster_microstates(
    Y: np.ndarray,
    method: Literal["minibatchkmeans", "kmeans"] = "kmeans",
    n_states: int | Literal["auto"] = "auto",
    random_state: int = 42,
    **kwargs,
) -> ClusteringResult:
    """Cluster reduced data into microstates and return clustering result."""

    if Y.shape[0] == 0:
        return ClusteringResult(labels=np.zeros((0,), dtype=int), n_states=0)
    if Y.shape[1] == 0:
        raise ValueError("Input array must have at least one feature")

    requested = n_states
    rationale: str | None = None

    if isinstance(n_states, str) and n_states == "auto":
        candidates = range(4, 21)
        scores: list[tuple[int, float]] = []
        for n in candidates:
            km = KMeans(n_clusters=n, random_state=random_state, n_init=10)
            labels = km.fit_predict(Y)
            if len(set(labels)) <= 1:
                score = -1.0
            else:
                score = silhouette_score(Y, labels)
            scores.append((n, float(score)))
        chosen, best = max(scores, key=lambda x: x[1])
        rationale = f"silhouette={best:.3f}"
        n_states = chosen
    else:
        n_states = int(n_states)

    if method == "minibatchkmeans":
        km = MiniBatchKMeans(n_clusters=n_states, random_state=random_state, **kwargs)
    elif method == "kmeans":
        km = KMeans(n_clusters=n_states, random_state=random_state, n_init=10, **kwargs)
    else:
        raise ValueError(f"Unsupported clustering method: {method}")

    labels = cast(np.ndarray, km.fit_predict(Y).astype(int))
    centers = getattr(km, "cluster_centers_", None)

    logger.info(
        "Clustering: requested=%s, actual=%d%s",
        requested,
        n_states,
        f" ({rationale})" if rationale else "",
    )
    return ClusteringResult(
        labels=labels, n_states=n_states, rationale=rationale, centers=centers
    )
