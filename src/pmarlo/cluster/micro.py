"""Microstate clustering utilities.

This module provides a thin wrapper around scikit-learn's clustering
algorithms with a small amount of logic to automatically select between
``KMeans`` and ``MiniBatchKMeans`` depending on the size of the dataset.  The
auto-selection helps avoid out-of-memory errors when clustering very large
data sets.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score

logger = logging.getLogger("pmarlo")


@dataclass
class ClusteringResult:
    """Result of microstate clustering."""

    labels: np.ndarray
    n_states: int
    rationale: str | None = None
    centers: np.ndarray | None = None

    @property
    def output_shape(self) -> tuple[int, ...]:
        """Number of microstates identified."""
        return (self.n_states,)


def cluster_microstates(
    Y: np.ndarray,
    method: Literal["auto", "minibatchkmeans", "kmeans"] = "auto",
    n_states: int | Literal["auto"] = "auto",
    random_state: int | None = 42,
    minibatch_threshold: int = 5_000_000,
    **kwargs,
) -> ClusteringResult:
    """Cluster reduced data into microstates and return clustering result.

    Parameters
    ----------
    Y:
        Reduced feature matrix of shape ``(n_frames, n_features)``.
    method:
        Clustering algorithm to use.  When ``"auto"`` (the default) the
        function switches to ``MiniBatchKMeans`` when
        ``n_frames * n_features`` exceeds ``minibatch_threshold``.
    n_states:
        Number of microstates to identify or ``"auto"`` to select the number
        based on the silhouette score.
    random_state:
        Seed for deterministic clustering.  When ``None`` the global NumPy
        random state is used.
    minibatch_threshold:
        Product of frames and features above which ``MiniBatchKMeans`` is used
        when ``method="auto"``.
    **kwargs:
        Additional keyword arguments forwarded to the underlying scikit-learn
        estimator.

    Returns
    -------
    ClusteringResult
        Labels and metadata describing the clustering.
    """

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

    # Determine clustering algorithm
    chosen_method: str
    if method == "auto":
        n_total = int(Y.shape[0] * Y.shape[1])
        if n_total > minibatch_threshold:
            logger.info(
                "Dataset size %d exceeds threshold %d; using MiniBatchKMeans",
                n_total,
                minibatch_threshold,
            )
            chosen_method = "minibatchkmeans"
        else:
            chosen_method = "kmeans"
    else:
        chosen_method = method

    if chosen_method == "minibatchkmeans":
        km = MiniBatchKMeans(n_clusters=n_states, random_state=random_state, **kwargs)
    elif chosen_method == "kmeans":
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
