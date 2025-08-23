"""Microstate clustering utilities."""

from __future__ import annotations

import logging
from typing import Literal, cast

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans

logger = logging.getLogger("pmarlo")


def cluster_microstates(
    Y: np.ndarray,
    method: Literal["auto", "minibatchkmeans", "kmeans"] = "auto",
    n_clusters: int = 200,
    random_state: int = 42,
    threshold: int = 1_000_000,
    **kwargs: object,
) -> np.ndarray:
    """Cluster reduced data into microstates and return integer labels.

    Parameters
    ----------
    Y:
        2D array of shape (n_frames, n_features).
    method:
        Clustering backend. ``"auto"`` chooses :class:`KMeans` for small data and
        switches to :class:`MiniBatchKMeans` when ``n_frames * n_features`` exceeds
        ``threshold``.
    n_clusters:
        Number of clusters to form.
    random_state:
        Deterministic seed for the clustering algorithm.
    threshold:
        Dataset size above which ``MiniBatchKMeans`` is used when ``method="auto"``.
    **kwargs:
        Additional keyword arguments forwarded to the clustering constructor.
    """
    if Y.shape[0] == 0:
        return np.zeros((0,), dtype=int)
    if Y.shape[1] == 0:
        raise ValueError("Input array must have at least one feature")

    chosen: Literal["minibatchkmeans", "kmeans"]
    if method == "auto":
        total = int(Y.shape[0]) * int(Y.shape[1])
        if total > threshold:
            logger.info(
                "Dataset size %s exceeds threshold %s; using MiniBatchKMeans",
                total,
                threshold,
            )
            chosen = "minibatchkmeans"
        else:
            chosen = "kmeans"
    else:
        chosen = method

    if chosen == "minibatchkmeans":
        km = MiniBatchKMeans(n_clusters=n_clusters, random_state=random_state, **kwargs)
    elif chosen == "kmeans":
        km = KMeans(
            n_clusters=n_clusters, random_state=random_state, n_init=10, **kwargs
        )
    else:  # pragma: no cover - defensive
        raise ValueError(f"Unsupported clustering method: {method}")

    labels = km.fit_predict(Y)
    return cast(np.ndarray, labels.astype(int))
