from __future__ import annotations

from typing import Literal, cast

import logging
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans

logger = logging.getLogger("pmarlo")


def cluster_microstates(
    Y: np.ndarray,
    method: Literal["minibatchkmeans", "kmeans"] = "minibatchkmeans",
    n_clusters: int = 200,
    random_state: int | None = 42,
    **kwargs,
) -> np.ndarray:
    """Cluster reduced data into microstates and return labels.

    Args:
        Y: Feature array of shape ``(n_samples, n_features)``.
        method: Clustering algorithm to use.
        n_clusters: Number of clusters to form.
        random_state: Seed for the scikit-learn estimator. ``None`` uses
            the library's global RNG state.
    """
    if Y.shape[0] == 0:
        return np.zeros((0,), dtype=int)
    if Y.shape[1] == 0:
        raise ValueError("Input array must have at least one feature")
    if method == "minibatchkmeans":
        km = MiniBatchKMeans(n_clusters=n_clusters, random_state=random_state, **kwargs)
    elif method == "kmeans":
        km = KMeans(
            n_clusters=n_clusters, random_state=random_state, n_init=10, **kwargs
        )
    else:
        raise ValueError(f"Unsupported clustering method: {method}")
    labels = km.fit_predict(Y)
    return cast(np.ndarray, labels.astype(int))
