from __future__ import annotations

from typing import Literal, cast

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans


def cluster_microstates(
    Y: np.ndarray,
    method: Literal["minibatchkmeans", "kmeans"] = "minibatchkmeans",
    n_clusters: int = 200,
    random_state: int = 42,
    **kwargs,
) -> np.ndarray:
    """Cluster reduced data into microstates and return labels."""
    if Y.size == 0:
        return np.zeros((0,), dtype=int)
    if method == "minibatchkmeans":
        km = MiniBatchKMeans(n_clusters=n_clusters, random_state=random_state, **kwargs)
    elif method == "kmeans":
        km = KMeans(
            n_clusters=n_clusters, random_state=random_state, n_init="auto", **kwargs
        )
    else:
        raise ValueError(f"Unsupported clustering method: {method}")
    labels = km.fit_predict(Y)
    return cast(np.ndarray, labels.astype(int))
