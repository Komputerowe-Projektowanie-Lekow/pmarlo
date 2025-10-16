"""Microstate clustering utilities for Markov state model construction.

This module provides intelligent clustering of reduced-dimensional feature data
into microstates, which serve as the foundation for Markov state model (MSM)
analysis. The implementation automatically selects between KMeans and MiniBatchKMeans
algorithms based on dataset size to prevent memory issues with large trajectories.

The module supports both manual and automatic determination of the optimal number
of microstates, with the latter using silhouette score optimization.

Examples
--------
>>> import numpy as np
>>> from pmarlo.markov_state_model.clustering import cluster_microstates
>>>
>>> # Create sample feature data
>>> features = np.random.rand(1000, 10)
>>>
>>> # Cluster with fixed number of states
>>> result = cluster_microstates(features, n_states=5, random_state=42)
>>> print(f"Clustered into {result.n_states} microstates")
>>>
>>> # Automatic state selection
>>> result = cluster_microstates(features, n_states="auto", random_state=42)
>>> print(f"Auto-selected {result.n_states} states with score: {result.rationale}")
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
    """Container for microstate clustering results and metadata.

    This dataclass holds the complete output of the clustering process,
    including state assignments, cluster centers, and decision rationale
    when automatic clustering is used.

    Attributes
    ----------
    labels : np.ndarray
        Cluster assignment for each data point. Shape matches the first
        dimension of the input feature matrix.
    n_states : int
        Number of microstates identified. Either the requested number
        or the auto-selected optimal number.
    rationale : str | None, optional
        Explanation of the clustering decision, particularly when
        n_states="auto" was used. Contains silhouette score information.
    centers : np.ndarray | None, optional
        Cluster centers in the feature space. Only available for
        KMeans-based algorithms.

    Examples
    --------
    >>> result = ClusteringResult(
    ...     labels=np.array([0, 1, 0, 1]),
    ...     n_states=2,
    ...     rationale="silhouette=0.85"
    ... )
    >>> print(f"Assigned {len(result.labels)} points to {result.n_states} clusters")
    """

    labels: np.ndarray
    n_states: int
    rationale: str | None = None
    centers: np.ndarray | None = None

    @property
    def output_shape(self) -> tuple[int, ...]:
        """Shape of the clustering output for compatibility with result APIs.

        Returns
        -------
        tuple[int, ...]
            Tuple containing only the number of states for API compatibility.
        """
        return (self.n_states,)


def _validate_clustering_inputs(Y: np.ndarray) -> None:
    """Validate inputs for clustering operations.

    Parameters
    ----------
    Y : np.ndarray
        Input feature matrix to validate.

    Raises
    ------
    ValueError
        If input dimensions are invalid.
    """
    if Y.ndim != 2:
        raise ValueError(f"Input must be 2D array, got shape {Y.shape}")

    if Y.shape[1] == 0:
        raise ValueError("Input array must have at least one feature")


def _select_clustering_method(
    method: Literal["auto", "minibatchkmeans", "kmeans"],
    Y: np.ndarray,
    minibatch_threshold: int,
) -> str:
    """Select the appropriate clustering algorithm based on method and data size.

    Parameters
    ----------
    method : Literal["auto", "minibatchkmeans", "kmeans"]
        Requested clustering method.
    Y : np.ndarray
        Input feature matrix.
    minibatch_threshold : int
        Size threshold for switching to MiniBatchKMeans.

    Returns
    -------
    str
        Selected clustering method ("kmeans" or "minibatchkmeans").

    Raises
    ------
    ValueError
        If method is not supported.
    """
    if method == "auto":
        n_total = int(Y.shape[0] * Y.shape[1])
        if n_total > minibatch_threshold:
            logger.info(
                "Dataset size %d exceeds threshold %d; using MiniBatchKMeans",
                n_total,
                minibatch_threshold,
            )
            return "minibatchkmeans"
        else:
            return "kmeans"
    elif method in ("kmeans", "minibatchkmeans"):
        return method
    else:
        raise ValueError(f"Unsupported clustering method: {method}")


def _auto_select_n_states(Y: np.ndarray, random_state: int | None) -> tuple[int, str]:
    """Automatically select optimal number of states using silhouette score.

    Parameters
    ----------
    Y : np.ndarray
        Input feature matrix.
    random_state : int | None
        Random state for reproducible clustering.

    Returns
    -------
    tuple[int, str]
        Optimal number of states and rationale string with silhouette score.
    """
    candidates = range(4, 21)
    scores: list[tuple[int, float]] = []

    for n in candidates:
        km = KMeans(n_clusters=n, random_state=random_state, n_init=10)
        labels = km.fit_predict(Y)

        if len(set(labels)) <= 1:
            score = -1.0
        else:
            score = float(silhouette_score(Y, labels))

        scores.append((n, score))

    chosen, best_score = max(scores, key=lambda x: x[1])
    rationale = f"silhouette={best_score:.3f}"

    logger.info(
        "Auto-selected %d states with silhouette score %.3f", chosen, best_score
    )

    return chosen, rationale


def _create_clustering_estimator(
    method: str, n_states: int, random_state: int | None, **kwargs
) -> KMeans | MiniBatchKMeans:
    """Create the appropriate clustering estimator.

    Parameters
    ----------
    method : str
        Clustering method ("kmeans" or "minibatchkmeans").
    n_states : int
        Number of clusters.
    random_state : int | None
        Random state for reproducibility.
    **kwargs
        Additional keyword arguments for the estimator.

    Returns
    -------
    KMeans | MiniBatchKMeans
        Configured clustering estimator.
    """
    if method == "minibatchkmeans":
        return MiniBatchKMeans(n_clusters=n_states, random_state=random_state, **kwargs)
    elif method == "kmeans":
        # Ensure n_init is an integer for KMeans
        kwargs.setdefault("n_init", 10)
        if "n_init" in kwargs:
            kwargs["n_init"] = int(kwargs["n_init"])
        return KMeans(n_clusters=n_states, random_state=random_state, **kwargs)
    else:
        raise ValueError(f"Unsupported method: {method}")


def cluster_microstates(
    Y: np.ndarray,
    method: Literal["auto", "minibatchkmeans", "kmeans"] = "auto",
    n_states: int | Literal["auto"] = "auto",
    random_state: int | None = 42,
    minibatch_threshold: int = 5_000_000,
    **kwargs,
) -> ClusteringResult:
    """Cluster reduced feature data into microstates for Markov state model analysis.

    This function provides intelligent clustering of high-dimensional feature data
    into discrete microstates. It supports automatic algorithm selection based on
    dataset size and automatic determination of optimal cluster count using
    silhouette score optimization.

    Parameters
    ----------
    Y : np.ndarray
        Reduced feature matrix of shape ``(n_frames, n_features)``.
        Each row represents a molecular conformation in reduced coordinates.
    method : Literal["auto", "minibatchkmeans", "kmeans"], default="auto"
        Clustering algorithm to use. When ``"auto"`` (the default), the function
        automatically switches to ``MiniBatchKMeans`` when the product of
        ``n_frames * n_features`` exceeds ``minibatch_threshold`` to prevent
        memory issues with large datasets.
    n_states : int | Literal["auto"], default="auto"
        Number of microstates to identify. If ``"auto"``, the optimal number
        is selected by maximizing the silhouette score over candidates from 4 to 20.
        If an integer, that exact number of states is used.
    random_state : int | None, default=42
        Seed for deterministic clustering. When ``None``, the global NumPy
        random state is used. Ensures reproducible results across runs.
    minibatch_threshold : int, default=5_000_000
        Size threshold for automatic method selection. When the product of
        ``n_frames * n_features`` exceeds this value and ``method="auto"``,
        ``MiniBatchKMeans`` is used instead of ``KMeans``.
    **kwargs
        Additional keyword arguments forwarded to the underlying scikit-learn
        clustering estimator (KMeans or MiniBatchKMeans).

    Returns
    -------
    ClusteringResult
        Complete clustering results containing:

        - labels: Cluster assignment for each frame
        - n_states: Number of identified microstates
        - rationale: Decision explanation (when auto-selection is used)
        - centers: Cluster centers in feature space (when available)

    Raises
    ------
    ValueError
        If input validation fails (wrong dimensions, unsupported method, etc.).

    Notes
    -----
    The clustering process involves these steps:

    1. **Input Validation**: Check array dimensions and data validity
    2. **State Selection**: Auto-select optimal k if requested using silhouette scores
    3. **Method Selection**: Choose between KMeans and MiniBatchKMeans based on size
    4. **Clustering**: Execute the selected algorithm
    5. **Result Packaging**: Return structured results with metadata

    The automatic method selection prevents memory issues with large trajectory
    datasets by switching to the more memory-efficient MiniBatchKMeans algorithm.

    Examples
    --------
    >>> import numpy as np
    >>> from pmarlo.markov_state_model.clustering import cluster_microstates
    >>>
    >>> # Create sample feature data (1000 frames, 10 features)
    >>> features = np.random.rand(1000, 10)
    >>>
    >>> # Manual clustering with 5 states
    >>> result = cluster_microstates(features, n_states=5, random_state=42)
    >>> print(f"Clustered into {result.n_states} microstates")
    >>>
    >>> # Automatic state and method selection
    >>> result = cluster_microstates(features, method="auto", n_states="auto")
    >>> print(f"Auto-selected {result.n_states} states: {result.rationale}")
    >>>
    >>> # Large dataset - will automatically use MiniBatchKMeans
    >>> large_features = np.random.rand(10000, 50)
    >>> result = cluster_microstates(large_features, minibatch_threshold=100_000)

    See Also
    --------
    ClusteringResult : Container for clustering results and metadata
    sklearn.cluster.KMeans : Standard K-means clustering
    sklearn.cluster.MiniBatchKMeans : Memory-efficient variant for large datasets
    """
    # Handle edge case of empty dataset
    if Y.shape[0] == 0:
        logger.info("Empty dataset provided, returning empty clustering result")
        return ClusteringResult(labels=np.zeros((0,), dtype=int), n_states=0)

    # Validate input dimensions and data
    _validate_clustering_inputs(Y)

    # Store original request for logging
    requested = n_states
    rationale: str | None = None

    # Auto-select number of states if requested
    if isinstance(n_states, str) and n_states == "auto":
        n_states, rationale = _auto_select_n_states(Y, random_state)
    else:
        n_states = int(n_states)

    # Select appropriate clustering algorithm
    chosen_method = _select_clustering_method(method, Y, minibatch_threshold)

    # Create and configure the clustering estimator
    estimator = _create_clustering_estimator(
        chosen_method, n_states, random_state, **kwargs
    )

    # Execute clustering
    logger.info(
        "Starting clustering with %s algorithm: %d states, %d samples, %d features",
        chosen_method,
        n_states,
        Y.shape[0],
        Y.shape[1],
    )

    labels = cast(np.ndarray, estimator.fit_predict(Y).astype(int))
    centers = getattr(estimator, "cluster_centers_", None)

    unique_labels = int(np.unique(labels).size)
    if unique_labels != n_states:
        message = (
            "Clustering produced {unique} unique microstates, expected {expected}. "
            "This typically indicates degenerate CV data or an incompatible "
            "n_microstates setting."
        ).format(unique=unique_labels, expected=n_states)
        logger.error(message)
        raise ValueError(message)

    if centers is not None and int(centers.shape[0]) != n_states:
        message = (
            "Clustering returned {centers} centers, expected {expected}. "
            "Verify CV spread and clustering configuration."
        ).format(centers=int(centers.shape[0]), expected=n_states)
        logger.error(message)
        raise ValueError(message)

    # Log completion with rationale if available
    logger.info(
        "Clustering completed: requested=%s, actual=%d%s",
        requested,
        n_states,
        f" ({rationale})" if rationale else "",
    )

    return ClusteringResult(
        labels=labels, n_states=n_states, rationale=rationale, centers=centers
    )
