"""Microstate clustering utilities for Markov state model construction.

This module provides intelligent clustering of reduced-dimensional feature data
into microstates, which serve as the foundation for Markov state model (MSM)
analysis. The implementation relies exclusively on scikit-learn's well-tested
clustering estimators to keep behaviour consistent across the project.

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
from typing import Any, Dict, Literal, Mapping

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score

try:  # pragma: no cover - optional dependency
    from sklearn.cluster import DBSCAN  # type: ignore[import]
except Exception:  # pragma: no cover - fallback path
    DBSCAN = None

from pmarlo.utils.dbscan import (
    estimate_dbscan_eps,
    fit_predict_dbscan,
    normalize_dbscan_kwargs,
    summarise_dbscan_kwargs,
)

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
        Cluster representatives in feature space. Available for algorithms that
        produce centroids (e.g. KMeans) or when derived from DBSCAN clusters.

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
) -> Literal["minibatchkmeans", "kmeans"]:
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
        return "kmeans"
    if method in ("kmeans", "minibatchkmeans"):
        return method
    else:
        raise ValueError(f"Unsupported clustering method: {method}")


def _auto_select_n_states(
    Y: np.ndarray,
    random_state: int | None,
    *,
    sample_size: int | None = None,
    override_n_states: int | None = None,
    estimator_kwargs: Mapping[str, Any] | None = None,
) -> tuple[int, str]:
    """Automatically select optimal number of states using silhouette score.

    Parameters
    ----------
    Y : np.ndarray
        Input feature matrix.
    random_state : int | None
        Random state for reproducible clustering.
    sample_size : int | None, optional
        Size of the random subset to use when computing silhouette scores.
    override_n_states : int | None, optional
        If provided, skip silhouette scoring and return this value directly.

    Returns
    -------
    tuple[int, str]
        Optimal number of states and rationale string with silhouette score.
    """
    if override_n_states is not None:
        if override_n_states <= 0:
            raise ValueError(
                "override_n_states must be a positive integer; "
                f"received {override_n_states}."
            )
        rationale = f"auto-override={override_n_states}"
        logger.info(
            "Auto-selection overridden; using %d states without silhouette scoring",
            override_n_states,
        )
        return override_n_states, rationale

    if sample_size is not None:
        if sample_size <= 1:
            raise ValueError(
                "sample_size must be greater than 1 when sampling for silhouette "
                "scoring."
            )
        effective_sample = min(int(sample_size), int(Y.shape[0]))
        if effective_sample < int(sample_size):
            logger.debug(
                "Requested sample_size %d exceeds dataset rows %d; using %d samples instead.",
                sample_size,
                Y.shape[0],
                effective_sample,
            )
        rng = np.random.default_rng(random_state)
        indices = rng.choice(Y.shape[0], size=effective_sample, replace=False)
        Y_sample = Y[indices]
        sample_note = f" sample={effective_sample}"
    else:
        Y_sample = Y
        sample_note = ""

    candidates = range(4, 21)
    scores: list[tuple[int, float]] = []

    estimator_kwargs = dict(estimator_kwargs or {})

    for n in candidates:
        km = _create_clustering_estimator("kmeans", n, random_state, **estimator_kwargs)
        km.fit(Y_sample)
        if hasattr(km, "labels_"):
            labels = np.asarray(km.labels_, dtype=int)
        else:  # pragma: no cover - defensive fallback
            labels = np.asarray(km.predict(Y_sample), dtype=int)

        if len(set(labels)) <= 1:
            score = -1.0
        else:
            score = float(silhouette_score(Y_sample, labels))

        scores.append((n, score))

    chosen, best_score = max(scores, key=lambda x: x[1])
    rationale = f"silhouette={best_score:.3f}{sample_note}"

    logger.info(
        "Auto-selected %d states with silhouette score %.3f", chosen, best_score
    )

    return chosen, rationale


_KMEANS_ALLOWED_KWARGS: frozenset[str] = frozenset(
    {"max_iter", "tol", "init", "n_init", "verbose", "copy_x", "algorithm"}
)
_MINIBATCH_ONLY_KWARGS: frozenset[str] = frozenset(
    {
        "batch_size",
        "compute_labels",
        "max_no_improvement",
        "init_size",
        "reassignment_ratio",
    }
)
_MINIBATCH_ALLOWED_KWARGS: frozenset[str] = (
    frozenset({"max_iter", "tol", "init", "n_init", "verbose"}) | _MINIBATCH_ONLY_KWARGS
)
_LEGACY_KWARGS: Dict[str, str] = {
    "tolerance": "tol",
    "init_strategy": "init",
    "initial_centers": "init",
}
_UNSUPPORTED_KWARGS: frozenset[str] = frozenset(
    {"metric", "n_jobs", "progress", "fixed_seed"}
)


def _normalize_kmeans_kwargs(
    method: Literal["minibatchkmeans", "kmeans"], kwargs: Mapping[str, Any]
) -> Dict[str, Any]:
    """Validate and normalise kwargs for scikit-learn KMeans estimators."""

    allowed = (
        _KMEANS_ALLOWED_KWARGS if method == "kmeans" else _MINIBATCH_ALLOWED_KWARGS
    )
    normalized: Dict[str, Any] = {}
    for raw_key, value in kwargs.items():
        key = str(raw_key)
        canonical = _LEGACY_KWARGS.get(key, key)
        if canonical in _UNSUPPORTED_KWARGS:
            raise TypeError(
                f"Parameter '{key}' is not supported by the scikit-learn clustering backend."
            )
        if canonical not in allowed:
            raise TypeError(
                "Unsupported clustering parameters for scikit-learn backend: "
                f"{key!r}"
            )
        normalized[canonical] = value
    return normalized


def _prepare_silhouette_kwargs(kwargs: Mapping[str, Any]) -> Dict[str, Any]:
    """Drop MiniBatch-only kwargs before sampling silhouette candidates."""

    filtered = {
        key: value
        for key, value in kwargs.items()
        if _LEGACY_KWARGS.get(key, key) not in _MINIBATCH_ONLY_KWARGS
    }
    return _normalize_kmeans_kwargs("kmeans", filtered)


def _create_clustering_estimator(
    method: Literal["minibatchkmeans", "kmeans"],
    n_states: int,
    random_state: int | None,
    **kwargs: Any,
) -> KMeans | MiniBatchKMeans:
    """Create the appropriate scikit-learn clustering estimator."""

    estimator_cls = MiniBatchKMeans if method == "minibatchkmeans" else KMeans
    estimator_kwargs = dict(kwargs)
    estimator_kwargs.setdefault("n_init", 1)
    if (
        "n_init" in estimator_kwargs
        and isinstance(estimator_kwargs["n_init"], str)
        and estimator_kwargs["n_init"].lower() == "auto"
    ):
        estimator_kwargs["n_init"] = "auto"
    elif "n_init" in estimator_kwargs:
        try:
            estimator_kwargs["n_init"] = int(estimator_kwargs["n_init"])
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "n_init must be provided as an integer or 'auto' for scikit-learn estimators"
            ) from exc
        if estimator_kwargs["n_init"] <= 0:
            raise ValueError(
                "n_init must be a positive integer when clustering microstates"
            )

    return estimator_cls(
        n_clusters=n_states,
        random_state=None if random_state is None else int(random_state),
        **estimator_kwargs,
    )


def _remap_labels_with_noise(
    Y: np.ndarray, raw_labels: np.ndarray, *, noise_label: int = -1
) -> tuple[np.ndarray, np.ndarray | None, int]:
    """Remap DBSCAN labels while preserving noise assignments."""

    unique_labels = [
        int(label) for label in np.unique(raw_labels) if int(label) != noise_label
    ]
    if not unique_labels:
        raise ValueError(
            "DBSCAN did not identify any clusters; consider relaxing 'eps' or "
            "'min_samples'."
        )

    mapping = {original: idx for idx, original in enumerate(unique_labels)}
    remapped = np.full(raw_labels.shape, noise_label, dtype=int)
    for idx, original in enumerate(raw_labels):
        if int(original) == noise_label:
            continue
        remapped[idx] = mapping[int(original)]

    centers = np.zeros((len(unique_labels), Y.shape[1]), dtype=float)
    for original, dense_label in mapping.items():
        mask = raw_labels == original
        centers[dense_label] = np.asarray(Y[mask], dtype=float).mean(axis=0)

    return remapped, centers if centers.size else None, len(unique_labels)


def _cluster_with_dbscan(
    Y: np.ndarray,
    *,
    random_state: int | None,
    kwargs: Mapping[str, Any],
) -> ClusteringResult:
    """Execute DBSCAN clustering and package the result."""

    if random_state is not None:
        logger.info(
            "DBSCAN is deterministic; ignoring requested random_state=%s", random_state
        )

    provided_kwargs = dict(kwargs or {})
    estimator_kwargs = normalize_dbscan_kwargs(provided_kwargs)
    auto_eps_info: dict[str, float] | None = None
    if "eps" not in provided_kwargs:
        auto_eps, meta = estimate_dbscan_eps(
            Y,
            min_samples=int(estimator_kwargs["min_samples"]),
            random_state=random_state,
        )
        estimator_kwargs["eps"] = auto_eps
        auto_eps_info = meta
        logger.info(
            "Auto-selected DBSCAN eps=%.4f using %d-sample %.0fth-percentile of %d-NN distances",
            auto_eps,
            int(meta["sample_size"]),
            meta["percentile"],
            int(meta["neighbor_rank"]),
        )
    if DBSCAN is not None:
        dbscan = DBSCAN(**estimator_kwargs)
        labels_raw = np.asarray(dbscan.fit_predict(Y), dtype=int)
    else:
        labels_raw, _, _ = fit_predict_dbscan(
            Y,
            eps=float(estimator_kwargs["eps"]),
            min_samples=int(estimator_kwargs["min_samples"]),
            metric=estimator_kwargs.get("metric"),
        )
        labels_raw = labels_raw.astype(int, copy=False)
    labels, centers, n_states = _remap_labels_with_noise(Y, labels_raw)
    noise_count = int(np.count_nonzero(labels < 0))
    rationale = summarise_dbscan_kwargs(estimator_kwargs)
    if noise_count:
        rationale = f"{rationale} noise={noise_count}"

    logger.info(
        "DBSCAN clustering complete: %d clusters, %d noise frames",
        n_states,
        noise_count,
    )

    return ClusteringResult(
        labels=labels,
        n_states=n_states,
        rationale=rationale,
        centers=centers,
    )


def cluster_microstates(
    Y: np.ndarray,
    method: Literal["auto", "minibatchkmeans", "kmeans", "dbscan"] = "auto",
    n_states: int | Literal["auto"] = "auto",
    random_state: int | None = 42,
    minibatch_threshold: int = 5_000_000,
    *,
    silhouette_sample_size: int | None = None,
    auto_n_states_override: int | None = None,
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
    method : Literal["auto", "minibatchkmeans", "kmeans", "dbscan"], default="auto"
        Clustering algorithm to use. When ``"auto"`` (the default), the function
        automatically switches to ``MiniBatchKMeans`` when the product of
        ``n_frames * n_features`` exceeds ``minibatch_threshold`` to prevent
        memory issues with large datasets. Selecting ``"dbscan"`` enables the
        density-based clustering workflow and ignores ``minibatch_threshold``.
    n_states : int | Literal["auto"], default="auto"
        Number of microstates to identify. If ``"auto"``, the optimal number
        is selected by maximizing the silhouette score over candidates from 4 to 20.
        If an integer, that exact number of states is used. When
        ``method="dbscan"``, this argument must be ``"auto"`` and the final
        number of clusters is determined directly by DBSCAN.
    random_state : int | None, default=42
        Seed for deterministic clustering. When ``None``, the global NumPy
        random state is used. Ensures reproducible results across runs.
    minibatch_threshold : int, default=5_000_000
        Size threshold for automatic method selection. When the product of
        ``n_frames * n_features`` exceeds this value and ``method="auto"``,
        ``MiniBatchKMeans`` is used instead of ``KMeans``.
    silhouette_sample_size : int | None, keyword-only, default=None
        Number of samples to use when computing silhouette scores during
        automatic state selection. When provided, a random subset of rows from
        ``Y`` with this size is used instead of the full dataset. Sampling is
        reproducible with ``random_state``. Values less than 2 are invalid.
    auto_n_states_override : int | None, keyword-only, default=None
        When ``n_states="auto"``, setting this parameter skips the silhouette
        optimization loop and directly uses the provided number of states.
        Useful when a pre-determined state count is known but the calling code
        still expects automatic selection semantics.
    **kwargs
        Additional keyword arguments forwarded to the underlying
        scikit-learn estimators. For K-means variants (``"auto"``,
        ``"kmeans"``, ``"minibatchkmeans"``) the supported parameters include
        ``max_iter``, ``tol``, ``init``, ``n_init``, ``verbose``,
        ``copy_x`` (KMeans only), ``algorithm`` (KMeans only),
        ``batch_size``, ``compute_labels``, ``max_no_improvement``,
        ``init_size`` and ``reassignment_ratio`` (MiniBatchKMeans only).
        When ``method="dbscan"``, ``**kwargs`` may contain the usual
        scikit-learn DBSCAN parameters such as ``eps``, ``min_samples``,
        ``metric``, ``leaf_size``, ``p``, ``algorithm`` and ``n_jobs``.

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
    sklearn.cluster.KMeans : Standard K-means clustering implementation
    sklearn.cluster.MiniBatchKMeans : Mini-batch variant for large datasets
    """
    # Handle edge case of empty dataset
    if Y.shape[0] == 0:
        logger.info("Empty dataset provided, returning empty clustering result")
        return ClusteringResult(labels=np.empty((0,), dtype=int), n_states=0)

    # Validate input dimensions and data
    _validate_clustering_inputs(Y)

    Y = np.asarray(Y, dtype=float)

    kwargs = dict(kwargs)
    method_normalized = str(method).lower()

    if method_normalized == "dbscan":
        if n_states != "auto":
            raise ValueError(
                "n_states must be 'auto' when clustering with DBSCAN so the algorithm "
                "can determine the number of microstates."
            )
        if "n_init" in kwargs:
            raise ValueError("n_init is not supported when method='dbscan'.")
        logger.info(
            "Starting DBSCAN clustering: %d samples, %d features",
            Y.shape[0],
            Y.shape[1],
        )
        return _cluster_with_dbscan(Y, random_state=random_state, kwargs=kwargs)

    # Store original request for logging
    requested = n_states
    rationale: str | None = None

    # Auto-select number of states if requested
    if isinstance(n_states, str) and n_states == "auto":
        silhouette_kwargs = _prepare_silhouette_kwargs(kwargs)
        n_states, rationale = _auto_select_n_states(
            Y,
            random_state,
            sample_size=silhouette_sample_size,
            override_n_states=auto_n_states_override,
            estimator_kwargs=silhouette_kwargs,
        )
    else:
        n_states = int(n_states)

    if n_states <= 0:
        raise ValueError(
            "Number of microstates must be a positive integer; " f"received {n_states}."
        )

    # Select appropriate clustering algorithm
    chosen_method = _select_clustering_method(method_normalized, Y, minibatch_threshold)

    if chosen_method != "minibatchkmeans":
        disallowed = [key for key in _MINIBATCH_ONLY_KWARGS if key in kwargs]
        if disallowed:
            raise ValueError(
                "Parameters {} are only supported when method='minibatchkmeans'.".format(
                    ", ".join(sorted(disallowed))
                )
            )

    estimator_kwargs = _normalize_kmeans_kwargs(chosen_method, kwargs)

    # Execute clustering
    logger.info(
        "Starting clustering with %s algorithm: %d states, %d samples, %d features",
        chosen_method,
        n_states,
        Y.shape[0],
        Y.shape[1],
    )

    estimator = _create_clustering_estimator(
        chosen_method, n_states, random_state, **estimator_kwargs
    )
    estimator.fit(Y)
    if hasattr(estimator, "labels_"):
        labels = np.asarray(estimator.labels_, dtype=int)
    else:  # pragma: no cover - defensive
        labels = np.asarray(estimator.predict(Y), dtype=int)
    centers_attr = getattr(estimator, "cluster_centers_", None)
    centers = None
    if centers_attr is not None:
        centers = np.asarray(centers_attr, dtype=float)

    unique_labels = int(np.unique(labels).size)

    if unique_labels == 0:
        message = (
            "Clustering produced zero unique microstates; verify input coverage "
            "and CV preprocessing."
        )
        logger.error(message)
        raise ValueError(message)

    if centers is not None and centers.shape[0] != unique_labels:
        logger.warning(
            "Clustering returned %d centers but %d unique labels; trimming metadata.",
            centers.shape[0],
            unique_labels,
        )
        centers = centers[:unique_labels]

    if unique_labels != n_states:
        message = (
            "Clustering produced {unique} unique microstates, expected {expected}. "
            "Proceeding with the observed value; inspect CV spread or adjust "
            "the requested microstate count."
        ).format(unique=unique_labels, expected=n_states)
        logger.warning(message)
        n_states = unique_labels

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
