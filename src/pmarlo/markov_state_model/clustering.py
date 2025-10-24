"""Microstate clustering utilities for Markov state model construction.

This module provides intelligent clustering of reduced-dimensional feature data
into microstates, which serve as the foundation for Markov state model (MSM)
analysis. The implementation relies on :mod:`deeptime`'s tested K-Means
estimators instead of maintaining custom wrappers around scikit-learn.

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
from typing import Any, Dict, Literal, Mapping, cast

import numpy as np
from sklearn.metrics import silhouette_score

from deeptime.clustering import KMeans, MiniBatchKMeans

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
        km = _create_clustering_estimator(
            "kmeans", n, random_state, **estimator_kwargs
        )
        model = km.fit_fetch(Y_sample)
        labels = np.asarray(model.transform(Y_sample), dtype=int)

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


_COMMON_KWARGS: frozenset[str] = frozenset(
    {"max_iter", "metric", "tolerance", "init_strategy", "n_jobs", "initial_centers"}
)
_ATTRIBUTE_KWARGS: frozenset[str] = frozenset({"fixed_seed", "progress"})
_MINIBATCH_ONLY_KWARGS: frozenset[str] = frozenset({"batch_size"})
_SUPPORTED_KWARGS: frozenset[str] = frozenset(
    set(_COMMON_KWARGS)
    | set(_ATTRIBUTE_KWARGS)
    | set(_MINIBATCH_ONLY_KWARGS)
)


def _validate_clustering_kwargs(
    method: Literal["auto", "minibatchkmeans", "kmeans"], kwargs: Mapping[str, Any]
) -> None:
    unsupported = set(kwargs) - set(_SUPPORTED_KWARGS)
    if unsupported:
        raise TypeError(
            "Unsupported clustering parameters for deeptime backend: "
            f"{sorted(unsupported)}"
        )

    if method == "kmeans" and any(k in kwargs for k in _MINIBATCH_ONLY_KWARGS):
        raise TypeError(
            "'batch_size' is only supported when method='minibatchkmeans'."
        )


def _split_kwargs_for_method(
    method: Literal["minibatchkmeans", "kmeans"], kwargs: Mapping[str, Any]
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    init_kwargs: Dict[str, Any] = {}
    attribute_kwargs: Dict[str, Any] = {}

    allowed_init = set(_COMMON_KWARGS)
    if method == "minibatchkmeans":
        allowed_init |= set(_MINIBATCH_ONLY_KWARGS)
    else:
        allowed_init |= {"progress"}

    for key, value in kwargs.items():
        if key in _ATTRIBUTE_KWARGS:
            attribute_kwargs[key] = value
        elif key in allowed_init:
            init_kwargs[key] = value

    return init_kwargs, attribute_kwargs


def _resolve_fixed_seed(
    random_state: int | None, attribute_kwargs: Dict[str, Any]
) -> int | bool:
    if "fixed_seed" in attribute_kwargs:
        return attribute_kwargs.pop("fixed_seed")
    if random_state is None:
        return False
    return int(random_state)


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
    if method not in {"minibatchkmeans", "kmeans"}:
        raise ValueError(f"Unsupported method: {method}")

    init_kwargs, attribute_kwargs = _split_kwargs_for_method(method, kwargs)
    fixed_seed = _resolve_fixed_seed(random_state, attribute_kwargs)

    if method == "minibatchkmeans":
        estimator = MiniBatchKMeans(n_clusters=n_states, **init_kwargs)
        estimator.fixed_seed = fixed_seed
        if "progress" in attribute_kwargs:
            estimator.progress = attribute_kwargs["progress"]
        return estimator

    estimator = KMeans(n_clusters=n_states, fixed_seed=fixed_seed, **init_kwargs)
    if "progress" in attribute_kwargs:
        estimator.progress = attribute_kwargs["progress"]
    return estimator


def _remap_labels_and_compute_inertia(
    Y: np.ndarray, raw_labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray | None, int, float]:
    """Remap raw clustering labels to a dense range and compute inertia."""

    unique_labels = np.unique(raw_labels)
    n_unique = int(unique_labels.size)
    if n_unique == 0:
        raise ValueError(
            "Clustering produced zero unique microstates; verify input coverage "
            "and CV preprocessing."
        )

    label_map = {int(label): idx for idx, label in enumerate(unique_labels)}
    remapped = np.empty_like(raw_labels, dtype=int)
    for idx, label in enumerate(raw_labels):
        remapped[idx] = label_map[int(label)]

    centers = np.zeros((n_unique, Y.shape[1]), dtype=float)
    for original_label, dense_label in label_map.items():
        mask = raw_labels == original_label
        if not np.any(mask):
            continue
        centers[dense_label] = np.asarray(Y[mask], dtype=float).mean(axis=0)

    diffs = Y - centers[remapped]
    inertia = float(np.sum(diffs * diffs))

    return remapped, centers if centers.size else None, n_unique, inertia


def cluster_microstates(
    Y: np.ndarray,
    method: Literal["auto", "minibatchkmeans", "kmeans"] = "auto",
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
        :mod:`deeptime.clustering` estimators. Supported parameters include
        ``max_iter``, ``metric``, ``tolerance``, ``init_strategy``, ``n_jobs``,
        and ``initial_centers`` for all methods. ``progress`` and ``fixed_seed``
        can be supplied to control deterministic behaviour for standard
        ``KMeans`` clustering. ``batch_size`` is supported only when
        ``method="minibatchkmeans"``. Supplying ``n_init`` performs multiple
        clustering restarts using different seeds and selects the run with the
        lowest within-cluster sum of squares.

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
    deeptime.clustering.KMeans : Standard K-means clustering implementation
    deeptime.clustering.MiniBatchKMeans : Mini-batch variant for large datasets
    """
    # Handle edge case of empty dataset
    if Y.shape[0] == 0:
        logger.info("Empty dataset provided, returning empty clustering result")
        return ClusteringResult(labels=np.empty((0,), dtype=int), n_states=0)

    # Validate input dimensions and data
    _validate_clustering_inputs(Y)

    Y = np.asarray(Y, dtype=float)

    kwargs = dict(kwargs)
    raw_n_init = kwargs.pop("n_init", None)
    if raw_n_init is None:
        n_init_restarts = 1
    else:
        try:
            n_init_restarts = int(raw_n_init)
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive guard
            raise TypeError(
                "n_init must be provided as an integer when clustering with deeptime"
            ) from exc
        if n_init_restarts <= 0:
            raise ValueError(
                "n_init must be a positive integer when clustering microstates"
            )

    if n_init_restarts > 1 and "fixed_seed" in kwargs:
        raise ValueError(
            "n_init cannot be combined with fixed_seed; provide only one mechanism "
            "for controlling clustering initialisations."
        )

    _validate_clustering_kwargs(method, kwargs)

    # Store original request for logging
    requested = n_states
    rationale: str | None = None

    # Auto-select number of states if requested
    if isinstance(n_states, str) and n_states == "auto":
        silhouette_kwargs = _split_kwargs_for_method("kmeans", kwargs)[0]
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
    chosen_method = _select_clustering_method(method, Y, minibatch_threshold)

    if "batch_size" in kwargs and chosen_method != "minibatchkmeans":
        raise ValueError(
            "batch_size was provided but the selected clustering method is "
            f"'{chosen_method}'. Specify method='minibatchkmeans' to use mini-batch "
            "parameters."
        )

    # Execute clustering with optional restarts
    logger.info(
        "Starting clustering with %s algorithm: %d states, %d samples, %d features",
        chosen_method,
        n_states,
        Y.shape[0],
        Y.shape[1],
    )

    seeds: list[int | None]
    if n_init_restarts == 1:
        seeds = [random_state]
    else:
        rng = np.random.default_rng(random_state)
        seeds = []
        if random_state is None:
            seeds.append(None)
        else:
            seeds.append(int(random_state))
        existing = {s for s in seeds if isinstance(s, int)}
        while len(seeds) < n_init_restarts:
            candidate = int(rng.integers(0, np.iinfo(np.int32).max))
            if candidate in existing:
                continue
            seeds.append(candidate)
            existing.add(candidate)

    best_run: dict[str, Any] | None = None

    for init_index, seed in enumerate(seeds):
        estimator = _create_clustering_estimator(
            chosen_method, n_states, seed, **kwargs
        )
        model = estimator.fit_fetch(Y)
        labels_raw = cast(np.ndarray, np.asarray(model.transform(Y), dtype=int))

        try:
            labels, centers, unique_labels, inertia = _remap_labels_and_compute_inertia(
                Y, labels_raw
            )
        except ValueError as exc:
            logger.error("%s", exc)
            raise

        run_info = {
            "labels": labels,
            "centers": centers,
            "unique": unique_labels,
            "inertia": inertia,
            "seed": seed,
            "iteration": init_index,
        }

        if best_run is None or inertia < best_run["inertia"]:
            best_run = run_info

    assert best_run is not None  # for mypy

    labels = cast(np.ndarray, best_run["labels"])
    centers = best_run["centers"]
    unique_labels = int(best_run["unique"])

    if n_init_restarts > 1:
        logger.info(
            "Selected best clustering from %d initialisations (iteration=%d, seed=%s, inertia=%.6f)",
            n_init_restarts,
            int(best_run["iteration"]),
            "None" if best_run["seed"] is None else int(best_run["seed"]),
            float(best_run["inertia"]),
        )

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
