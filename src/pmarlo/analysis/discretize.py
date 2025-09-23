"""Discretisation helpers for MSM analysis of learned collective variables."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Sequence

import logging

import numpy as np

from sklearn.cluster import KMeans, MiniBatchKMeans


logger = logging.getLogger("pmarlo")


DatasetLike = Mapping[str, Any] | MutableMapping[str, Any]


@dataclass(slots=True)
class MSMDiscretizationResult:
    """Container with the outcome of MSM discretisation."""

    assignments: Dict[str, np.ndarray]
    centers: np.ndarray | None
    counts: np.ndarray
    transition_matrix: np.ndarray
    lag_time: int
    diag_mass: float
    cluster_mode: str


def _looks_like_split(value: Any) -> bool:
    if isinstance(value, (Mapping, MutableMapping)):
        candidate = value.get("X")
        if candidate is None:
            return False
        arr = np.asarray(candidate)
    elif hasattr(value, "X"):
        arr = np.asarray(getattr(value, "X"))
    else:
        arr = np.asarray(value)

    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
        return False
    return bool(np.isfinite(arr).all())


def _normalise_splits(dataset: DatasetLike) -> Dict[str, Any]:
    splits: Dict[str, Any] = {}

    maybe_splits = dataset.get("splits") if isinstance(dataset, Mapping) else None
    if isinstance(maybe_splits, Mapping):
        for name, value in maybe_splits.items():
            if _looks_like_split(value):
                splits[str(name)] = value

    if not splits:
        for name, value in dataset.items():  # type: ignore[assignment]
            if str(name).startswith("__"):
                continue
            if _looks_like_split(value):
                splits[str(name)] = value

    if not splits and _looks_like_split(dataset):
        splits["all"] = dataset

    if not splits:
        raise ValueError("No continuous CV splits found in dataset")

    return splits


def _coerce_array(obj: Any, *, copy: bool = False) -> np.ndarray:
    if isinstance(obj, (Mapping, MutableMapping)):
        arr = obj.get("X")
    elif hasattr(obj, "X"):
        arr = getattr(obj, "X")
    else:
        arr = obj

    array = np.array(arr, dtype=np.float64, copy=copy)
    if array.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {array.shape}")
    if array.shape[0] == 0:
        raise ValueError("Split is empty")
    return array


def _coerce_weights(weights: Any, n_frames: int, split_name: str) -> np.ndarray | None:
    if weights is None:
        return None

    candidate: Any
    if isinstance(weights, Mapping):
        candidate = weights.get(split_name)
    else:
        candidate = weights

    if candidate is None:
        return None

    arr = np.asarray(candidate, dtype=np.float64).reshape(-1)
    if arr.shape[0] != n_frames:
        raise ValueError(
            f"Frame weights for split '{split_name}' have length {arr.shape[0]},"
            f" expected {n_frames}",
        )
    return arr


def _minibatch_threshold(n_frames: int, n_features: int) -> bool:
    return n_frames * n_features >= 5_000_000


class _KMeansDiscretizer:
    def __init__(
        self,
        n_states: int,
        *,
        random_state: int | None = None,
    ) -> None:
        self.n_states = int(n_states)
        self.random_state = random_state
        self.model: KMeans | MiniBatchKMeans | None = None

    def fit(self, X: np.ndarray) -> None:
        if _minibatch_threshold(X.shape[0], X.shape[1]):
            self.model = MiniBatchKMeans(
                n_clusters=self.n_states,
                random_state=self.random_state,
            )
        else:
            self.model = KMeans(
                n_clusters=self.n_states,
                random_state=self.random_state,
                n_init=10,
            )
        self.model.fit(X)

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Discretizer has not been fitted")
        labels = self.model.predict(X)
        return labels.astype(np.int32, copy=False)

    @property
    def centers(self) -> np.ndarray | None:
        if self.model is None:
            return None
        centers = getattr(self.model, "cluster_centers_", None)
        if centers is None:
            return None
        return np.asarray(centers, dtype=np.float64)


class _GridDiscretizer:
    def __init__(self, *, target_states: int) -> None:
        self.target_states = max(int(target_states), 1)
        self.edges: list[np.ndarray] = []
        self.mapping: Dict[tuple[int, ...], int] = {}

    def fit(self, X: np.ndarray) -> None:
        n_features = X.shape[1]
        bins_per_dim = max(int(round(self.target_states ** (1.0 / n_features))), 1)
        self.edges = []
        for col in range(n_features):
            data = X[:, col]
            lo = float(np.min(data))
            hi = float(np.max(data))
            if not np.isfinite(lo) or not np.isfinite(hi):
                raise ValueError("Non-finite values encountered while building grid")
            if lo == hi:
                lo -= 0.5
                hi += 0.5
            self.edges.append(np.linspace(lo, hi, bins_per_dim + 1, dtype=np.float64))
        combos = self._compute_indices(X)
        for combo in combos:
            key = tuple(int(x) for x in combo)
            if key not in self.mapping:
                self.mapping[key] = len(self.mapping)

    def _compute_indices(self, X: np.ndarray) -> np.ndarray:
        indices = []
        for dim, edges in enumerate(self.edges):
            idx = np.clip(np.digitize(X[:, dim], edges) - 1, 0, len(edges) - 2)
            indices.append(idx)
        return np.vstack(indices).T

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.edges:
            raise RuntimeError("Discretizer has not been fitted")
        combos = self._compute_indices(X)
        labels = np.empty(combos.shape[0], dtype=np.int32)
        for i, combo in enumerate(combos):
            key = tuple(int(x) for x in combo)
            state = self.mapping.get(key)
            if state is None:
                state = len(self.mapping)
                self.mapping[key] = state
            labels[i] = state
        return labels

    @property
    def centers(self) -> np.ndarray | None:
        if not self.edges:
            return None
        mesh = np.meshgrid(*[
            (edges[:-1] + edges[1:]) / 2.0 for edges in self.edges
        ], indexing="ij")
        coords = np.stack([m.ravel() for m in mesh], axis=1)
        return coords


def _iter_segments(length: int) -> Iterable[tuple[int, int]]:
    yield 0, length


def _weighted_counts(
    labels: np.ndarray,
    *,
    n_states: int,
    lag_time: int,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    counts = np.zeros((n_states, n_states), dtype=np.float64)
    if labels.size == 0 or lag_time <= 0:
        return counts

    for start, stop in _iter_segments(labels.size):
        length = stop - start
        if length <= lag_time:
            continue
        src = labels[start : stop - lag_time]
        dst = labels[start + lag_time : stop]
        if weights is not None:
            w = weights[start : stop - lag_time]
            np.add.at(counts, (src, dst), w)
        else:
            np.add.at(counts, (src, dst), 1.0)
    return counts


def _normalise_counts(C: np.ndarray) -> np.ndarray:
    row_sums = C.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        T = np.divide(C, row_sums, out=np.zeros_like(C), where=row_sums > 0)
    return T


def discretize_dataset(
    dataset: DatasetLike,
    *,
    cluster_mode: str = "kmeans",
    n_microstates: int = 150,
    lag_time: int = 1,
    frame_weights: Mapping[str, Sequence[float]] | Sequence[float] | np.ndarray | None = None,
    random_state: int | None = None,
) -> MSMDiscretizationResult:
    """Discretise continuous CVs into microstates and build MSM statistics."""

    if lag_time < 1:
        raise ValueError("lag_time must be >= 1")

    splits = _normalise_splits(dataset)
    discretizer: _KMeansDiscretizer | _GridDiscretizer

    train_key = "train" if "train" in splits else next(iter(splits))
    train_data = _coerce_array(splits[train_key])

    if cluster_mode == "kmeans":
        discretizer = _KMeansDiscretizer(n_microstates, random_state=random_state)
    elif cluster_mode == "grid":
        discretizer = _GridDiscretizer(target_states=n_microstates)
    else:
        raise ValueError("cluster_mode must be 'kmeans' or 'grid'")

    discretizer.fit(train_data)

    assignments: Dict[str, np.ndarray] = {}
    max_state = -1
    for name, split in splits.items():
        X = _coerce_array(split)
        labels = discretizer.transform(X)
        assignments[name] = labels
        if labels.size:
            max_state = max(max_state, int(labels.max()))

    n_states = max_state + 1 if max_state >= 0 else 0

    train_labels = assignments[train_key]
    weights = _coerce_weights(frame_weights, train_labels.size, train_key)

    counts = _weighted_counts(
        train_labels,
        n_states=n_states,
        lag_time=lag_time,
        weights=weights,
    )

    transition = _normalise_counts(counts)

    if n_states:
        diag_mass = float(np.trace(transition) / n_states)
    else:
        diag_mass = float("nan")

    if np.isfinite(diag_mass) and diag_mass > 0.95:
        logger.warning("MSM diagonal mass high (%.3f)", diag_mass)

    zero_states = np.where(counts.sum(axis=1) == 0)[0]
    if counts.shape[0] and zero_states.size / counts.shape[0] > 0.3:
        logger.warning(
            "More than 30%% of the microstates are empty (%d/%d)",
            zero_states.size,
            counts.shape[0],
        )

    return MSMDiscretizationResult(
        assignments=assignments,
        centers=discretizer.centers,
        counts=counts,
        transition_matrix=transition,
        lag_time=lag_time,
        diag_mass=diag_mass,
        cluster_mode=cluster_mode,
    )
