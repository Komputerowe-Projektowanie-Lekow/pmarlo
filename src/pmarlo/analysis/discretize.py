"""Discretisation helpers for MSM analysis of learned collective variables."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans

from .counting import expected_pairs
from .errors import CountingLogicError, PruningFailedError
from .validation import validate_features

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
    assignment_masks: Dict[str, np.ndarray] = field(default_factory=dict)
    segment_lengths: Dict[str, List[int]] = field(default_factory=dict)
    segment_strides: Dict[str, List[int]] = field(default_factory=dict)
    counted_pairs: Dict[str, int] = field(default_factory=dict)
    expected_pairs: Dict[str, int] = field(default_factory=dict)
    feature_schema: Dict[str, Any] = field(default_factory=dict)
    fingerprint: Dict[str, Any] = field(default_factory=dict)
    feature_stats: Dict[str, Any] = field(default_factory=dict)
    counts_before_prune: np.ndarray | None = None
    state_counts_before_prune: np.ndarray | None = None
    state_counts: np.ndarray | None = None
    pruned_state_indices: np.ndarray | None = None


class FeatureMismatchError(ValueError):
    """Raised when feature schemas between splits are incompatible."""

    def __init__(
        self,
        message: str,
        *,
        differences: Sequence[str] | None = None,
        expected: Mapping[str, Any] | None = None,
        actual: Mapping[str, Any] | None = None,
    ) -> None:
        self.differences = list(differences or [])
        self.expected_schema = dict(expected or {})
        self.actual_schema = dict(actual or {})
        if self.differences:
            diff_text = "\n".join(f"- {item}" for item in self.differences)
            message = f"{message}\n{diff_text}"
        super().__init__(message)


class NoAssignmentsError(RuntimeError):
    """Raised when no frames receive a valid state assignment."""

    def __init__(
        self,
        split: str,
        *,
        preview: np.ndarray,
        center_norms: np.ndarray | None,
    ) -> None:
        self.split = str(split)
        self.preview = np.asarray(preview)
        self.center_norms = None if center_norms is None else np.asarray(center_norms)
        preview_repr = repr(self.preview)
        centers_repr = (
            "[]"
            if self.center_norms is None
            else repr(self.center_norms.astype(float, copy=False))
        )
        super().__init__(
            "No state assignments were produced for split"
            f" '{self.split}'. Preview (up to 10 rows): {preview_repr}; "
            f"centroid norms: {centers_repr}"
        )


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


def _coerce_feature_names(raw: Any) -> list[str]:
    try:
        if raw is None:
            return []
        if isinstance(raw, Mapping):
            raw = raw.get("names")
        if raw is None:
            return []
        if isinstance(raw, (str, bytes)):
            return [str(raw)]
        return [str(item) for item in raw if item is not None]
    except Exception:
        return []


def _extract_feature_schema(split: Any, n_features: int) -> Dict[str, Any]:
    names: list[str] = []

    if isinstance(split, (Mapping, MutableMapping)):
        schema = split.get("feature_schema")
        if isinstance(schema, Mapping):
            names = _coerce_feature_names(schema)
        if not names:
            candidate = split.get("cv_names")
            if candidate is None:
                candidate = split.get("feature_names")
            names = _coerce_feature_names(candidate)
    else:
        schema = getattr(split, "feature_schema", None)
        if isinstance(schema, Mapping):
            names = _coerce_feature_names(schema)
        if not names and hasattr(split, "feature_names"):
            names = _coerce_feature_names(getattr(split, "feature_names"))
        if not names and hasattr(split, "columns"):
            names = _coerce_feature_names(getattr(split, "columns"))

    return {"names": names, "n_features": int(n_features)}


def _diff_feature_names(expected: Sequence[str], actual: Sequence[str]) -> list[str]:
    if not expected:
        return []

    expected_list = list(expected)
    actual_list = list(actual)
    differences: list[str] = []

    expected_set = {name for name in expected_list}
    actual_set = {name for name in actual_list}

    missing = [name for name in expected_list if name not in actual_set]
    if missing:
        differences.append("missing: " + ", ".join(repr(name) for name in missing))

    unexpected = [name for name in actual_list if name not in expected_set]
    if unexpected:
        differences.append(
            "unexpected: " + ", ".join(repr(name) for name in unexpected)
        )

    if not missing and not unexpected:
        mismatched_positions = [
            f"position {idx}: expected {repr(exp)}, got {repr(act)}"
            for idx, (exp, act) in enumerate(zip(expected_list, actual_list))
            if exp != act
        ]
        if mismatched_positions:
            differences.append("order mismatch: " + "; ".join(mismatched_positions))

    return differences


def _validate_feature_schema(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    split_name: str,
) -> None:
    expected_count = int(reference.get("n_features", 0))
    actual_count = int(candidate.get("n_features", 0))
    differences: list[str] = []

    if actual_count != expected_count:
        differences.append(
            f"n_features mismatch: expected {expected_count}, got {actual_count}"
        )
    else:
        expected_names = list(reference.get("names") or [])
        actual_names = list(candidate.get("names") or [])
        differences.extend(_diff_feature_names(expected_names, actual_names))

    if differences:
        raise FeatureMismatchError(
            f"Feature schema mismatch for split '{split_name}'",
            differences=differences,
            expected=reference,
            actual=candidate,
        )


def _normalise_feature_schema_for_fit(
    feature_schema: Mapping[str, Any] | None,
    n_features: int,
) -> Dict[str, Any]:
    schema = dict(feature_schema or {})
    observed = int(schema.get("n_features", n_features))
    if observed != n_features:
        raise ValueError(
            "Feature schema reports "
            f"{observed} features, but training data has {n_features}"
        )

    names = list(schema.get("names") or [])
    if names:
        if len(names) != n_features:
            raise ValueError(
                "Feature schema names length "
                f"{len(names)} does not match n_features {n_features}"
            )
        schema["names"] = [str(name) for name in names]
    else:
        schema["names"] = [f"feature_{idx}" for idx in range(n_features)]

    schema["n_features"] = n_features
    return schema


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


def _append_segment(
    lengths: list[int],
    strides: list[int],
    length: Any,
    stride_value: Any,
) -> None:
    try:
        length_int = int(length)
    except Exception:
        return
    if length_int <= 0:
        return
    try:
        stride_int = 1 if stride_value is None else int(stride_value)
    except Exception:
        stride_int = 1
    if stride_int <= 0:
        stride_int = 1
    lengths.append(length_int)
    strides.append(stride_int)


def _segments_from_split_metadata(
    split: Mapping[str, Any],
) -> tuple[list[int], list[int]]:
    lengths: list[int] = []
    strides: list[int] = []

    segments_meta = split.get("segments") or split.get("__segments__")
    if isinstance(segments_meta, Iterable):
        for entry in segments_meta:
            if isinstance(entry, Mapping):
                length_val = entry.get("length")
                if length_val is None:
                    start = entry.get("start")
                    stop = entry.get("stop")
                    if start is not None and stop is not None:
                        try:
                            length_val = int(stop) - int(start)
                        except Exception:
                            length_val = None
                stride_val = entry.get("stride") or entry.get("effective_frame_stride")
            else:
                length_val = entry
                stride_val = None
            if length_val is not None:
                _append_segment(lengths, strides, length_val, stride_val)

    if not lengths:
        raw_lengths = split.get("segment_lengths")
        if isinstance(raw_lengths, Iterable):
            for value in raw_lengths:
                _append_segment(lengths, strides, value, None)

    return lengths, strides


def _segments_from_dataset_shards(
    dataset: DatasetLike, split_name: str
) -> tuple[list[int], list[int]]:
    lengths: list[int] = []
    strides: list[int] = []

    shards = dataset.get("__shards__") if isinstance(dataset, Mapping) else None
    if not isinstance(shards, Iterable):
        return lengths, strides

    for entry in shards:
        if not isinstance(entry, Mapping):
            continue
        split_label = entry.get("split")
        if split_label is not None and str(split_label) != split_name:
            continue
        length_val = entry.get("length")
        if length_val is None:
            start = entry.get("start")
            stop = entry.get("stop")
            if start is not None and stop is not None:
                try:
                    length_val = int(stop) - int(start)
                except Exception:
                    length_val = None
        stride_val = entry.get("effective_frame_stride")
        if length_val is not None:
            _append_segment(lengths, strides, length_val, stride_val)
    return lengths, strides


def _truncate_segments(
    lengths: Sequence[int],
    strides: Sequence[int],
    total_frames: int,
) -> tuple[list[int], list[int]]:
    consumed = 0
    sanitised_lengths: list[int] = []
    sanitised_strides: list[int] = []
    for length, stride in zip(lengths, strides):
        if consumed >= total_frames:
            break
        remaining = total_frames - consumed
        value = min(int(length), remaining)
        if value <= 0:
            continue
        sanitised_lengths.append(value)
        sanitised_strides.append(max(1, int(stride)))
        consumed += value

    if not sanitised_lengths and total_frames > 0:
        return [total_frames], [1]
    return sanitised_lengths, sanitised_strides


def _resolve_shard_segments_for_split(
    dataset: DatasetLike,
    split_name: str,
    split: Any,
    total_frames: int,
) -> tuple[list[int], list[int]]:
    lengths: list[int] = []
    strides: list[int] = []

    if isinstance(split, Mapping):
        meta_lengths, meta_strides = _segments_from_split_metadata(split)
        lengths.extend(meta_lengths)
        strides.extend(meta_strides)

    if not lengths:
        shard_lengths, shard_strides = _segments_from_dataset_shards(
            dataset, split_name
        )
        lengths.extend(shard_lengths)
        strides.extend(shard_strides)

    return _truncate_segments(lengths, strides, total_frames)


def _lengths_to_segments(
    lengths: Sequence[int], total_frames: int
) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    offset = 0
    for value in lengths:
        try:
            length_int = int(value)
        except Exception:
            continue
        if length_int <= 0:
            continue
        start = offset
        stop = min(total_frames, offset + length_int)
        if stop > start:
            segments.append((start, stop))
        offset = stop
        if offset >= total_frames:
            break
    if not segments:
        segments.append((0, total_frames))
    return segments


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
        self.feature_schema: Dict[str, Any] | None = None

    def fit(
        self,
        X: np.ndarray,
        feature_schema: Mapping[str, Any] | None = None,
    ) -> None:
        self.feature_schema = _normalise_feature_schema_for_fit(
            feature_schema,
            X.shape[1],
        )
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

    def transform(
        self,
        X: np.ndarray,
        feature_schema: Mapping[str, Any] | None = None,
        *,
        split_name: str | None = None,
    ) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Discretizer has not been fitted")
        if feature_schema is not None and self.feature_schema is not None:
            _validate_feature_schema(
                self.feature_schema,
                feature_schema,
                split_name=split_name or "split",
            )
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
        self.feature_schema: Dict[str, Any] | None = None

    def fit(
        self,
        X: np.ndarray,
        feature_schema: Mapping[str, Any] | None = None,
    ) -> None:
        self.feature_schema = _normalise_feature_schema_for_fit(
            feature_schema,
            X.shape[1],
        )
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

    def transform(
        self,
        X: np.ndarray,
        feature_schema: Mapping[str, Any] | None = None,
        *,
        split_name: str | None = None,
    ) -> np.ndarray:
        if not self.edges:
            raise RuntimeError("Discretizer has not been fitted")
        if feature_schema is not None and self.feature_schema is not None:
            _validate_feature_schema(
                self.feature_schema,
                feature_schema,
                split_name=split_name or "split",
            )
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
        mesh = np.meshgrid(
            *[(edges[:-1] + edges[1:]) / 2.0 for edges in self.edges], indexing="ij"
        )
        coords = np.stack([m.ravel() for m in mesh], axis=1)
        return coords


def _iter_segments(
    length: int, segments: Iterable[tuple[int, int]] | None = None
) -> Iterable[tuple[int, int]]:
    if segments is not None:
        for start, stop in segments:
            start_idx = max(0, int(start))
            stop_idx = min(length, int(stop))
            if stop_idx > start_idx:
                yield start_idx, stop_idx
        return
    yield 0, length


def _weighted_counts(
    labels: np.ndarray,
    *,
    n_states: int,
    lag_time: int,
    weights: np.ndarray | None = None,
    segments: Iterable[tuple[int, int]] | None = None,
    stride: int = 1,
) -> tuple[np.ndarray, int]:
    counts = np.zeros((n_states, n_states), dtype=np.float64)
    if labels.size == 0 or lag_time <= 0:
        return counts, 0

    total_pairs = 0
    step = max(1, int(stride))

    for start, stop in _iter_segments(labels.size, segments):
        length = stop - start
        if length <= lag_time:
            continue
        src = labels[start : stop - lag_time : step]
        dst = labels[start + lag_time : stop : step]
        if src.size == 0:
            continue
        if weights is not None:
            seg_weights = weights[start : stop - lag_time : step]
        else:
            seg_weights = None
        valid = (src >= 0) & (dst >= 0)
        if not np.any(valid):
            continue
        if seg_weights is not None:
            np.add.at(counts, (src[valid], dst[valid]), seg_weights[valid])
        else:
            np.add.at(counts, (src[valid], dst[valid]), 1.0)
        total_pairs += int(np.count_nonzero(valid))
    return counts, total_pairs


def _compute_state_counts(
    labels: np.ndarray,
    *,
    n_states: int,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    if n_states <= 0:
        return np.zeros((0,), dtype=np.float64)
    valid = (labels >= 0) & (labels < n_states)
    if not np.any(valid):
        return np.zeros((n_states,), dtype=np.float64)
    if weights is not None:
        counts = np.bincount(
            labels[valid],
            weights=weights[valid],
            minlength=n_states,
        )
    else:
        counts = np.bincount(labels[valid], minlength=n_states)
    return counts.astype(np.float64, copy=False)


def _remap_states(labels: np.ndarray, mapping: np.ndarray) -> np.ndarray:
    remapped = np.full_like(labels, -1)
    valid = (labels >= 0) & (labels < mapping.size)
    remapped_valid = mapping[labels[valid]]
    remapped[valid] = remapped_valid
    return remapped.astype(np.int32, copy=False)


def _normalise_counts(C: np.ndarray) -> np.ndarray:
    row_sums = C.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        T = np.divide(C, row_sums, out=np.zeros_like(C), where=row_sums > 0)
    return T


def _prepare_discretizer_and_schema(
    splits: Mapping[str, Any],
    *,
    cluster_mode: str,
    n_microstates: int,
    random_state: int | None,
) -> tuple[
    str,
    np.ndarray,
    Dict[str, Any],
    Dict[str, Dict[str, Any]],
    _KMeansDiscretizer | _GridDiscretizer,
]:
    train_key = "train" if "train" in splits else next(iter(splits))
    train_data = _coerce_array(splits[train_key])
    feature_schema = _extract_feature_schema(splits[train_key], train_data.shape[1])
    feature_names = list(feature_schema.get("names", []))

    stats_by_split: Dict[str, Dict[str, Any]] = {}
    stats_by_split[train_key] = validate_features(train_data, feature_names)

    discretizer: _KMeansDiscretizer | _GridDiscretizer
    if cluster_mode == "kmeans":
        discretizer = _KMeansDiscretizer(n_microstates, random_state=random_state)
    elif cluster_mode == "grid":
        discretizer = _GridDiscretizer(target_states=n_microstates)
    else:
        raise ValueError("cluster_mode must be 'kmeans' or 'grid'")

    discretizer.fit(train_data, feature_schema)
    fitted_schema = discretizer.feature_schema or feature_schema
    feature_schema = fitted_schema
    feature_names = list(feature_schema.get("names", []))
    stats_by_split[train_key]["feature_names"] = feature_names
    stats_by_split[train_key]["n_features"] = int(
        feature_schema.get("n_features", train_data.shape[1])
    )
    return train_key, train_data, feature_schema, stats_by_split, discretizer


def _process_split_assignment(
    dataset: DatasetLike,
    split_name: str,
    split: Any,
    feature_schema: Mapping[str, Any],
    discretizer: _KMeansDiscretizer | _GridDiscretizer,
    *,
    lag_time: int,
    stats_by_split: Dict[str, Dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, list[int], list[int]]:
    X = _coerce_array(split)
    split_schema = _extract_feature_schema(split, X.shape[1])
    _validate_feature_schema(feature_schema, split_schema, split_name=split_name)

    split_names = list(split_schema.get("names", []))
    stats = stats_by_split.get(split_name)
    if stats is None:
        stats = validate_features(X, split_names)
        stats_by_split[split_name] = stats
    stats["feature_names"] = split_names
    stats["n_features"] = int(split_schema.get("n_features", X.shape[1]))

    split_lengths, split_strides = _resolve_shard_segments_for_split(
        dataset, split_name, split, X.shape[0]
    )
    stats["segment_lengths"] = list(split_lengths)
    stats["expected_pairs"] = expected_pairs(split_lengths, lag_time, 1)
    stats["segment_strides"] = list(split_strides)

    labels = discretizer.transform(
        X,
        feature_schema=split_schema,
        split_name=split_name,
    )
    labels = np.asarray(labels, dtype=np.int32)
    valid_mask = np.isfinite(labels) & (labels >= 0)
    n_assigned = int(np.count_nonzero(valid_mask))
    unique_states = (
        np.unique(labels[valid_mask]) if n_assigned else np.asarray([], dtype=np.int32)
    )
    logger.debug(
        "Split %s state assignment summary: %d assigned frames across %d states",
        split_name,
        n_assigned,
        unique_states.size,
    )
    if n_assigned == 0:
        preview = np.array(X[:10], dtype=np.float64, copy=False)
        centers = discretizer.centers
        center_norms = (
            None
            if centers is None
            else np.linalg.norm(np.asarray(centers, dtype=np.float64), axis=1)
        )
        raise NoAssignmentsError(
            split_name,
            preview=preview,
            center_norms=center_norms,
        )

    return (
        labels,
        valid_mask.astype(bool, copy=False),
        list(split_lengths),
        list(split_strides),
    )


def _ensure_train_assignments(
    train_key: str,
    train_data: np.ndarray,
    discretizer: _KMeansDiscretizer | _GridDiscretizer,
    train_mask: np.ndarray,
) -> None:
    if np.all(train_mask):
        return
    preview = np.array(train_data[:10], dtype=np.float64, copy=False)
    centers = discretizer.centers
    center_norms = (
        None
        if centers is None
        else np.linalg.norm(np.asarray(centers, dtype=np.float64), axis=1)
    )
    raise NoAssignmentsError(
        train_key,
        preview=preview,
        center_norms=center_norms,
    )


def _prune_zero_rows_if_needed(
    counts: np.ndarray,
    counted_pairs_train: int,
    train_labels: np.ndarray,
    assignments: Dict[str, np.ndarray],
    assignment_masks: Dict[str, np.ndarray],
    *,
    zero_rows_before: int,
    row_sums_before: np.ndarray,
    min_out_threshold: int,
    lag_time: int,
    weights: np.ndarray | None,
    train_segments: Iterable[tuple[int, int]],
    train_key: str,
) -> tuple[np.ndarray, int, np.ndarray, np.ndarray | None, int, int]:
    n_states = counts.shape[0]
    if zero_rows_before == 0:
        return (
            counts,
            counted_pairs_train,
            train_labels,
            None,
            zero_rows_before,
            n_states,
        )

    prune_mask = row_sums_before == 0
    if min_out_threshold > 0:
        prune_mask |= row_sums_before < float(min_out_threshold)
    keep_mask = ~prune_mask
    if not np.any(keep_mask):
        raise PruningFailedError(
            f"Pruning removed all microstates (zero_rows={zero_rows_before}, "
            f"min_out_count={min_out_threshold})"
        )

    pruned_state_indices = np.where(prune_mask)[0].astype(np.int32, copy=False)
    mapping = np.full(n_states, -1, dtype=np.int32)
    mapping[keep_mask] = np.arange(int(np.count_nonzero(keep_mask)), dtype=np.int32)

    for split_name, labels in list(assignments.items()):
        remapped = _remap_states(
            np.asarray(labels, dtype=np.int32, copy=False), mapping
        )
        assignments[split_name] = remapped
        mask = assignment_masks.get(split_name)
        if mask is not None:
            assignment_masks[split_name] = np.asarray(mask, dtype=bool) & (
                remapped >= 0
            )

    train_labels = assignments[train_key]
    n_states = int(np.count_nonzero(keep_mask))
    counts, counted_pairs_train = _weighted_counts(
        train_labels,
        n_states=n_states,
        lag_time=lag_time,
        weights=weights,
        segments=train_segments,
    )
    row_sums_after = counts.sum(axis=1)
    zero_rows_after = int(np.count_nonzero(row_sums_after == 0))
    if zero_rows_after > 0:
        raise PruningFailedError(
            f"Pruning left {zero_rows_after} zero-row microstates "
            f"(min_out_count={min_out_threshold})"
        )

    return (
        counts,
        counted_pairs_train,
        train_labels,
        pruned_state_indices,
        zero_rows_after,
        n_states,
    )


def discretize_dataset(
    dataset: DatasetLike,
    *,
    cluster_mode: str = "kmeans",
    n_microstates: int = 150,
    lag_time: int = 1,
    frame_weights: (
        Mapping[str, Sequence[float]] | Sequence[float] | np.ndarray | None
    ) = None,
    min_out_count: int = 0,
    random_state: int | None = None,
) -> MSMDiscretizationResult:
    """Discretise continuous CVs into microstates and build MSM statistics."""

    if lag_time < 1:
        raise ValueError("lag_time must be >= 1")

    splits = _normalise_splits(dataset)
    (
        train_key,
        train_data,
        feature_schema,
        stats_by_split,
        discretizer,
    ) = _prepare_discretizer_and_schema(
        splits,
        cluster_mode=cluster_mode,
        n_microstates=n_microstates,
        random_state=random_state,
    )

    segment_lengths_by_split: Dict[str, List[int]] = {}
    segment_strides_by_split: Dict[str, List[int]] = {}
    assignments: Dict[str, np.ndarray] = {}
    assignment_masks: Dict[str, np.ndarray] = {}
    max_state = -1

    for name, split in splits.items():
        (
            labels,
            valid_mask,
            split_lengths,
            split_strides,
        ) = _process_split_assignment(
            dataset,
            name,
            split,
            feature_schema,
            discretizer,
            lag_time=lag_time,
            stats_by_split=stats_by_split,
        )
        assignments[name] = labels
        assignment_masks[name] = valid_mask
        segment_lengths_by_split[name] = split_lengths
        segment_strides_by_split[name] = split_strides
        if labels.size:
            max_state = max(max_state, int(labels.max()))

    n_states = max_state + 1 if max_state >= 0 else 0

    train_labels = np.asarray(assignments[train_key], dtype=np.int32, copy=False)
    train_mask = assignment_masks[train_key]
    _ensure_train_assignments(train_key, train_data, discretizer, train_mask)

    weights = _coerce_weights(frame_weights, train_labels.size, train_key)

    train_lengths = segment_lengths_by_split.get(train_key) or [train_labels.size]
    train_segments = _lengths_to_segments(train_lengths, train_labels.size)

    counts, counted_pairs_train = _weighted_counts(
        train_labels,
        n_states=n_states,
        lag_time=lag_time,
        weights=weights,
        segments=train_segments,
    )
    counts_before_prune = counts.copy()
    counted_pairs_before = counted_pairs_train
    expected_pairs_train = expected_pairs(train_lengths, lag_time, 1)

    state_counts_before = _compute_state_counts(
        train_labels,
        n_states=n_states,
        weights=weights,
    )

    row_sums_before = counts_before_prune.sum(axis=1)
    zero_rows_before = int(np.count_nonzero(row_sums_before == 0))
    min_out_threshold = max(0, int(min_out_count))
    (
        counts,
        counted_pairs_train,
        train_labels,
        pruned_state_indices,
        zero_rows_after,
        n_states,
    ) = _prune_zero_rows_if_needed(
        counts,
        counted_pairs_train,
        train_labels,
        assignments,
        assignment_masks,
        zero_rows_before=zero_rows_before,
        row_sums_before=row_sums_before,
        min_out_threshold=min_out_threshold,
        lag_time=lag_time,
        weights=weights,
        train_segments=train_segments,
        train_key=train_key,
    )

    state_counts_final = _compute_state_counts(
        train_labels,
        n_states=n_states,
        weights=weights,
    )

    if expected_pairs_train > 0 and counted_pairs_train == 0:
        raise CountingLogicError(
            f"No transition pairs counted for split '{train_key}' "
            f"despite expected {expected_pairs_train} pairs"
        )

    stats_by_split[train_key]["expected_pairs"] = int(expected_pairs_train)
    stats_by_split[train_key]["counted_pairs_before_prune"] = int(counted_pairs_before)
    stats_by_split[train_key]["counted_pairs"] = int(counted_pairs_train)
    stats_by_split[train_key]["zero_rows_before_prune"] = int(zero_rows_before)
    stats_by_split[train_key]["zero_rows_after_prune"] = int(zero_rows_after)
    if pruned_state_indices is not None and pruned_state_indices.size:
        stats_by_split[train_key]["pruned_state_indices"] = pruned_state_indices.astype(
            int
        ).tolist()
        stats_by_split[train_key]["prune_min_out_count"] = int(min_out_threshold)

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

    fingerprint = {
        "mode": str(cluster_mode),
        "n_states": int(max(n_states, 0)),
        "seed": None if random_state is None else int(random_state),
        "feature_schema": {
            "names": list(feature_schema.get("names", [])),
            "n_features": int(feature_schema.get("n_features", 0)),
        },
        "expected_pairs": int(expected_pairs_train),
        "counted_pairs": int(counted_pairs_train),
        "segment_lengths": {train_key: train_lengths},
        "segment_strides": {train_key: segment_strides_by_split.get(train_key, [])},
        "zero_rows_before_prune": int(zero_rows_before),
        "zero_rows_after_prune": int(zero_rows_after),
        "pruned_state_count": (
            int(pruned_state_indices.size) if pruned_state_indices is not None else 0
        ),
        "min_out_count": int(min_out_threshold),
    }

    return MSMDiscretizationResult(
        assignments=assignments,
        assignment_masks=assignment_masks,
        segment_lengths=segment_lengths_by_split,
        segment_strides=segment_strides_by_split,
        counted_pairs={train_key: int(counted_pairs_train)},
        expected_pairs={train_key: int(expected_pairs_train)},
        centers=discretizer.centers,
        counts=counts,
        transition_matrix=transition,
        lag_time=lag_time,
        diag_mass=diag_mass,
        cluster_mode=cluster_mode,
        feature_schema=feature_schema,
        fingerprint=fingerprint,
        feature_stats=stats_by_split,
        counts_before_prune=counts_before_prune,
        state_counts_before_prune=state_counts_before,
        state_counts=state_counts_final,
        pruned_state_indices=pruned_state_indices,
    )
