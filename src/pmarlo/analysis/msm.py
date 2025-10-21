"""Utilities for preparing MSM inputs with learned CV whitening."""

from __future__ import annotations

from typing import Any, Mapping, MutableMapping

import numpy as np

from .discretize import MSMDiscretizationResult, discretize_dataset
from .project_cv import apply_whitening_from_metadata

DatasetLike = MutableMapping[str, Any]


def ensure_msm_inputs_whitened(dataset: DatasetLike | Mapping[str, Any]) -> bool:
    """Ensure the continuous CV matrix in ``dataset`` is whitened.

    The function mutates ``dataset["X"]`` in-place when whitening metadata is
    available.  It returns ``True`` when metadata was found and applied.
    """

    if not isinstance(dataset, (MutableMapping, dict)):
        return False

    X = dataset.get("X")  # type: ignore[assignment]
    if X is None:
        return False

    artifacts = dataset.get("__artifacts__")  # type: ignore[assignment]
    summary: Any | None = None
    if isinstance(artifacts, Mapping):
        summary = artifacts.get("mlcv_deeptica")
    if not isinstance(summary, (MutableMapping, dict)):
        return False

    whitened, applied = apply_whitening_from_metadata(
        np.asarray(X, dtype=np.float64), summary
    )
    if applied:
        dataset["X"] = whitened  # type: ignore[index]
    return applied


def prepare_msm_discretization(
    dataset: DatasetLike | Mapping[str, Any],
    *,
    cluster_mode: str = "kmeans",
    n_microstates: int = 150,
    lag_time: int = 1,
    frame_weights: Mapping[str, Any] | Any | None = None,
    min_out_count: int = 0,
    random_state: int | None = None,
    apply_whitening: bool = True,
) -> MSMDiscretizationResult:
    """Whiten CVs when possible and build MSM discretisation statistics."""

    if isinstance(dataset, (MutableMapping, dict)):
        if apply_whitening:
            ensure_msm_inputs_whitened(dataset)
        if frame_weights is None:
            candidate = dataset.get("frame_weights")
            if candidate is not None:
                frame_weights = candidate

    result = discretize_dataset(
        dataset,
        cluster_mode=cluster_mode,
        n_microstates=n_microstates,
        lag_time=lag_time,
        frame_weights=frame_weights,
        min_out_count=min_out_count,
        random_state=random_state,
    )

    if isinstance(dataset, MutableMapping):
        artifacts = dataset.setdefault("__artifacts__", {})  # type: ignore[assignment]
        if isinstance(artifacts, MutableMapping):
            artifacts["feature_stats"] = result.feature_stats  # type: ignore[index]
            artifacts["state_assignments"] = {
                split: {
                    "n_assigned": int(np.count_nonzero(mask)),
                    "total": int(mask.size),
                }
                for split, mask in result.assignment_masks.items()
            }
            artifacts["segment_lengths"] = result.segment_lengths  # type: ignore[index]
            artifacts["segment_strides"] = result.segment_strides  # type: ignore[index]
            artifacts["expected_pairs"] = result.expected_pairs  # type: ignore[index]
            artifacts["counted_pairs"] = result.counted_pairs  # type: ignore[index]
            if result.pruned_state_indices is not None:
                artifacts["pruned_state_indices"] = result.pruned_state_indices.tolist()  # type: ignore[index]
            if result.state_counts is not None:
                artifacts["state_counts_post_prune"] = result.state_counts.tolist()  # type: ignore[index]
            if result.state_counts_before_prune is not None:
                artifacts["state_counts_pre_prune"] = result.state_counts_before_prune.tolist()  # type: ignore[index]

    return result
