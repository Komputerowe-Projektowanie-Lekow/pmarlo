"""Tests for MSM discretisation feature schema validation."""

from __future__ import annotations

import numpy as np
import pytest

from pmarlo.analysis.discretize import FeatureMismatchError, discretize_dataset


def _make_split(data: np.ndarray, names: list[str]) -> dict[str, object]:
    return {
        "X": data,
        "feature_schema": {"names": names, "n_features": len(names)},
    }


def test_discretize_dataset_raises_on_column_reordering() -> None:
    base = np.array(
        [
            [0.1, 1.2, -3.4],
            [0.5, 1.0, -3.2],
            [0.3, 1.5, -3.0],
            [0.7, 1.1, -3.6],
        ],
        dtype=np.float64,
    )

    train_split = _make_split(base, ["a", "b", "c"])
    reordered = base[:, [1, 0, 2]]
    val_split = _make_split(reordered, ["b", "a", "c"])

    dataset = {"splits": {"train": train_split, "val": val_split}}

    with pytest.raises(FeatureMismatchError) as excinfo:
        discretize_dataset(dataset, n_microstates=3, lag_time=1, random_state=1)

    message = str(excinfo.value)
    assert "Feature schema mismatch" in message
    assert "order mismatch" in message or "position" in message


def test_discretize_dataset_raises_when_column_missing() -> None:
    base = np.array(
        [
            [0.1, 1.2, -3.4],
            [0.5, 1.0, -3.2],
            [0.3, 1.5, -3.0],
            [0.7, 1.1, -3.6],
        ],
        dtype=np.float64,
    )

    train_split = _make_split(base, ["a", "b", "c"])
    trimmed = base[:, :2]
    test_split = _make_split(trimmed, ["a", "b"])

    dataset = {"splits": {"train": train_split, "test": test_split}}

    with pytest.raises(FeatureMismatchError) as excinfo:
        discretize_dataset(dataset, n_microstates=3, lag_time=1, random_state=2)

    message = str(excinfo.value)
    assert "n_features mismatch" in message
