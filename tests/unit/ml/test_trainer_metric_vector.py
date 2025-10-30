"""Tests for the metric vector extraction helper."""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest


def _ensure_mdtraj_stub() -> None:
    """Provide a lightweight mdtraj stub to satisfy module imports during tests."""

    if "mdtraj" in sys.modules:
        return
    stub = types.ModuleType("mdtraj")
    stub.Trajectory = type("Trajectory", (), {})

    def _unavailable(*args: object, **kwargs: object) -> None:  # pragma: no cover - defensive
        raise RuntimeError("mdtraj functionality is not available in unit tests")

    stub.compute_phi = _unavailable
    stub.compute_psi = _unavailable
    stub.compute_rg = _unavailable
    sys.modules["mdtraj"] = stub


_ensure_mdtraj_stub()

from pmarlo.ml.deeptica.trainer import _metric_vector


def test_metric_vector_from_numpy_array() -> None:
    """A numpy array is converted to a list of floats."""

    metrics = {"values": np.array([1.0, 2.5, 3.75])}

    assert _metric_vector(metrics, "values") == [1.0, 2.5, 3.75]


def test_metric_vector_missing_key_raises_key_error() -> None:
    """Missing metrics raise an explicit KeyError."""

    with pytest.raises(KeyError, match="Metric 'missing' is missing"):
        _metric_vector({}, "missing")


def test_metric_vector_empty_sequence_raises_value_error() -> None:
    """Empty sequences are treated as invalid data."""

    metrics = {"values": []}

    with pytest.raises(ValueError, match="contains no values"):
        _metric_vector(metrics, "values")


def test_metric_vector_non_numeric_item_raises_value_error() -> None:
    """Non-numeric entries propagate a descriptive ValueError."""

    metrics = {"values": [1.0, "not-a-number", 2.0]}

    with pytest.raises(ValueError, match="non-numeric value"):
        _metric_vector(metrics, "values")
