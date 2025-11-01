from __future__ import annotations

"""Performance benchmarks for learned CV inference and storage."""

import os
from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np
import pytest

from pmarlo.transform.apply import (
    _store_transformed_features,
    _transform_features_with_model,
)

pytestmark = [pytest.mark.perf, pytest.mark.benchmark, pytest.mark.transform]

pytest.importorskip("pytest_benchmark", reason="pytest-benchmark not installed")

if not os.getenv("PMARLO_RUN_PERF"):
    pytest.skip(
        "perf tests disabled; set PMARLO_RUN_PERF=1 to run", allow_module_level=True
    )


class _Transformer(Protocol):
    def transform(self, X: np.ndarray) -> np.ndarray: ...


@dataclass
class _LinearModel:
    weights: np.ndarray

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X @ self.weights


@dataclass
class _NonLinearModel:
    weights: np.ndarray

    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.tanh(X @ self.weights)


@pytest.fixture(scope="module")
def feature_matrix() -> np.ndarray:
    rng = np.random.default_rng(2024)
    return rng.normal(size=(16_384, 18)).astype(np.float64)


@pytest.fixture(scope="module")
def linear_model(feature_matrix: np.ndarray) -> _LinearModel:
    rng = np.random.default_rng(17)
    weights = rng.normal(size=(feature_matrix.shape[1], 6)).astype(np.float64)
    return _LinearModel(weights=weights)


@pytest.fixture(scope="module")
def nonlinear_model(feature_matrix: np.ndarray) -> _NonLinearModel:
    rng = np.random.default_rng(99)
    weights = rng.normal(size=(feature_matrix.shape[1], 8)).astype(np.float64)
    return _NonLinearModel(weights=weights)


def _benchmark_transform(
    benchmark: Callable[[Callable[[], np.ndarray]], np.ndarray],
    model: _Transformer,
    features: np.ndarray,
) -> np.ndarray:
    def _run() -> np.ndarray:
        return _transform_features_with_model(model, features)

    return benchmark(_run)


def test_linear_model_inference(benchmark, feature_matrix, linear_model):
    """Benchmark linear inference pass for trained CV models."""

    transformed = _benchmark_transform(benchmark, linear_model, feature_matrix)
    assert transformed.shape == (feature_matrix.shape[0], linear_model.weights.shape[1])
    expected = feature_matrix @ linear_model.weights
    np.testing.assert_allclose(transformed, expected, rtol=1e-7, atol=1e-9)


def test_nonlinear_model_inference(benchmark, feature_matrix, nonlinear_model):
    """Benchmark nonlinear inference used for DeepTICA activation stacks."""

    transformed = _benchmark_transform(benchmark, nonlinear_model, feature_matrix)
    assert transformed.shape == (
        feature_matrix.shape[0],
        nonlinear_model.weights.shape[1],
    )
    assert np.all(np.abs(transformed) <= 1.0)


def test_transform_storage_round_trip(benchmark, feature_matrix, nonlinear_model):
    """Benchmark storing transformed CVs alongside metadata updates."""

    def _apply_and_store() -> dict[str, object]:
        dataset = {"X": feature_matrix.copy()}
        transformed = _transform_features_with_model(nonlinear_model, dataset["X"])
        _store_transformed_features(dataset, transformed)
        return dataset

    dataset = benchmark(_apply_and_store)
    assert dataset["X"].shape == (
        feature_matrix.shape[0],
        nonlinear_model.weights.shape[1],
    )
    assert dataset["cv_names"] == tuple(
        f"DeepTICA_{i+1}" for i in range(nonlinear_model.weights.shape[1])
    )
    assert dataset["periodic"] == tuple(
        False for _ in range(nonlinear_model.weights.shape[1])
    )
