from __future__ import annotations

"""Performance benchmarks for discretization assignments.

These tests ensure that data points are assigned to the nearest learned
cluster centers when using the discretization utilities in
``pmarlo.analysis.discretize``.

Run with: pytest -m benchmark tests/perf/test_discretize_assignment_perf.py
"""

import os
from typing import Dict, Tuple

import numpy as np
import pytest

from pmarlo.analysis.discretize import _KMeansDiscretizer, discretize_dataset

pytestmark = [pytest.mark.perf, pytest.mark.benchmark, pytest.mark.analysis]

pytest.importorskip("pytest_benchmark", reason="pytest-benchmark not installed")

if not os.getenv("PMARLO_RUN_PERF"):
    pytest.skip(
        "perf tests disabled; set PMARLO_RUN_PERF=1 to run", allow_module_level=True
    )


def _generate_gaussian_clusters(
    *,
    n_clusters: int,
    samples_per_cluster: int,
    n_features: int,
    seed: int = 123,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate clustered data with well-separated Gaussian blobs."""

    rng = np.random.default_rng(seed)
    centers = rng.normal(loc=0.0, scale=5.0, size=(n_clusters, n_features))
    clusters = []
    for center in centers:
        cluster_samples = center + rng.normal(
            scale=0.2, size=(samples_per_cluster, n_features)
        )
        clusters.append(cluster_samples)
    data = np.vstack(clusters).astype(np.float64)
    return data, centers.astype(np.float64)


def _feature_schema(n_features: int) -> Dict[str, object]:
    return {
        "names": [f"feature_{idx}" for idx in range(n_features)],
        "n_features": n_features,
    }


@pytest.fixture(scope="module")
def fitted_kmeans_discretizer() -> (
    Tuple[_KMeansDiscretizer, np.ndarray, Dict[str, object]]
):
    """Fit a KMeans discretizer on synthetic clustered data."""

    data, _ = _generate_gaussian_clusters(
        n_clusters=4, samples_per_cluster=250, n_features=6, seed=21
    )
    schema = _feature_schema(data.shape[1])
    discretizer = _KMeansDiscretizer(n_states=4, random_state=7)
    discretizer.fit(data, schema)
    return discretizer, data, schema


def test_kmeans_assignment_matches_nearest_centers(
    benchmark, fitted_kmeans_discretizer
):
    """Assignments should match the nearest learned centers for training points."""

    discretizer, data, _ = fitted_kmeans_discretizer

    def _assign() -> np.ndarray:
        return discretizer.transform(data)

    labels = benchmark(_assign)
    centers = discretizer.centers
    assert centers is not None

    distances = np.linalg.norm(data[:, None, :] - centers[None, :, :], axis=2)
    expected = np.argmin(distances, axis=1).astype(np.int32)
    np.testing.assert_array_equal(labels, expected)


def test_kmeans_assignment_new_points(benchmark, fitted_kmeans_discretizer):
    """New points near centers should be assigned to the nearest centroid."""

    discretizer, data, schema = fitted_kmeans_discretizer
    centers = discretizer.centers
    assert centers is not None

    rng = np.random.default_rng(99)
    points = []
    for center in centers:
        points.append(center + rng.normal(scale=0.1, size=(150, centers.shape[1])))
    new_points = np.vstack(points).astype(np.float64)

    def _assign_new() -> np.ndarray:
        return discretizer.transform(new_points, feature_schema=schema)

    labels = benchmark(_assign_new)
    distances = np.linalg.norm(new_points[:, None, :] - centers[None, :, :], axis=2)
    expected = np.argmin(distances, axis=1).astype(np.int32)
    np.testing.assert_array_equal(labels, expected)


def test_discretize_dataset_assignments_follow_centers(benchmark):
    """discretize_dataset should align split assignments with nearest centers."""

    data, _ = _generate_gaussian_clusters(
        n_clusters=5, samples_per_cluster=220, n_features=5, seed=52
    )
    schema = _feature_schema(data.shape[1])
    split_index = int(len(data) * 0.6)
    train_data = data[:split_index]
    valid_data = data[split_index:]

    dataset = {
        "train": {"X": train_data, "feature_schema": schema},
        "validation": {"X": valid_data, "feature_schema": schema},
    }

    def _run_discretize() -> object:
        return discretize_dataset(
            dataset,
            cluster_mode="kmeans",
            n_microstates=5,
            random_state=11,
        )

    result = benchmark(_run_discretize)
    centers = result.centers
    assert centers is not None and centers.shape[0] == 5

    validation_labels = result.assignments["validation"]
    validation_mask = result.assignment_masks["validation"]
    assert validation_labels.shape[0] == valid_data.shape[0]
    assert validation_mask.shape[0] == valid_data.shape[0]

    distances = np.linalg.norm(valid_data[:, None, :] - centers[None, :, :], axis=2)
    expected = np.argmin(distances, axis=1).astype(np.int32)
    np.testing.assert_array_equal(
        validation_labels[validation_mask], expected[validation_mask]
    )
