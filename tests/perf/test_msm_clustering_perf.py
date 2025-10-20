from __future__ import annotations

"""Performance benchmarks for MSM clustering operations.

These benchmarks measure clustering performance which is critical for:
- Microstate clustering (KMeans vs MiniBatchKMeans)
- Automatic state selection (silhouette scoring)
- Large dataset handling

Run with: pytest -m benchmark tests/perf/test_msm_clustering_perf.py
"""

import os

import numpy as np
import pytest

pytestmark = [pytest.mark.perf, pytest.mark.benchmark, pytest.mark.msm]

# Optional plugin
pytest.importorskip("pytest_benchmark", reason="pytest-benchmark not installed")

if not os.getenv("PMARLO_RUN_PERF"):
    pytest.skip(
        "perf tests disabled; set PMARLO_RUN_PERF=1 to run", allow_module_level=True
    )


def _generate_synthetic_features(
    n_samples: int, n_features: int, n_clusters: int = 5, seed: int = 42
) -> np.ndarray:
    """Generate synthetic feature data with known cluster structure."""
    rng = np.random.default_rng(seed)

    # Create cluster centers
    centers = rng.standard_normal((n_clusters, n_features)) * 10.0

    # Generate samples around centers
    samples_per_cluster = n_samples // n_clusters
    features = []

    for i in range(n_clusters):
        cluster_samples = (
            centers[i] + rng.standard_normal((samples_per_cluster, n_features)) * 2.0
        )
        features.append(cluster_samples)

    # Handle remaining samples
    remaining = n_samples - (samples_per_cluster * n_clusters)
    if remaining > 0:
        extra = (
            centers[0] + rng.standard_normal((remaining, n_features)) * 2.0
        )
        features.append(extra)

    return np.vstack(features).astype(np.float32)


@pytest.fixture
def small_dataset():
    """Small dataset (1K samples) for quick clustering."""
    return _generate_synthetic_features(1000, 10, n_clusters=5)


@pytest.fixture
def medium_dataset():
    """Medium dataset (10K samples) for KMeans benchmark."""
    return _generate_synthetic_features(10_000, 10, n_clusters=5)


@pytest.fixture
def large_dataset():
    """Large dataset (50K samples) for MiniBatchKMeans benchmark."""
    return _generate_synthetic_features(50_000, 10, n_clusters=5)


def test_kmeans_small_dataset(benchmark, small_dataset):
    """Benchmark KMeans clustering on small dataset (baseline)."""
    from pmarlo.markov_state_model.clustering import cluster_microstates

    def _cluster():
        return cluster_microstates(
            small_dataset, n_states=5, random_state=42
        )

    result = benchmark(_cluster)
    assert result.n_states == 5
    assert len(result.labels) == len(small_dataset)


def test_kmeans_medium_dataset(benchmark, medium_dataset):
    """Benchmark KMeans clustering on medium dataset (10K samples)."""
    from pmarlo.markov_state_model.clustering import cluster_microstates

    def _cluster():
        return cluster_microstates(
            medium_dataset, n_states=5, random_state=42
        )

    result = benchmark(_cluster)
    assert result.n_states == 5
    assert len(result.labels) == len(medium_dataset)


def test_minibatch_kmeans_large_dataset(benchmark, large_dataset):
    """Benchmark MiniBatchKMeans on large dataset (50K samples).

    This tests automatic switching to MiniBatchKMeans for large datasets.
    """
    from pmarlo.markov_state_model.clustering import cluster_microstates

    def _cluster():
        return cluster_microstates(
            large_dataset, n_states=5, random_state=42
        )

    result = benchmark(_cluster)
    assert result.n_states == 5
    assert len(result.labels) == len(large_dataset)


def test_auto_clustering_small(benchmark, small_dataset):
    """Benchmark automatic state selection on small dataset (silhouette scoring)."""
    from pmarlo.markov_state_model.clustering import cluster_microstates

    def _cluster():
        return cluster_microstates(
            small_dataset, n_states="auto", random_state=42
        )

    result = benchmark(_cluster)
    assert result.n_states > 0
    assert result.rationale is not None
    assert "silhouette" in result.rationale.lower()


def test_auto_clustering_with_range(benchmark, medium_dataset):
    """Benchmark automatic clustering with custom range (optimization overhead)."""
    from pmarlo.markov_state_model.clustering import cluster_microstates

    def _cluster():
        # Auto clustering with specific range
        return cluster_microstates(
            medium_dataset,
            n_states="auto",
            auto_min=3,
            auto_max=10,
            random_state=42,
        )

    result = benchmark(_cluster)
    assert 3 <= result.n_states <= 10
    assert result.rationale is not None


def test_high_dimensional_clustering(benchmark):
    """Benchmark clustering on high-dimensional data (DeepTICA output simulation)."""
    # Simulate DeepTICA output: fewer samples, more dimensions
    features = _generate_synthetic_features(5000, 50, n_clusters=8)

    from pmarlo.markov_state_model.clustering import cluster_microstates

    def _cluster():
        return cluster_microstates(features, n_states=8, random_state=42)

    result = benchmark(_cluster)
    assert result.n_states == 8
    assert len(result.labels) == len(features)


def test_clustering_with_centers(benchmark, medium_dataset):
    """Benchmark clustering and center computation (used for visualization)."""
    from pmarlo.markov_state_model.clustering import cluster_microstates

    def _cluster():
        result = cluster_microstates(
            medium_dataset, n_states=5, random_state=42
        )
        # Centers should be computed by default
        return result

    result = benchmark(_cluster)
    assert result.centers is not None
    assert result.centers.shape == (5, medium_dataset.shape[1])


def test_repeated_clustering_stability(benchmark, small_dataset):
    """Benchmark repeated clustering to measure stability overhead."""
    from pmarlo.markov_state_model.clustering import cluster_microstates

    def _cluster_multiple():
        results = []
        for _ in range(5):
            result = cluster_microstates(
                small_dataset, n_states=5, random_state=42
            )
            results.append(result)
        return results

    results = benchmark(_cluster_multiple)
    assert len(results) == 5

    # Verify stability with same seed
    labels_0 = results[0].labels
    for result in results[1:]:
        np.testing.assert_array_equal(result.labels, labels_0)


def test_many_states_clustering(benchmark, medium_dataset):
    """Benchmark clustering with many states (stress test)."""
    from pmarlo.markov_state_model.clustering import cluster_microstates

    def _cluster():
        return cluster_microstates(
            medium_dataset, n_states=50, random_state=42
        )

    result = benchmark(_cluster)
    assert result.n_states == 50
    assert len(np.unique(result.labels)) == 50


def test_silhouette_scoring_overhead(benchmark, small_dataset):
    """Benchmark silhouette score computation (used in auto clustering)."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    # Pre-cluster to measure just scoring overhead
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    labels = kmeans.fit_predict(small_dataset)

    def _compute_score():
        return silhouette_score(small_dataset, labels)

    score = benchmark(_compute_score)
    assert -1.0 <= score <= 1.0

