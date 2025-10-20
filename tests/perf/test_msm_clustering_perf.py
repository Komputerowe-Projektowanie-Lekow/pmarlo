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

from pmarlo.markov_state_model._clustering import ClusteringMixin

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
    """Small dataset (800 samples) for quick clustering."""
    return _generate_synthetic_features(800, 10, n_clusters=5)


@pytest.fixture
def medium_dataset():
    """Medium dataset (4K samples) for KMeans benchmark."""
    return _generate_synthetic_features(4_000, 10, n_clusters=5)


@pytest.fixture
def large_dataset():
    """Large dataset (12K samples) for MiniBatchKMeans benchmark."""
    return _generate_synthetic_features(12_000, 10, n_clusters=5)


def _generate_tica_projection(
    n_trajectories: int = 4,
    trajectory_length: int = 800,
    feature_dim: int = 6,
    lag: int = 5,
    n_components: int = 3,
) -> tuple[np.ndarray, list[int]]:
    """Generate synthetic trajectories and return their TICA projection."""

    deeptime = pytest.importorskip(
        "deeptime",
        reason="deeptime is required for TICA-projected clustering benchmarks",
    )
    from deeptime.decomposition import TICA

    rng = np.random.default_rng(314159)
    raw_trajectories: list[np.ndarray] = []

    for _ in range(n_trajectories):
        base = rng.standard_normal((trajectory_length, feature_dim))
        time_axis = np.linspace(0.0, 2.0 * np.pi, trajectory_length)
        base[:, 0] += np.sin(time_axis) * 0.5
        base[:, 1] += np.cos(time_axis) * 0.5
        base[:, 2] += 0.1 * time_axis
        raw_trajectories.append(base.astype(np.float32))

    tica = TICA(lagtime=int(max(1, lag)), dim=int(max(1, n_components)))
    model = tica.fit(raw_trajectories).fetch_model()
    projected = model.transform(raw_trajectories)

    concatenated = np.concatenate(projected, axis=0).astype(np.float32)
    lengths = [traj.shape[0] for traj in projected]
    return concatenated, lengths


@pytest.fixture
def tica_projected_dataset():
    """TICA-projected dataset used for clustering mixin benchmarks."""

    return _generate_tica_projection()


class _MockTrajectory:
    """Minimal trajectory stub providing ``n_frames`` attribute for the mixin."""

    def __init__(self, n_frames: int) -> None:
        self.n_frames = n_frames


class _ClusteringHarness(ClusteringMixin):
    """Concrete harness to exercise :class:`ClusteringMixin` in benchmarks."""

    def __init__(self, features: np.ndarray, lengths: list[int], seed: int = 11) -> None:
        self.features = features
        self.random_state = seed
        self.trajectories = [_MockTrajectory(length) for length in lengths]
        self.dtrajs: list[np.ndarray] = []
        self.cluster_centers = None
        self.n_states = 0


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
    """Benchmark KMeans clustering on medium dataset (4K samples)."""
    from pmarlo.markov_state_model.clustering import cluster_microstates

    def _cluster():
        return cluster_microstates(
            medium_dataset, n_states=5, random_state=42
        )

    result = benchmark(_cluster)
    assert result.n_states == 5
    assert len(result.labels) == len(medium_dataset)


def test_minibatch_kmeans_large_dataset(benchmark, large_dataset):
    """Benchmark MiniBatchKMeans on large dataset (12K samples).

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


def test_auto_clustering_with_override(benchmark, medium_dataset):
    """Benchmark automatic clustering when overriding silhouette selection."""
    from pmarlo.markov_state_model.clustering import cluster_microstates

    override_states = 6

    def _cluster():
        return cluster_microstates(
            medium_dataset,
            n_states="auto",
            random_state=42,
            auto_n_states_override=override_states,
        )

    result = benchmark(_cluster)
    assert result.n_states == override_states
    assert result.rationale is not None
    assert "override" in result.rationale.lower()


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


def test_clustering_mixin_kmeans_on_tica_projection(
    benchmark, tica_projected_dataset
):
    """Benchmark full mixin-based KMeans clustering on TICA-projected data."""

    features, lengths = tica_projected_dataset

    def _cluster_with_mixin():
        harness = _ClusteringHarness(features.copy(), lengths, seed=23)
        harness.cluster_features(n_states=8, algorithm="kmeans", random_state=23)
        return harness

    result = benchmark(_cluster_with_mixin)

    assert isinstance(result, _ClusteringHarness)
    assert result.n_states == 8
    assert result.cluster_centers is not None
    assert result.cluster_centers.shape == (8, features.shape[1])
    assert len(result.dtrajs) == len(lengths)
    assert sum(len(traj) for traj in result.dtrajs) == features.shape[0]
    for traj, expected_length in zip(result.dtrajs, lengths):
        assert traj.shape[0] == expected_length


def test_regspace_clustering_on_tica_projection(
    benchmark, tica_projected_dataset
):
    """Benchmark RegularSpace clustering on TICA features."""

    pytest.importorskip(
        "deeptime", reason="deeptime is required for Regspace clustering benchmark"
    )
    from deeptime.clustering import RegularSpace

    features, _ = tica_projected_dataset

    def _cluster_regspace():
        estimator = RegularSpace(dmin=0.35, max_centers=60)
        model = estimator.fit(features).fetch_model()
        assignments = model.transform(features)
        return assignments, model.cluster_centers

    labels, centers = benchmark(_cluster_regspace)

    assert labels.shape[0] == features.shape[0]
    assert centers.shape[1] == features.shape[1]
    assert centers.shape[0] >= 2
    assert centers.shape[0] <= 60
    assert len(np.unique(labels)) == centers.shape[0]

