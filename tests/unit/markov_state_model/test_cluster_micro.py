from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

pytest.importorskip("sklearn")

from pmarlo.markov_state_model.clustering import cluster_microstates


def test_returns_empty_for_no_samples():
    Y = np.empty((0, 2))
    result = cluster_microstates(Y, n_states=2)
    assert result.labels.size == 0
    assert result.n_states == 0


def test_raises_for_no_features():
    Y = np.empty((3, 0))
    with pytest.raises(ValueError, match="at least one feature"):
        cluster_microstates(Y)


def test_kmeans_n_init_forwarding(monkeypatch):
    import pmarlo.markov_state_model.clustering as clustering

    Y = np.vstack([np.zeros((4, 2)), np.ones((4, 2))])
    seen_n_init: list[Any] = []

    class DummyKMeans:
        def __init__(self, n_clusters, random_state=None, **kwargs):
            seen_n_init.append(kwargs.get("n_init"))
            self.n_clusters = n_clusters

        def fit(self, data):
            self.labels_ = np.arange(data.shape[0]) % self.n_clusters
            self.cluster_centers_ = np.zeros((self.n_clusters, data.shape[1]))
            return self

    monkeypatch.setattr(clustering, "KMeans", DummyKMeans)

    clustering.cluster_microstates(Y, method="kmeans", n_states=2)
    clustering.cluster_microstates(Y, method="kmeans", n_states=2, n_init=5)

    assert seen_n_init[0] == 1  # default
    assert seen_n_init[1] == 5


def test_auto_and_fixed_states():
    from sklearn.datasets import make_blobs

    X, _ = make_blobs(n_samples=200, centers=8, n_features=2, random_state=0)
    fixed = cluster_microstates(X, n_states=8, random_state=0)
    assert len(np.unique(fixed.labels)) == 8

    auto = cluster_microstates(X, n_states="auto", random_state=0)
    assert 4 <= auto.n_states <= 20
    assert auto.rationale is not None


def test_auto_switches_to_minibatch():
    Y = np.random.rand(10, 10)
    with patch("pmarlo.markov_state_model.clustering.MiniBatchKMeans") as mock_mb:
        estimator = MagicMock()
        estimator.fit.return_value = estimator
        estimator.labels_ = np.zeros(10, dtype=int)
        estimator.cluster_centers_ = np.zeros((2, Y.shape[1]))
        mock_mb.return_value = estimator
        cluster_microstates(Y, method="auto", n_states=2, minibatch_threshold=50)
        assert mock_mb.called


def test_auto_selection_sampling(monkeypatch):
    import pmarlo.markov_state_model.clustering as clustering

    Y = np.random.rand(50, 3)
    fit_sizes: list[int] = []

    class DummyKMeans:
        def __init__(self, n_clusters, random_state=None, n_init=10, **_):
            self.n_clusters = n_clusters

        def fit(self, data):
            fit_sizes.append(data.shape[0])
            self.labels_ = np.arange(data.shape[0]) % self.n_clusters
            self.cluster_centers_ = np.zeros((self.n_clusters, data.shape[1]))
            return self

    def fake_silhouette(data, labels):
        assert data.shape[0] == labels.shape[0] == 10
        return 0.1

    monkeypatch.setattr(clustering, "KMeans", DummyKMeans)
    monkeypatch.setattr(clustering, "silhouette_score", fake_silhouette)

    result = clustering.cluster_microstates(
        Y,
        n_states="auto",
        random_state=0,
        silhouette_sample_size=10,
    )

    assert result.rationale is not None and "sample=10" in result.rationale
    assert fit_sizes.count(10) == 17  # 4 through 20 inclusive
    assert fit_sizes.count(50) == 1  # final clustering uses full dataset


def test_auto_selection_override(monkeypatch):
    import pmarlo.markov_state_model.clustering as clustering

    Y = np.random.rand(40, 2)
    created: list[int] = []

    class DummyKMeans:
        def __init__(self, n_clusters, random_state=None, n_init=10, **_):
            created.append(n_clusters)
            self.n_clusters = n_clusters

        def fit(self, data):
            self.labels_ = np.arange(data.shape[0]) % self.n_clusters
            self.cluster_centers_ = np.zeros((self.n_clusters, data.shape[1]))
            return self

    monkeypatch.setattr(clustering, "KMeans", DummyKMeans)

    result = clustering.cluster_microstates(
        Y,
        n_states="auto",
        auto_n_states_override=7,
    )

    assert result.n_states == 7
    assert result.rationale == "auto-override=7"
    assert created == [7]


def test_dbscan_clustering_produces_states():
    from sklearn.datasets import make_blobs

    X, _ = make_blobs(
        n_samples=200,
        centers=[[0, 0], [5, 5]],
        cluster_std=0.3,
        random_state=42,
    )
    result = cluster_microstates(
        X,
        method="dbscan",
        n_states="auto",
        eps=0.8,
        min_samples=5,
    )
    assert result.n_states >= 2
    assert np.count_nonzero(result.labels >= 0) == X.shape[0]
    assert result.centers is not None
    assert result.centers.shape[0] == result.n_states


def test_dbscan_auto_eps_selection_without_manual_parameters():
    from sklearn.datasets import make_blobs

    X, _ = make_blobs(
        n_samples=300,
        centers=[[0, 0], [4, 4], [-4, 4]],
        cluster_std=0.25,
        random_state=7,
    )
    result = cluster_microstates(X, method="dbscan", n_states="auto")
    assert result.n_states >= 3


def test_dbscan_respects_user_supplied_eps(monkeypatch):
    import pmarlo.markov_state_model.clustering as clustering

    called = False

    def fake_estimate(*args, **kwargs):
        nonlocal called
        called = True
        return 0.1, {"sample_size": 10, "neighbor_rank": 5, "percentile": 90.0}

    monkeypatch.setattr(clustering, "estimate_dbscan_eps", fake_estimate)
    data = np.vstack([np.zeros((20, 2)), np.ones((20, 2))])

    result = cluster_microstates(
        data,
        method="dbscan",
        n_states="auto",
        eps=0.5,
        min_samples=3,
    )

    assert result.n_states >= 1
    assert called is False


def test_dbscan_requires_auto_state_count():
    data = np.random.rand(20, 2)
    with pytest.raises(ValueError, match="must be 'auto'"):
        cluster_microstates(data, method="dbscan", n_states=3)
