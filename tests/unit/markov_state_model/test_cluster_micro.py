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


def test_n_init_restarts_select_best(monkeypatch):
    import pmarlo.markov_state_model.clustering as clustering

    Y = np.vstack([np.zeros((5, 2)), np.ones((5, 2))])
    seeds: list[Any] = []

    class DummyModel:
        def __init__(self, seed):
            self.seed = seed

        def transform(self, data):
            if self.seed in (None, 42):
                return np.zeros(data.shape[0], dtype=int)
            labels = np.zeros(data.shape[0], dtype=int)
            labels[data.shape[0] // 2 :] = 1
            return labels

    def fake_create(method, n_states, random_state, **kwargs):
        class DummyEstimator:
            def __init__(self, seed):
                self.seed = seed

            def fit_fetch(self, data):
                seeds.append(self.seed)
                return DummyModel(self.seed)

        return DummyEstimator(random_state)

    monkeypatch.setattr(clustering, "_create_clustering_estimator", fake_create)

    result = clustering.cluster_microstates(
        Y,
        method="kmeans",
        n_states=2,
        random_state=42,
        n_init=3,
    )

    assert len(seeds) == 3
    assert len(set(seeds)) == len(seeds)
    assert np.unique(result.labels).size == 2
    assert set(seed for seed in seeds if seed is not None) >= {42}


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
        result_mock = MagicMock()
        result_mock.transform.return_value = np.zeros(10, dtype=int)
        mock_mb.return_value.fit_fetch.return_value = result_mock
        cluster_microstates(Y, method="auto", n_states=2, minibatch_threshold=50)
        assert mock_mb.called


def test_auto_selection_sampling(monkeypatch):
    import pmarlo.markov_state_model.clustering as clustering

    Y = np.random.rand(50, 3)
    fit_sizes: list[int] = []

    class DummyResult:
        def __init__(self, n_clusters):
            self.n_clusters = n_clusters

        def transform(self, data):
            return np.arange(data.shape[0]) % self.n_clusters

    class DummyKMeans:
        def __init__(self, n_clusters, random_state=None, n_init=10):
            self.n_clusters = n_clusters

        def fit_fetch(self, data):
            fit_sizes.append(data.shape[0])
            return DummyResult(self.n_clusters)

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

    class DummyResult:
        def __init__(self, n_clusters):
            self.n_clusters = n_clusters

        def transform(self, data):
            return np.arange(data.shape[0]) % self.n_clusters

    class DummyKMeans:
        def __init__(self, n_clusters, random_state=None, n_init=10):
            created.append(n_clusters)
            self.n_clusters = n_clusters

        def fit_fetch(self, data):
            return DummyResult(self.n_clusters)

    monkeypatch.setattr(clustering, "KMeans", DummyKMeans)

    result = clustering.cluster_microstates(
        Y,
        n_states="auto",
        auto_n_states_override=7,
    )

    assert result.n_states == 7
    assert result.rationale == "auto-override=7"
    assert created == [7]
