from unittest.mock import patch

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


def test_kmeans_uses_int_n_init():
    Y = np.random.rand(10, 2)
    with patch("pmarlo.markov_state_model.clustering.KMeans") as mock_kmeans:
        instance = mock_kmeans.return_value
        instance.fit_predict.return_value = np.zeros(10, dtype=int)
        cluster_microstates(Y, method="kmeans", n_states=2)
        assert isinstance(mock_kmeans.call_args.kwargs["n_init"], int)


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
        mock_mb.return_value.fit_predict.return_value = np.zeros(10, dtype=int)
        cluster_microstates(Y, method="auto", n_states=2, minibatch_threshold=50)
        assert mock_mb.called


def test_auto_selection_sampling(monkeypatch):
    import pmarlo.markov_state_model.clustering as clustering

    Y = np.random.rand(50, 3)
    fit_sizes: list[int] = []

    class DummyKMeans:
        def __init__(self, n_clusters, random_state=None, n_init=10):
            self.n_clusters = n_clusters

        def fit_predict(self, data):
            fit_sizes.append(data.shape[0])
            return np.arange(data.shape[0]) % self.n_clusters

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
        def __init__(self, n_clusters, random_state=None, n_init=10):
            created.append(n_clusters)
            self.n_clusters = n_clusters

        def fit_predict(self, data):
            return np.arange(data.shape[0]) % self.n_clusters

    monkeypatch.setattr(clustering, "KMeans", DummyKMeans)

    result = clustering.cluster_microstates(
        Y,
        n_states="auto",
        auto_n_states_override=7,
    )

    assert result.n_states == 7
    assert result.rationale == "auto-override=7"
    assert created == [7]
