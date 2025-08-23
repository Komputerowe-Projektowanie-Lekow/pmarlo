import numpy as np
import pytest
from unittest.mock import patch

from pmarlo.cluster.micro import cluster_microstates


def test_returns_empty_for_no_samples():
    Y = np.empty((0, 2))
    labels = cluster_microstates(Y, n_clusters=2)
    assert labels.size == 0
    assert labels.dtype == int


def test_raises_for_no_features():
    Y = np.empty((3, 0))
    with pytest.raises(ValueError, match="at least one feature"):
        cluster_microstates(Y)


def test_kmeans_uses_int_n_init():
    Y = np.random.rand(10, 2)
    with patch("pmarlo.cluster.micro.KMeans") as mock_kmeans:
        instance = mock_kmeans.return_value
        instance.fit_predict.return_value = np.zeros(10, dtype=int)
        cluster_microstates(Y, method="kmeans", n_clusters=2)
        assert isinstance(mock_kmeans.call_args.kwargs["n_init"], int)
