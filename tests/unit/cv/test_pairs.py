from __future__ import annotations

import numpy as np

from pmarlo.features.pairs import (
    lagged_time_pairs,
    make_training_pairs_from_trajectory,
)


def test_lagged_time_pairs_uniform():
    length = 20
    lag = 5
    i, j = lagged_time_pairs(length, lag)
    assert i.dtype == np.int64 and j.dtype == np.int64
    assert i.size == length - lag
    assert np.all(j - i == lag)


def test_make_training_pairs_from_trajectory_uniform():
    X = np.zeros((10, 2))
    X_list, (t, tlag) = make_training_pairs_from_trajectory(X, lag=3)
    assert len(X_list) == 1
    assert t.size == 10 - 3
    assert np.all((tlag - t) == 3)
    assert t.min() >= 0 and tlag.max() < 10
