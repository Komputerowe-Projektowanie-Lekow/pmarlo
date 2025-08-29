from __future__ import annotations

import numpy as np

from pmarlo.cv.pairs import make_training_pairs_from_shards, scaled_time_pairs


def test_scaled_time_pairs_uniform():
    L = 20
    tau = 5.0
    i, j = scaled_time_pairs(L, None, tau)
    assert i.dtype == np.int64 and j.dtype == np.int64
    assert i.size == L - int(round(tau))
    assert np.all(j - i == int(round(tau)))


def test_make_training_pairs_from_shards_uniform():
    X1 = np.zeros((10, 2))
    X2 = np.zeros((7, 2))
    tau = 3.0
    X_list, (t, tlag) = make_training_pairs_from_shards(
        [(X1, None, None, 300.0), (X2, None, None, 300.0)], tau_scaled=tau
    )
    assert len(X_list) == 2
    assert t.size == (10 - 3) + (7 - 3)
    assert np.all((tlag - t) == 3)
    assert t.min() >= 0 and tlag.max() < (10 + 7)


def test_scaled_time_pairs_with_bias_generates_pairs():
    L = 30
    # Create a region with higher weights -> faster scaled time
    logw = np.zeros(L)
    logw[10:20] = np.log(5.0)
    i, j = scaled_time_pairs(L, logw, tau_scaled=6.0)
    assert i.size > 0
    assert j.size == i.size
    assert np.all(j > i)


def test_make_training_pairs_from_shards_with_bias():
    L = 25
    X = np.random.default_rng(0).normal(size=(L, 3))
    bias = np.linspace(0.0, 1.0, L)
    X_list, (t, tlag) = make_training_pairs_from_shards(
        [(X, None, bias, 300.0)], tau_scaled=4.0
    )
    assert len(X_list) == 1 and X_list[0].shape == (L, 3)
    assert t.size > 0 and tlag.size == t.size
    assert np.all(tlag > t)
