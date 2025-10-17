from __future__ import annotations

import numpy as np
import pytest

from pmarlo.pairs.core import PairInfo, build_pair_info


def test_build_pair_info_uniform_pairs():
    blocks = [
        np.zeros((10, 1), dtype=np.float32),
        np.zeros((7, 1), dtype=np.float32),
    ]
    info = build_pair_info(blocks, [3])
    assert isinstance(info, PairInfo)
    assert info.idx_t.shape == info.idx_tau.shape
    assert info.idx_t.size == (10 - 3) + (7 - 3)
    assert info.weights.shape == (info.idx_t.size,)
    assert info.diagnostics["pairs_by_shard"] == [7, 4]


def test_build_pair_info_rejects_empty_schedule():
    blocks = [np.zeros((4, 1), dtype=np.float32)]
    with pytest.raises(ValueError):
        build_pair_info(blocks, [])


def test_build_pair_info_respects_provided_pairs():
    blocks = [np.zeros((6, 1), dtype=np.float32)]
    manual_pairs = (np.array([0, 1, 2], dtype=np.int64), np.array([2, 3, 4], dtype=np.int64))
    info = build_pair_info(blocks, [2], pairs=manual_pairs)
    assert np.array_equal(info.idx_t, manual_pairs[0])
    assert np.array_equal(info.idx_tau, manual_pairs[1])


def test_build_pair_info_rejects_invalid_pairs():
    blocks = [np.zeros((6, 1), dtype=np.float32)]
    manual_pairs = (np.array([0, 1, 2], dtype=np.int64), np.array([0, 2, 3], dtype=np.int64))
    with pytest.raises(ValueError):
        build_pair_info(blocks, [2], pairs=manual_pairs)
