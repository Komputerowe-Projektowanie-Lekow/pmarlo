from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from pmarlo.transform.apply import (
    _compute_pairs_metadata,
    _prepare_learn_cv_arrays,
)


def test_prepare_learn_cv_arrays_filters_invalid_shards_alignment():
    X = np.arange(20, dtype=float).reshape(10, 2)
    dataset = {
        "X": X,
        "__shards__": [
            {"id": "bad", "start": 5, "stop": 5},
            {"id": "ok1", "start": 0, "stop": 6},
            {"id": "ok2", "start": 6, "stop": 12},
        ],
    }

    X_all, shards_meta, shard_ranges, X_list = _prepare_learn_cv_arrays(dataset)

    assert [entry["id"] for entry in shards_meta] == ["ok1", "ok2"]
    assert shard_ranges == [(0, 6), (6, 10)]
    assert len(X_list) == 2
    assert dataset["__shards__"] == shards_meta

    per_shard, total_pairs, warnings = _compute_pairs_metadata(
        1, shard_ranges, shards_meta, X_all.shape[0]
    )

    assert total_pairs == 8
    assert all(item["pairs"] > 0 for item in per_shard)
    assert warnings == []
