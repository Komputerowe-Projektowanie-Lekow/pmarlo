from __future__ import annotations

import numpy as np
import pytest

from pmarlo.shards.pair_builder import PairBuilder
from pmarlo.shards.schema import FeatureSpec, Shard, ShardMeta


def make_shard(n_frames: int = 6) -> Shard:
    meta = ShardMeta(
        schema_version="1.0",
        shard_id="T300K_seg0002_rep000",
        temperature_K=300.0,
        beta=1.0,
        replica_id=0,
        segment_id=2,
        exchange_window_id=0,
        n_frames=n_frames,
        dt_ps=0.002,
        feature_spec=FeatureSpec(name="demo", scaler="identity", columns=("f1", "f2")),
        provenance={},
    )
    return Shard(
        meta=meta,
        X=np.zeros((n_frames, 2), dtype=np.float32),
        t_index=np.arange(n_frames, dtype=np.int64),
        dt_ps=0.002,
    )


def test_pair_builder_requires_positive_tau():
    with pytest.raises(ValueError):
        PairBuilder(0)


def test_pairs_do_not_cross_shard_boundary():
    shard = make_shard(n_frames=5)
    builder = PairBuilder(tau_steps=2)
    pairs = builder.make_pairs(shard)
    assert pairs.tolist() == [[0, 2], [1, 3], [2, 4]]
    assert np.all(pairs[:, 1] - pairs[:, 0] == 2)


def test_pair_builder_updates_tau_dynamically():
    shard = make_shard(n_frames=6)
    builder = PairBuilder(tau_steps=1)
    builder.set_tau(3)
    pairs = builder.make_pairs(shard)
    assert builder.tau == 3
    assert pairs.tolist() == [[0, 3], [1, 4], [2, 5]]
