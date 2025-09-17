from __future__ import annotations

import numpy as np

from pmarlo.features.samplers import BalancedTempSampler
from pmarlo.shards.pair_builder import PairBuilder
from pmarlo.shards.schema import FeatureSpec, Shard, ShardMeta


def make_shard(temperature: float, n_frames: int = 10) -> Shard:
    feature_spec = FeatureSpec("test", "identity", ("f1", "f2"))
    meta = ShardMeta(
        schema_version="1.0",
        shard_id=f"T{int(temperature)}K_seg0001_rep000",
        temperature_K=temperature,
        beta=1.0,
        replica_id=0,
        segment_id=1,
        exchange_window_id=0,
        n_frames=n_frames,
        dt_ps=0.002,
        feature_spec=feature_spec,
        provenance={},
    )
    return Shard(
        meta=meta,
        X=np.zeros((n_frames, len(feature_spec.columns)), dtype=np.float32),
        t_index=np.arange(n_frames, dtype=np.int64),
        dt_ps=0.002,
    )


def test_balanced_sampler_instantiation():
    shards_by_temp = {300.0: [make_shard(300.0)], 320.0: [make_shard(320.0)]}
    sampler = BalancedTempSampler(shards_by_temp, PairBuilder(tau_steps=1))
    batch = sampler.sample_batch(pairs_per_temperature=1)
    assert len(batch) == 2
