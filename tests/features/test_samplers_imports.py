from __future__ import annotations

import sys
import types

import numpy as np

# Provide lightweight stubs for optional mlcolvar dependency
if "mlcolvar" not in sys.modules:
    mlcolvar = types.ModuleType("mlcolvar")
    sys.modules["mlcolvar"] = mlcolvar
    cvs = types.ModuleType("mlcolvar.cvs")
    cvs.DeepTICA = object  # type: ignore[attr-defined]
    sys.modules["mlcolvar.cvs"] = cvs
    utils = types.ModuleType("mlcolvar.utils.timelagged")
    utils.create_timelagged_dataset = lambda *a, **k: None
    sys.modules["mlcolvar.utils.timelagged"] = utils

from pmarlo.features.samplers import BalancedTempSampler
from pmarlo.shards.pair_builder import PairBuilder
from pmarlo.shards.schema import FeatureSpec, Shard, ShardMeta


def _make_shard(temp: float = 300.0, n_frames: int = 4) -> Shard:
    meta = ShardMeta(
        schema_version="1.0",
        shard_id=f"T{int(temp)}K_seg0001_rep000",
        temperature_K=temp,
        beta=1.0,
        replica_id=0,
        segment_id=1,
        exchange_window_id=0,
        n_frames=n_frames,
        dt_ps=0.002,
        feature_spec=FeatureSpec(name="demo", scaler="identity", columns=("f1",)),
        provenance={},
    )
    return Shard(
        meta=meta,
        X=np.zeros((n_frames, 1), dtype=np.float32),
        t_index=np.arange(n_frames, dtype=np.int64),
        dt_ps=0.002,
    )


def test_balanced_temp_sampler_instantiation():
    shard = _make_shard()
    sampler = BalancedTempSampler({300.0: [shard]}, PairBuilder(tau_steps=1))
    batch = sampler.sample_batch(pairs_per_temperature=1)
    assert len(batch) == 1


def test_weights_and_rare_regions_interact():
    shard = _make_shard(n_frames=5)
    sampler = BalancedTempSampler(
        {300.0: [shard]},
        PairBuilder(tau_steps=1),
        rare_boost=1.0,
        random_seed=0,
    )
    sampler.set_frame_weights(
        shard.meta.shard_id, np.array([0.60, 0.20, 0.10, 0.05, 0.05], dtype=np.float64)
    )
    embeddings = np.array([[0.0], [0.0], [1.0], [1.0], [1.0]], dtype=np.float32)
    sampler.set_cv_embeddings(shard.meta.shard_id, embeddings)

    pairs = sampler.pair_builder.make_pairs(shard)
    sampler._update_occupancy(shard, 300.0, pairs[pairs[:, 0] < 2])
    weights = sampler._pair_weights(shard, 300.0, pairs)
    assert weights is not None
    assert weights[pairs[:, 0] >= 2].mean() > weights[pairs[:, 0] < 2].mean()
