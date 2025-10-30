from __future__ import annotations

import numpy as np

from pmarlo.markov_state_model.msm_builder import MSMBuilder
from pmarlo.markov_state_model.reweighter import Reweighter
from pmarlo.shards.schema import FeatureSpec, Shard, ShardMeta


def make_shard(shard_id: str, n_frames: int = 3) -> Shard:
    spec = FeatureSpec(name="demo", scaler="identity", columns=("a",))
    meta = ShardMeta(
        schema_version="1.0",
        shard_id=shard_id,
        temperature_K=300.0,
        beta=1.0 / (0.00831446261815324 * 300.0),
        replica_id=0,
        segment_id=0,
        exchange_window_id=0,
        n_frames=n_frames,
        dt_ps=0.002,
        feature_spec=spec,
        provenance={},
    )
    return Shard(
        meta=meta,
        X=np.zeros((n_frames, 1), dtype=np.float32),
        t_index=np.arange(n_frames, dtype=np.int64),
        dt_ps=0.002,
        energy=np.linspace(0.0, 1.0, n_frames, dtype=np.float64),
    )


def test_reweighter_placeholder_weights():
    rw = Reweighter(temperature_ref_K=300.0)
    weights = rw.frame_weights([make_shard("shard-1")])
    assert list(weights.keys()) == ["shard-1"]


def test_msm_builder_with_weights():
    builder = MSMBuilder(tau_steps=2, n_clusters=2)
    Y = np.random.rand(5, 2)
    weights = np.array([0.5, 0.1, 0.1, 0.2, 0.1], dtype=np.float64)
    weights /= weights.sum()
    result = builder.fit([Y], weights_list=[weights])
    np.testing.assert_allclose(result.pi.sum(), 1.0)
