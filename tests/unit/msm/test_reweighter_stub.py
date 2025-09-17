from __future__ import annotations

import numpy as np

from pmarlo.markov_state_model.reweighter import Reweighter
from pmarlo.shards.schema import FeatureSpec, Shard, ShardMeta


def _make_shard(shard_id: str, temperature: float, energy: np.ndarray) -> Shard:
    spec = FeatureSpec(name="demo", scaler="identity", columns=("a",))
    meta = ShardMeta(
        schema_version="1.0",
        shard_id=shard_id,
        temperature_K=temperature,
        beta=1.0 / (0.00831446261815324 * temperature),
        replica_id=0,
        segment_id=0,
        exchange_window_id=0,
        n_frames=energy.size,
        dt_ps=0.002,
        feature_spec=spec,
        provenance={},
    )
    return Shard(
        meta=meta,
        X=np.zeros((energy.size, 1), dtype=np.float32),
        t_index=np.arange(energy.size, dtype=np.int64),
        dt_ps=0.002,
        energy=energy,
    )


def test_reweighter_prefers_low_energy_frames():
    shard = _make_shard("shard-1", 320.0, np.array([5.0, 1.0], dtype=np.float64))
    rw = Reweighter(temperature_ref_K=300.0)
    weights = rw.frame_weights([shard])
    w = weights["shard-1"]
    assert w[1] > w[0]
    np.testing.assert_allclose(w.sum(), 1.0)
