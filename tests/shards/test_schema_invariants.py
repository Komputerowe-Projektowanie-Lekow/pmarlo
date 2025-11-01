from __future__ import annotations

import numpy as np
import pytest

from pmarlo.shards.schema import FeatureSpec, Shard, ShardMeta, validate_invariants


def make_meta(shard_id: str = "T300K_seg0001_rep000", n_frames: int = 5) -> ShardMeta:
    return ShardMeta(
        schema_version="1.0",
        shard_id=shard_id,
        temperature_K=300.0,
        beta=1.0,
        replica_id=0,
        segment_id=1,
        exchange_window_id=0,
        n_frames=n_frames,
        dt_ps=0.002,
        feature_spec=FeatureSpec(name="demo", scaler="identity", columns=("feat",)),
        provenance={},
    )


def test_validate_invariants_passes_for_contiguous_indices():
    n_frames = 4
    shard = Shard(
        meta=make_meta(n_frames=n_frames),
        X=np.zeros((n_frames, 1), dtype=np.float32),
        t_index=np.arange(n_frames, dtype=np.int64),
        dt_ps=0.002,
    )
    validate_invariants(shard)  # should not raise


def test_validate_invariants_fails_for_non_monotonic_indices():
    n_frames = 4
    shard = Shard(
        meta=make_meta(n_frames=n_frames),
        X=np.zeros((n_frames, 1), dtype=np.float32),
        t_index=np.array([0, 2, 1, 3], dtype=np.int64),
        dt_ps=0.002,
    )
    with pytest.raises(ValueError):
        validate_invariants(shard)
