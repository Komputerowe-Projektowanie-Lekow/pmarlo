from __future__ import annotations

"""Performance benchmarks for the balanced sampler used in REMD training."""

import os
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pytest

from pmarlo.samplers.balanced import BalancedTempSampler
from pmarlo.shards.pair_builder import PairBuilder
from pmarlo.shards.schema import FeatureSpec, Shard, ShardMeta

pytestmark = [pytest.mark.perf, pytest.mark.benchmark, pytest.mark.samplers]

pytest.importorskip("pytest_benchmark", reason="pytest-benchmark not installed")

if not os.getenv("PMARLO_RUN_PERF"):
    pytest.skip(
        "perf tests disabled; set PMARLO_RUN_PERF=1 to run", allow_module_level=True
    )


@dataclass(frozen=True)
class _ShardSpec:
    temperature: float
    replica_id: int
    segment_id: int
    seed: int


def _make_shard(spec: _ShardSpec, n_frames: int, n_features: int) -> Shard:
    rng = np.random.default_rng(spec.seed)
    feature_spec = FeatureSpec(
        name="demux_features",
        scaler="standard",
        columns=tuple(f"f{i}" for i in range(n_features)),
    )
    shard_id = f"T{int(round(spec.temperature))}K_seg{spec.segment_id:04d}_rep{spec.replica_id:03d}"
    meta = ShardMeta(
        schema_version="1.0",
        shard_id=shard_id,
        temperature_K=spec.temperature,
        beta=1.0 / spec.temperature,
        replica_id=spec.replica_id,
        segment_id=spec.segment_id,
        exchange_window_id=0,
        n_frames=n_frames,
        dt_ps=2.0,
        feature_spec=feature_spec,
        provenance={"kind": "demux"},
    )
    X = rng.normal(size=(n_frames, n_features)).astype(np.float32)
    t_index = np.arange(n_frames, dtype=np.int64)
    return Shard(meta=meta, X=X, t_index=t_index, dt_ps=meta.dt_ps)


def _build_shards_by_temperature(
    temperatures: List[float],
    shards_per_temperature: int,
    n_frames: int,
    n_features: int,
) -> Dict[float, List[Shard]]:
    shards_by_temp: Dict[float, List[Shard]] = {}
    for t_idx, temperature in enumerate(temperatures):
        shards: List[Shard] = []
        for shard_idx in range(shards_per_temperature):
            spec = _ShardSpec(
                temperature=temperature,
                replica_id=shard_idx,
                segment_id=t_idx * shards_per_temperature + shard_idx,
                seed=10_000 + t_idx * 97 + shard_idx * 13,
            )
            shards.append(_make_shard(spec, n_frames=n_frames, n_features=n_features))
        shards_by_temp[temperature] = shards
    return shards_by_temp


@pytest.fixture(scope="module")
def sampler_workload() -> Dict[float, List[Shard]]:
    temperatures = [285.0, 300.0, 315.0, 330.0]
    return _build_shards_by_temperature(
        temperatures=temperatures,
        shards_per_temperature=4,
        n_frames=2048,
        n_features=6,
    )


def test_balanced_sampler_uniform_sampling(benchmark, sampler_workload):
    """Benchmark uniform sampling across temperatures without rare boosts."""

    sampler = BalancedTempSampler(
        sampler_workload,
        PairBuilder(tau_steps=4),
        rare_boost=0.0,
        random_seed=123,
    )

    def _sample_batch():
        return sampler.sample_batch(pairs_per_temperature=64)

    batch = benchmark(_sample_batch)
    assert len(batch) == len(sampler_workload)
    for shard, pairs in batch:
        assert pairs.ndim == 2
        assert pairs.shape[1] == 2
        assert 0 < pairs.shape[0] <= 64


def test_balanced_sampler_with_frame_weights(benchmark, sampler_workload):
    """Benchmark frame-weight aware sampling to ensure weights remain normalized."""

    sampler = BalancedTempSampler(
        sampler_workload,
        PairBuilder(tau_steps=6),
        rare_boost=0.0,
        random_seed=456,
    )

    for shards in sampler_workload.values():
        for shard in shards:
            weights = np.linspace(1.0, 2.0, shard.meta.n_frames, dtype=np.float64)
            sampler.set_frame_weights(shard.meta.shard_id, weights)

    def _sample_weighted():
        return sampler.sample_batch(pairs_per_temperature=96)

    batch = benchmark(_sample_weighted)
    assert len(batch) == len(sampler_workload)
    for shard, pairs in batch:
        weights = sampler._pair_weights(shard, shard.meta.temperature_K, pairs)
        assert weights is not None
        assert pairs.shape[0] == weights.shape[0]
        np.testing.assert_allclose(np.sum(weights), 1.0, rtol=1e-6)


def test_balanced_sampler_with_rare_boost_embeddings(benchmark, sampler_workload):
    """Benchmark rare-event boosting with cached CV embeddings."""

    sampler = BalancedTempSampler(
        sampler_workload,
        PairBuilder(tau_steps=5),
        rare_boost=0.4,
        random_seed=789,
    )

    for shards in sampler_workload.values():
        for shard in shards:
            embeddings = np.vstack(
                (
                    np.sin(np.linspace(0, 4.0, shard.meta.n_frames)),
                    np.cos(np.linspace(0, 4.0, shard.meta.n_frames)),
                )
            ).T.astype(np.float32)
            sampler.set_cv_embeddings(shard.meta.shard_id, embeddings)

    def _sample_rare_boost():
        return sampler.sample_batch(pairs_per_temperature=72)

    batch = benchmark(_sample_rare_boost)
    assert len(batch) == len(sampler_workload)
    for shard, pairs in batch:
        assert pairs.size > 0
        temperature = shard.meta.temperature_K
        assert temperature in sampler._occupancy
        assert sampler._occupancy[temperature]
