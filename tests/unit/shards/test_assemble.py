from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pmarlo.shards.assemble import load_shards
from pmarlo.shards.format import write_shard_npz_json
from pmarlo.shards.id import canonical_shard_id
from pmarlo.shards.schema import FeatureSpec, Shard, ShardMeta


def _write_shard(
    tmp_path: Path, *, kind: str, segment_id: int, replica_id: int = 0
) -> Path:
    feature_spec = FeatureSpec(
        name="angles",
        scaler="identity",
        columns=("phi", "psi"),
    )
    temperature = 300.0
    meta = ShardMeta(
        schema_version="pmarlo.shard.v1",
        shard_id="placeholder",
        temperature_K=temperature,
        beta=1.0,
        replica_id=replica_id,
        segment_id=segment_id,
        exchange_window_id=0,
        n_frames=5,
        dt_ps=1.0,
        feature_spec=feature_spec,
        provenance={"kind": kind},
    )
    X = np.arange(meta.n_frames * len(feature_spec.columns), dtype=np.float32).reshape(
        meta.n_frames, len(feature_spec.columns)
    )
    shard = Shard(
        meta=meta,
        X=X,
        t_index=np.arange(meta.n_frames, dtype=np.int64),
        dt_ps=meta.dt_ps,
        energy=None,
        bias=None,
        w_frame=None,
    )
    canonical_id = canonical_shard_id(shard.meta)
    meta = ShardMeta(
        schema_version=shard.meta.schema_version,
        shard_id=canonical_id,
        temperature_K=shard.meta.temperature_K,
        beta=shard.meta.beta,
        replica_id=shard.meta.replica_id,
        segment_id=shard.meta.segment_id,
        exchange_window_id=shard.meta.exchange_window_id,
        n_frames=shard.meta.n_frames,
        dt_ps=shard.meta.dt_ps,
        feature_spec=shard.meta.feature_spec,
        provenance=shard.meta.provenance,
    )
    shard = Shard(
        meta=meta,
        X=shard.X,
        t_index=shard.t_index,
        dt_ps=shard.dt_ps,
        energy=shard.energy,
        bias=shard.bias,
        w_frame=shard.w_frame,
    )
    npz_path = tmp_path / f"{canonical_id}.npz"
    json_path = tmp_path / f"{canonical_id}.json"
    write_shard_npz_json(shard, npz_path, json_path)
    return json_path


def test_load_shards_rejects_mixed_kinds(tmp_path: Path) -> None:
    demux_json = _write_shard(tmp_path, kind="demux", segment_id=1)
    replica_json = _write_shard(tmp_path, kind="replica", segment_id=2)

    with pytest.raises(ValueError, match="Mixed shard kinds"):
        load_shards([demux_json, replica_json])


def test_load_shards_requires_kind_metadata(tmp_path: Path) -> None:
    shard_json = _write_shard(tmp_path, kind="demux", segment_id=3)
    payload = json.loads(shard_json.read_text())
    payload["provenance"].pop("kind", None)
    shard_json.write_text(json.dumps(payload, indent=2))

    with pytest.raises(ValueError, match="provenance\.kind"):
        load_shards([shard_json])
