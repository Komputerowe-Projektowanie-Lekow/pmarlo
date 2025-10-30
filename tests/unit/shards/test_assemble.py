from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pmarlo.shards.assemble import load_shards
from pmarlo.shards.format import write_shard_npz_json
from pmarlo.shards.schema import FeatureSpec, Shard, ShardMeta


def _canonical_shard_id(
    kind: str, *, temperature: float, segment_id: int, replica_id: int
) -> str:
    t_kelvin = int(round(float(temperature)))
    if kind == "replica":
        return f"replica_T{t_kelvin}K_seg{segment_id:04d}_rep{replica_id:03d}"
    return f"T{t_kelvin}K_seg{segment_id:04d}_rep{replica_id:03d}"


def _write_shard(
    tmp_path: Path, *, kind: str, segment_id: int, replica_id: int = 0
) -> Path:
    feature_spec = FeatureSpec(
        name="angles",
        scaler="identity",
        columns=("phi", "psi"),
    )
    temperature = 300.0
    shard_id = _canonical_shard_id(
        kind, temperature=temperature, segment_id=segment_id, replica_id=replica_id
    )
    meta = ShardMeta(
        schema_version="pmarlo.shard.v1",
        shard_id=shard_id,
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
    npz_path = tmp_path / f"{shard_id}.npz"
    json_path = tmp_path / f"{shard_id}.json"
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
