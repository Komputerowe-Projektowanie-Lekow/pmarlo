from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pmarlo.data.demux_dataset import build_demux_dataset, validate_demux_coverage
from pmarlo.data.shard import read_shard, write_shard
from pmarlo.data.shard_io import load_shard_meta


def _mk_src(
    traj_path: Path,
    *,
    run_id: str,
    segment_id: int,
    replica_id: int = 0,
) -> dict:
    return {
        "traj": str(traj_path),
        "run_id": run_id,
        "created_at": "1970-01-01T00:00:00Z",
        "kind": "demux",
        "segment_id": segment_id,
        "replica_id": replica_id,
        "exchange_window_id": 0,
    }


def _mk_shard(
    tmp: Path,
    segment_id: int,
    traj_path: Path,
    temperature: float = 300.0,
    replica_id: int = 0,
) -> Path:
    n = 12
    cvs = {
        "phi": np.linspace(-np.pi, np.pi, n),
        "psi": np.linspace(-np.pi, np.pi, n),
    }
    periodic = {"phi": True, "psi": True}
    shard_id = f"T{int(round(temperature))}K_seg{segment_id:04d}_rep{replica_id:03d}"
    source = _mk_src(
        traj_path,
        run_id=traj_path.parent.name,
        segment_id=segment_id,
        replica_id=replica_id,
    )
    return write_shard(
        out_dir=tmp,
        shard_id=shard_id,
        cvs=cvs,
        dtraj=None,
        periodic=periodic,
        seed=0,
        temperature=float(temperature),
        source=source,
    )


def _feature_from_json(meta_like) -> np.ndarray:
    json_path = Path(meta_like.json_path)
    _, X, _ = read_shard(json_path)
    return X


def test_build_demux_dataset_basic(tmp_path: Path):
    run = tmp_path / "run-20250101-000000"
    run.mkdir(parents=True)
    d300 = run / "demux_T300K.dcd"
    d300.write_bytes(b"")
    # Two shards at 300K
    s1 = _mk_shard(tmp_path, 1, d300, 300.0)
    s2 = _mk_shard(tmp_path, 2, d300, 300.0)
    m1 = load_shard_meta(s1)
    m2 = load_shard_meta(s2)
    ds = build_demux_dataset(
        [m1, m2], 300.0, lag_steps=3, feature_fn=_feature_from_json
    )
    assert ds.temperature_K == pytest.approx(300.0)
    assert len(ds.shards) == 2
    assert len(ds.X_list) == 2
    # With n=12 and lag=3 per shard, pairs per shard = 9
    assert ds.pairs.shape[1] == 2
    assert ds.pairs.shape[0] == 18
    # default weights (no bias) are ones
    assert ds.weights.shape[0] == ds.pairs.shape[0]
    assert np.allclose(ds.weights, 1.0)


def test_build_demux_dataset_mixed_temps_rejected(tmp_path: Path):
    run = tmp_path / "run-20250101-000000"
    run.mkdir(parents=True)
    d300 = run / "demux_T300K.dcd"
    d350 = run / "demux_T350K.dcd"
    d300.write_bytes(b"")
    d350.write_bytes(b"")
    s1 = _mk_shard(tmp_path, 3, d300, 300.0)
    s2 = _mk_shard(tmp_path, 4, d350, 350.0)
    m1 = load_shard_meta(s1)
    m2 = load_shard_meta(s2)
    # Target 350 uses only 350K shard; valid
    ds = build_demux_dataset(
        [m1, m2], 350.0, lag_steps=2, feature_fn=_feature_from_json
    )
    assert len(ds.shards) == 1 and ds.temperature_K == pytest.approx(350.0)

    # Mixed temperatures in input are fine; builder selects only the requested T
    ds2 = build_demux_dataset(
        [m1, m2], 300.0, lag_steps=2, feature_fn=_feature_from_json
    )
    assert ds2.temperature_K == pytest.approx(300.0)
    assert len(ds2.shards) == 1


def test_validate_demux_coverage(tmp_path: Path):
    run = tmp_path / "run-20250101-000000"
    run.mkdir(parents=True)
    d300 = run / "demux_T300K.dcd"
    d350 = run / "demux_T350K.dcd"
    d300.write_bytes(b"")
    d350.write_bytes(b"")
    s1 = _mk_shard(tmp_path, 3, d300, 300.0)
    s2 = _mk_shard(tmp_path, 4, d350, 350.0)
    m1 = load_shard_meta(s1)
    m2 = load_shard_meta(s2)
    cov = validate_demux_coverage([m1, m2])
    assert 300.0 in cov["temperatures"] and 350.0 in cov["temperatures"]
    assert cov["frames"][300.0] > 0 and cov["frames"][350.0] > 0
