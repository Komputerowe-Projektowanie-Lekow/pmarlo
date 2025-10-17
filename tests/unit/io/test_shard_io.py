from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pmarlo import read_shard, write_shard


def _source(*, run: str, segment: int, replica: int, kind: str = "demux") -> dict:
    return {
        "created_at": "1970-01-01T00:00:00Z",
        "kind": kind,
        "run_id": run,
        "segment_id": segment,
        "replica_id": replica,
        "exchange_window_id": 0,
    }


def _shard_id(temperature: float, segment: int, replica: int) -> str:
    return f"T{int(round(temperature))}K_seg{segment:04d}_rep{replica:03d}"


def test_shard_roundtrip(tmp_path: Path):
    cvs = {
        "phi": np.linspace(-np.pi, np.pi, 50),
        "psi": np.cos(np.linspace(0, 1, 50)),
    }
    periodic = {"phi": True, "psi": False}
    dtraj = np.arange(50, dtype=np.int32) % 3
    temperature = 300.0
    segment = 0
    replica = 0
    shard_id = _shard_id(temperature, segment, replica)
    json_path = write_shard(
        out_dir=tmp_path,
        shard_id=shard_id,
        cvs=cvs,
        dtraj=dtraj,
        periodic=periodic,
        seed=123,
        temperature=temperature,
        source=_source(run="unit-test", segment=segment, replica=replica),
    )

    details, X, d2 = read_shard(json_path)
    assert X.shape == (50, 2)
    assert X.dtype == np.float64
    assert d2 is not None and d2.dtype == np.int32 and d2.shape == (50,)
    assert details.meta.n_frames == 50
    assert details.cv_names == ("phi", "psi")
    assert details.periodic == (True, False)
    assert details.temperature_K == pytest.approx(temperature)


def test_deterministic_json_bytes(tmp_path: Path):
    cvs = {"a": np.arange(10.0), "b": np.arange(10.0) * 2.0}
    periodic = {"a": False, "b": False}
    temperature = 310.0
    shard_id = _shard_id(temperature, 1, 0)
    p1 = write_shard(
        out_dir=tmp_path / "w1",
        shard_id=shard_id,
        cvs=cvs,
        dtraj=None,
        periodic=periodic,
        seed=7,
        temperature=temperature,
        source=_source(run="same", segment=1, replica=0),
    )
    p2 = write_shard(
        out_dir=tmp_path / "w2",
        shard_id=shard_id,
        cvs=cvs,
        dtraj=None,
        periodic=periodic,
        seed=7,
        temperature=temperature,
        source=_source(run="same", segment=1, replica=0),
    )
    assert p1.read_bytes() == p2.read_bytes()


def test_hash_mismatch_raises(tmp_path: Path):
    cvs = {"x": np.arange(6.0), "y": np.arange(6.0) ** 2}
    periodic = {"x": False, "y": False}
    shard_id = _shard_id(300.0, 2, 0)
    json_path = write_shard(
        out_dir=tmp_path,
        shard_id=shard_id,
        cvs=cvs,
        dtraj=None,
        periodic=periodic,
        seed=0,
        temperature=300.0,
        source=_source(run="tamper", segment=2, replica=0),
    )
    npz_path = json_path.with_suffix(".npz")
    with np.load(npz_path) as f:
        X = f["X"].copy()
        data = {name: f[name] for name in f.files}
    X[0, 0] += 1.0
    data["X"] = X
    np.savez(npz_path, **data)

    with pytest.raises(ValueError):
        read_shard(json_path)
