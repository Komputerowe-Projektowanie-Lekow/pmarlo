from __future__ import annotations

from pathlib import Path

import numpy as np

from pmarlo import ShardMeta, read_shard, write_shard


def test_shard_roundtrip(tmp_path: Path):
    cvs = {
        "phi": np.linspace(-np.pi, np.pi, 50),
        "psi": np.cos(np.linspace(0, 1, 50)),
    }
    periodic = {"phi": True, "psi": False}
    dtraj = np.arange(50, dtype=np.int32) % 3
    json_path = write_shard(
        out_dir=tmp_path,
        shard_id="shard_000",
        cvs=cvs,
        dtraj=dtraj,
        periodic=periodic,
        seed=123,
        temperature=300.0,
        source={"note": "unit-test"},
    )

    meta, X, d2 = read_shard(json_path)
    assert isinstance(meta, ShardMeta)
    assert X.shape == (50, 2)
    assert X.dtype == np.float64
    assert d2 is not None and d2.dtype == np.int32 and d2.shape == (50,)
    assert meta.n_frames == 50
    assert tuple(meta.cv_names) == ("phi", "psi")
    assert tuple(meta.periodic) == (True, False)

    # Recompute hash to confirm
    from pmarlo.data.shard import _sha256_bytes

    expected = _sha256_bytes(X, d2)
    assert meta.arrays_hash == expected


def test_deterministic_json_bytes(tmp_path: Path):
    cvs = {"a": np.arange(10.0), "b": np.arange(10.0) * 2.0}
    periodic = {"a": False, "b": False}
    p1 = write_shard(
        out_dir=tmp_path / "w1",
        shard_id="s",
        cvs=cvs,
        dtraj=None,
        periodic=periodic,
        seed=7,
        temperature=310.0,
        source={"note": "same"},
    )
    p2 = write_shard(
        out_dir=tmp_path / "w2",
        shard_id="s",
        cvs=cvs,
        dtraj=None,
        periodic=periodic,
        seed=7,
        temperature=310.0,
        source={"note": "same"},
    )
    assert p1.read_bytes() == p2.read_bytes()


def test_hash_mismatch_raises(tmp_path: Path):
    cvs = {"x": np.arange(6.0), "y": np.arange(6.0) ** 2}
    periodic = {"x": False, "y": False}
    json_path = write_shard(
        out_dir=tmp_path,
        shard_id="bad",
        cvs=cvs,
        dtraj=None,
        periodic=periodic,
        seed=0,
        temperature=300.0,
        source={},
    )
    # Tamper with the NPZ to corrupt the hash
    npz_path = json_path.with_name("bad.npz")
    with np.load(npz_path) as f:
        X = f["X"].copy()
        d = f["dtraj"].copy()
    X[0, 0] += 1.0
    np.savez(npz_path, X=X, dtraj=d)

    import pytest

    with pytest.raises(ValueError):
        read_shard(json_path)
