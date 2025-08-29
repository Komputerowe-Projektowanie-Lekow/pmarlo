from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pmarlo import read_shard, write_shard


def _sha256_bytes(*arrays: np.ndarray) -> str:
    # Local copy of the hashing used in pmarlo.data.shard
    from hashlib import sha256

    h = sha256()
    for arr in arrays:
        a = np.ascontiguousarray(arr)
        h.update(str(a.dtype.str).encode("utf-8"))
        h.update(str(a.shape).encode("utf-8"))
        h.update(a.tobytes())
    return h.hexdigest()


def test_write_with_bias_and_read_sets_flags(tmp_path: Path):
    out = tmp_path / "shards"
    n = 10
    phi = np.linspace(-np.pi, np.pi, n)
    psi = np.sin(np.linspace(0, 2 * np.pi, n))
    bias = np.linspace(0.0, 1.0, n)

    jp = write_shard(
        out_dir=out,
        shard_id="shard_0000",
        cvs={"phi": phi, "psi": psi},
        dtraj=None,
        periodic={"phi": True, "psi": False},
        seed=123,
        temperature=300.0,
        source={"created_at": "1970-01-01T00:00:00Z"},
        bias_potential=bias,
    )

    # NPZ contains bias_potential array with matching length
    npz_path = jp.with_name("shard_0000.npz")
    with np.load(npz_path) as z:
        assert "bias_potential" in z.files
        assert z["bias_potential"].shape == (n,)

    meta, X, dtraj = read_shard(jp)
    assert dtraj is None
    assert X.shape == (n, 2)
    # Source carries explicit flags
    assert bool(meta.source.get("has_bias")) is True
    assert float(meta.source.get("temperature_K", 0.0)) == 300.0


def test_read_shard_compatibility_missing_bias_key(tmp_path: Path):
    # Craft an "old" shard: NPZ has only X/dtraj, JSON matches arrays_hash
    out = tmp_path / "old"
    out.mkdir(parents=True, exist_ok=True)
    shard_id = "shard_0001"
    X = np.column_stack([np.arange(5.0), np.arange(5.0) + 1.0]).astype(np.float64)
    dtraj = np.array([], dtype=np.int32)
    arrays_hash = _sha256_bytes(X)  # since dtraj is empty, hash only over X

    # Write NPZ without bias_potential key
    np.savez(out / f"{shard_id}.npz", X=X, dtraj=dtraj)

    # Minimal JSON metadata expected by reader
    meta = {
        "shard_id": shard_id,
        "seed": 7,
        "temperature": 310.0,
        "n_frames": int(X.shape[0]),
        "cv_names": ["phi", "psi"],
        "periodic": [False, False],
        "created_at": "1970-01-01T00:00:00Z",
        "source": {"note": "old-format"},
        "arrays_hash": arrays_hash,
        "schema_version": "pmarlo.shard.v1",
    }
    (out / f"{shard_id}.json").write_text(
        json.dumps(meta, sort_keys=True, separators=(",", ":"))
    )

    meta_read, X_read, dtraj_read = read_shard(out / f"{shard_id}.json")
    assert dtraj_read is None
    np.testing.assert_allclose(X_read, X)
    # Reader derives has_bias=False and populates temperature_K
    assert bool(meta_read.source.get("has_bias")) is False
    assert float(meta_read.source.get("temperature_K", 0.0)) == 310.0
