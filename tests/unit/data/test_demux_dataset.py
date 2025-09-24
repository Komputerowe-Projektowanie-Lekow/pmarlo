from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pmarlo.data.demux_dataset import build_demux_dataset, validate_demux_coverage
from pmarlo.data.shard import read_shard, write_shard
from pmarlo.data.shard_io import load_shard_meta


def _mk_src(traj_path: Path, run_id: str | None = None) -> dict:
    return {
        "traj": str(traj_path),
        "run_id": run_id or traj_path.parent.name,
        "created_at": "1970-01-01T00:00:00Z",
    }


def _mk_shard(
    tmp: Path, name: str, traj_path: Path, temperature: float = 300.0
) -> Path:
    n = 12
    cvs = {
        "phi": np.linspace(-np.pi, np.pi, n),
        "psi": np.linspace(-np.pi, np.pi, n),
    }
    periodic = {"phi": True, "psi": True}
    return write_shard(
        out_dir=tmp,
        shard_id=name,
        cvs=cvs,
        dtraj=None,
        periodic=periodic,
        seed=0,
        temperature=float(temperature),
        source=_mk_src(traj_path),
    )


def _feature_from_json(meta_like) -> np.ndarray:
    # meta_like should carry path info only; use sibling JSON path stored in legacy
    # For tests we assume meta_like.legacy["shard_id"] exists and JSON sits next to it
    # Safer: reconstruct JSON path from source.traj basename and our tmp folder
    # Here we prefer to re-open the JSON via a simple heuristic
    # In tests we call load_shard_meta on a specific JSON path, so stash it
    raw = getattr(meta_like, "legacy", None)
    if isinstance(raw, dict):
        # The test harness knows the JSON path; store it in legacy["__json_path__"] if available
        jp = raw.get("__json_path__")
        if isinstance(jp, str) and jp:
            _, X, _ = read_shard(Path(jp))
            return X
    # Fallback: not expected in tests
    raise RuntimeError("test meta missing __json_path__ for feature extraction")


def test_build_demux_dataset_basic(tmp_path: Path):
    run = tmp_path / "run-20250101-000000"
    run.mkdir(parents=True)
    d300 = run / "demux_T300K.dcd"
    d300.write_bytes(b"")
    # Two shards at 300K
    s1 = _mk_shard(tmp_path, "s1", d300, 300.0)
    s2 = _mk_shard(tmp_path, "s2", d300, 300.0)
    m1 = load_shard_meta(s1)
    m2 = load_shard_meta(s2)
    # Inject JSON path for feature lookup in tests
    m1.legacy["__json_path__"] = str(s1)
    m2.legacy["__json_path__"] = str(s2)

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
    s1 = _mk_shard(tmp_path, "a", d300, 300.0)
    s2 = _mk_shard(tmp_path, "b", d350, 350.0)
    m1 = load_shard_meta(s1)
    m2 = load_shard_meta(s2)
    m1.legacy["__json_path__"] = str(s1)
    m2.legacy["__json_path__"] = str(s2)

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
    s1 = _mk_shard(tmp_path, "a", d300, 300.0)
    s2 = _mk_shard(tmp_path, "b", d350, 350.0)
    m1 = load_shard_meta(s1)
    m2 = load_shard_meta(s2)
    cov = validate_demux_coverage([m1, m2])
    assert 300.0 in cov["temperatures"] and 350.0 in cov["temperatures"]
    assert cov["frames"][300.0] > 0 and cov["frames"][350.0] > 0
