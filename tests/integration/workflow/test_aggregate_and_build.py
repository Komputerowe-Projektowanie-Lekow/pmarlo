from __future__ import annotations

from itertools import count
from pathlib import Path

import numpy as np
import pytest

from pmarlo import aggregate_and_build, write_shard
from pmarlo.transform import AppliedOpts, BuildOpts
from pmarlo.transform.build import BuildResult
from pmarlo.transform.plan import TransformPlan, TransformStep

pytestmark = pytest.mark.integration


_SEGMENT_COUNTER = count()


def _canonical_shard_id(temperature_K: float, segment_id: int, replica_id: int) -> str:
    temp = int(round(temperature_K))
    return f"T{temp}K_seg{segment_id:04d}_rep{replica_id:03d}"


def _source_metadata(segment_id: int, replica_id: int) -> dict[str, object]:
    return {
        "created_at": "1970-01-01T00:00:00Z",
        "kind": "demux",
        "run_id": "integration-test",
        "segment_id": int(segment_id),
        "replica_id": int(replica_id),
        "exchange_window_id": 0,
    }


def _mk_shard(
    tmp: Path, name: str, order: tuple[str, str], periodic=(True, True)
) -> Path:
    cvs = {
        order[0]: np.random.default_rng(0).normal(size=30),
        order[1]: np.random.default_rng(1).normal(size=30),
    }
    periodic_map = {order[0]: periodic[0], order[1]: periodic[1]}
    segment_id = next(_SEGMENT_COUNTER)
    replica_id = 0
    shard_id = _canonical_shard_id(300.0, segment_id, replica_id)
    return write_shard(
        out_dir=tmp,
        shard_id=shard_id,
        cvs=cvs,
        dtraj=None,
        periodic=periodic_map,
        seed=0,
        temperature=300.0,
        source=_source_metadata(segment_id, replica_id),
    )


def test_aggregate_and_build_bundle(tmp_path: Path):
    s1 = _mk_shard(tmp_path, "s1", ("phi", "psi"), (True, True))
    s2 = _mk_shard(tmp_path, "s2", ("phi", "psi"), (True, True))

    plan = TransformPlan(steps=(TransformStep("SMOOTH_FES", {"sigma": 0.5}),))
    opts = BuildOpts(seed=123, temperature=300.0, n_states=None)
    applied = AppliedOpts(bins={"phi": 16, "psi": 16}, lag=2)
    bundle = tmp_path / "bundle.json"

    result, ds_hash = aggregate_and_build(
        [s1, s2], opts=opts, plan=plan, applied=applied, out_bundle=bundle
    )
    assert result.metadata is not None
    assert len(ds_hash) == 64
    assert bundle.exists() and bundle.stat().st_size > 0

    # Roundtrip JSON
    loaded = BuildResult.from_json(bundle.read_text())
    assert loaded.metadata is not None


def test_aggregate_mismatch_order_raises(tmp_path: Path):
    s1 = _mk_shard(tmp_path, "a", ("phi", "psi"))
    s2 = _mk_shard(tmp_path, "b", ("psi", "phi"))

    plan = TransformPlan(steps=())
    opts = BuildOpts(seed=0)
    applied = AppliedOpts()

    import pytest

    with pytest.raises(ValueError):
        aggregate_and_build(
            [s1, s2],
            opts=opts,
            plan=plan,
            applied=applied,
            out_bundle=tmp_path / "b.json",
        )
