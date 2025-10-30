from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pmarlo.data.aggregate import load_shards_as_dataset
from pmarlo.data.shard import write_shard
from pmarlo.utils.errors import TemperatureConsistencyError


def _mk_src(
    traj_path: Path,
    *,
    run_id: str,
    segment_id: int,
    replica_id: int,
    kind: str,
) -> dict:
    return {
        "traj": str(traj_path),
        "run_id": run_id,
        "created_at": "1970-01-01T00:00:00Z",
        "kind": kind,
        "segment_id": segment_id,
        "replica_id": replica_id,
        "exchange_window_id": 0,
    }


def _mk_shard(
    tmp: Path,
    segment_id: int,
    traj_path: Path,
    temperature: float = 300.0,
    *,
    replica_id: int = 0,
    kind: str = "demux",
) -> Path:
    cvs = {
        "phi": np.linspace(-np.pi, np.pi, 10),
        "psi": np.linspace(-np.pi, np.pi, 10),
    }
    periodic = {"phi": True, "psi": True}
    # Include kind in shard_id to prevent collisions
    t_kelvin = int(round(temperature))
    if kind == "replica":
        shard_id = f"replica_T{t_kelvin}K_seg{segment_id:04d}_rep{replica_id:03d}"
    else:
        shard_id = f"T{t_kelvin}K_seg{segment_id:04d}_rep{replica_id:03d}"
    source = _mk_src(
        traj_path,
        run_id=traj_path.parent.name,
        segment_id=segment_id,
        replica_id=replica_id,
        kind=kind,
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


def test_mixed_kinds_hard_fail(tmp_path: Path):
    run = tmp_path / "run-20250101-000000"
    run.mkdir(parents=True)
    # demux shard
    demux_traj = run / "demux_T300K.dcd"
    demux_traj.write_bytes(b"")
    s1 = _mk_shard(tmp_path, 1, demux_traj, 300.0)
    # replica shard (should be rejected)
    replica_traj = run / "replica_00.dcd"
    replica_traj.write_bytes(b"")
    s2 = _mk_shard(tmp_path, 2, replica_traj, 300.0, kind="replica")

    with pytest.raises(TemperatureConsistencyError):
        load_shards_as_dataset([s1, s2])


def test_mixed_temperatures_hard_fail(tmp_path: Path):
    run = tmp_path / "run-20250101-000000"
    run.mkdir(parents=True)
    d300 = run / "demux_T300K.dcd"
    d300.write_bytes(b"")
    d350 = run / "demux_T350K.dcd"
    d350.write_bytes(b"")
    s1 = _mk_shard(tmp_path, 3, d300, 300.0)
    s2 = _mk_shard(tmp_path, 4, d350, 350.0)

    with pytest.raises(TemperatureConsistencyError):
        load_shards_as_dataset([s1, s2])
