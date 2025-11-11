from __future__ import annotations

import json
from pathlib import Path

import mdtraj as md
import numpy as np
import pytest

pytest.importorskip("sklearn")

from pmarlo.replica_exchange import config as demux_config
from pmarlo.replica_exchange.replica_exchange import ReplicaExchange


def _make_minimal_traj(tmp_path: Path, n_frames: int = 2):
    top = md.Topology()
    chain = top.add_chain()
    residue = top.add_residue("ALA", chain)
    top.add_atom("CA", md.element.carbon, residue)
    xyz = np.zeros((n_frames, 1, 3), dtype=np.float32)
    for i in range(n_frames):
        xyz[i, 0, 0] = i
    traj = md.Trajectory(xyz, top)
    pdb = tmp_path / "model.pdb"
    traj[0].save_pdb(pdb)
    dcd = tmp_path / "replica_00.dcd"
    traj.save_dcd(dcd)
    dcd2 = tmp_path / "replica_01.dcd"
    traj.save_dcd(dcd2)
    return str(pdb), str(dcd), str(dcd2)


def _build_minimal_remd(
    pdb: str, dcd0: str, dcd1: str, tmp_path: Path
) -> ReplicaExchange:
    remd = ReplicaExchange.__new__(ReplicaExchange)
    remd.pdb_file = pdb
    remd.trajectory_files = [Path(dcd0), Path(dcd1)]
    remd.temperatures = [300.0, 310.0]
    remd.n_replicas = 2
    # Two segments: first target at replica 0, then at replica 1
    remd.exchange_history = [[0, 1], [1, 0]]
    remd.reporter_stride = 1
    remd.dcd_stride = 1
    remd.exchange_frequency = 1
    remd.output_dir = tmp_path
    remd.integrators = []
    remd._replica_reporter_stride = [1, 1]
    return remd


def test_demux_facade_streaming_enabled(tmp_path: Path):
    pdb, dcd0, dcd1 = _make_minimal_traj(tmp_path)
    remd = _build_minimal_remd(pdb, dcd0, dcd1, tmp_path)

    path = remd.demux_trajectories(target_temperature=300.0, equilibration_steps=0)

    assert path is not None
    meta_path = Path(path).with_suffix(".meta.json")
    assert meta_path.exists()
    data = json.loads(meta_path.read_text())
    assert data.get("schema_version") == 2
    # 2 segments, 1 frame each -> 2 frames total
    traj = md.load(str(path), top=pdb)
    assert traj.n_frames == 2


def test_demux_facade_config_flag_is_ignored(tmp_path: Path):
    pdb, dcd0, dcd1 = _make_minimal_traj(tmp_path)
    remd = _build_minimal_remd(pdb, dcd0, dcd1, tmp_path)

    prev = demux_config.DEMUX_STREAMING_ENABLED
    demux_config.DEMUX_STREAMING_ENABLED = False
    try:
        path = remd.demux_trajectories(target_temperature=300.0, equilibration_steps=0)
    finally:
        demux_config.DEMUX_STREAMING_ENABLED = prev

    assert path is not None
    meta_path = Path(path).with_suffix(".meta.json")
    assert meta_path.exists()
    data = json.loads(meta_path.read_text())
    assert data.get("schema_version") == 2
    traj = md.load(str(path), top=pdb)
    assert traj.n_frames == 2
