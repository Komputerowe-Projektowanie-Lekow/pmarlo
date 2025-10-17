from __future__ import annotations

from pathlib import Path

import mdtraj as md
import numpy as np
import pytest

pytest.importorskip("sklearn")

from pmarlo.demultiplexing.demux_hints import load_demux_hints
from pmarlo.demultiplexing.demux_metadata import DemuxMetadata
from pmarlo.replica_exchange import config as demux_config
from pmarlo.replica_exchange.replica_exchange import ReplicaExchange


def _make_topology_and_trajs(tmp: Path):
    # Minimal topology with 1 atom
    top = md.Topology()
    chain = top.add_chain()
    residue = top.add_residue("ALA", chain)
    top.add_atom("CA", md.element.carbon, residue)

    # Replica 0: 3 frames so segments at indices 0 and 2 are real
    xyz0 = np.zeros((3, 1, 3), dtype=np.float32)
    for i in range(3):
        xyz0[i, 0, 0] = 100 + i
    traj0 = md.Trajectory(xyz0, top)
    pdb = tmp / "model.pdb"
    traj0[0].save_pdb(pdb)
    dcd0 = tmp / "replica_00.dcd"
    traj0.save_dcd(dcd0)

    # Replica 1: only 1 frame -> segment at index 1 will be missing and filled
    xyz1 = np.zeros((1, 1, 3), dtype=np.float32)
    xyz1[0, 0, 0] = 200  # distinguish replica 1 frame
    traj1 = md.Trajectory(xyz1, top)
    dcd1 = tmp / "replica_01.dcd"
    traj1.save_dcd(dcd1)

    return str(pdb), str(dcd0), str(dcd1)


def _build_remd_stub(pdb: str, dcd0: str, dcd1: str, outdir: Path) -> ReplicaExchange:
    remd = ReplicaExchange.__new__(ReplicaExchange)
    remd.pdb_file = pdb
    remd.trajectory_files = [Path(dcd0), Path(dcd1)]
    remd.temperatures = [300.0, 310.0]
    remd.n_replicas = 2
    # Three segments, target temperature index 0 alternates replicas: 0 -> 1 -> 0
    remd.exchange_history = [[0, 1], [1, 0], [0, 1]]
    remd.reporter_stride = 1
    remd.dcd_stride = 1
    remd.exchange_frequency = 1
    remd.output_dir = outdir
    remd.integrators = []
    remd._replica_reporter_stride = [1, 1]
    # Streaming settings (optional overrides)
    remd.demux_io_backend = "mdtraj"
    remd.demux_fill_policy = "repeat"
    return remd


def test_demux_e2e_streaming_small(tmp_path: Path):
    # Ensure streaming path
    prev_backend = demux_config.DEMUX_IO_BACKEND
    prev_policy = demux_config.DEMUX_FILL_POLICY
    demux_config.DEMUX_IO_BACKEND = "mdtraj"
    demux_config.DEMUX_FILL_POLICY = "repeat"
    try:
        pdb, dcd0, dcd1 = _make_topology_and_trajs(tmp_path)
        remd = _build_remd_stub(pdb, dcd0, dcd1, tmp_path)

        # Run demux to 300 K
        path = remd.demux_trajectories(target_temperature=300.0, equilibration_steps=0)
        assert path is not None and "T300K" in path

        # Validate demuxed frames: 3 segments * 1 frame each = 3 frames
        demux = md.load(path, top=pdb)
        assert demux.n_frames == 3
        # x coordinates should be: seg0(100), seg1(fill->100), seg2(102)
        xs = demux.xyz[:, 0, 0]
        assert np.isclose(xs[0], 100)
        assert np.isclose(xs[1], 100)  # filled
        assert np.isclose(xs[2], 102)

        # Metadata and hints
        meta_path = Path(path).with_suffix(".meta.json")
        assert meta_path.exists()
        meta = DemuxMetadata.from_json(meta_path)
        assert meta.schema_version == 2
        assert meta.segment_count == 3
        # frames_per_segment mode should be 1
        assert meta.frames_per_segment == 1
        # Hints reflect a gap in the middle segment
        hints = load_demux_hints(meta_path)
        assert hints.fill_policy == "repeat"
        assert hints.contiguous_blocks == [(0, 1), (2, 3)]
    finally:
        demux_config.DEMUX_IO_BACKEND = prev_backend
        demux_config.DEMUX_FILL_POLICY = prev_policy
