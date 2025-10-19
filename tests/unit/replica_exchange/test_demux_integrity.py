from pathlib import Path

import mdtraj as md
import numpy as np
import pytest

pytest.importorskip("sklearn")

from pmarlo.demultiplexing.demux_metadata import DemuxIntegrityError
from pmarlo.replica_exchange.replica_exchange import ReplicaExchange


def _create_traj(tmp_path, n_frames):
    topology = md.Topology()
    chain = topology.add_chain()
    residue = topology.add_residue("ALA", chain)
    topology.add_atom("CA", md.element.carbon, residue)
    xyz = np.random.rand(n_frames, 1, 3)
    traj = md.Trajectory(xyz, topology)
    pdb_path = tmp_path / "model.pdb"
    traj[0].save_pdb(pdb_path)
    dcd_path = tmp_path / "traj.dcd"
    traj.save_dcd(dcd_path)
    return pdb_path, dcd_path, traj


def test_demux_repair_missing_segment(tmp_path):
    """Test that duplicate mappings in exchange history raise an error."""
    from pmarlo.utils.errors import DemuxIntegrityError

    pdb_path, dcd_path, traj = _create_traj(tmp_path, 3)
    dcd_path2 = tmp_path / "traj2.dcd"
    traj.save_dcd(dcd_path2)
    remd = ReplicaExchange.__new__(ReplicaExchange)
    remd.pdb_file = str(pdb_path)
    remd.trajectory_files = [Path(dcd_path), Path(dcd_path2)]
    remd.temperatures = [300.0, 310.0]
    remd.n_replicas = 2
    remd.exchange_history = [[0, 1], [1, 1], [0, 1]]  # [1, 1] is duplicate - invalid
    remd.reporter_stride = None
    remd.dcd_stride = 1
    remd.exchange_frequency = 1
    remd.output_dir = tmp_path
    remd.integrators = []
    remd._replica_reporter_stride = []

    # Should raise error instead of silently repairing (no fallbacks)
    with pytest.raises(DemuxIntegrityError, match="not a permutation"):
        remd.demux_trajectories(target_temperature=300.0, equilibration_steps=0)


def test_demux_broken_metadata_raises(tmp_path):
    pdb_path, dcd_path, _ = _create_traj(tmp_path, 2)
    remd = ReplicaExchange.__new__(ReplicaExchange)
    remd.pdb_file = str(pdb_path)
    remd.trajectory_files = [Path(dcd_path)]
    remd.temperatures = [300.0]
    remd.n_replicas = 1
    remd.exchange_history = [[0], [0]]
    remd.reporter_stride = None
    remd.dcd_stride = 1
    remd.exchange_frequency = 2
    remd.output_dir = tmp_path
    remd.integrators = []
    remd._replica_reporter_stride = [4]
    with pytest.raises(DemuxIntegrityError):
        remd.demux_trajectories(target_temperature=300.0, equilibration_steps=0)
