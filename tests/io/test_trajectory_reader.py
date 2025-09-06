from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import mdtraj as md

from pmarlo.io.trajectory_reader import (
    MDTrajReader,
    TrajectoryMissingTopologyError,
)


def _make_tiny_traj(tmp_path: Path, n_frames: int = 6, n_atoms: int = 3):
    # Create minimal topology with n_atoms carbons
    top = md.Topology()
    chain = top.add_chain()
    residue = top.add_residue("GLY", chain)
    for _ in range(n_atoms):
        top.add_atom("C", md.element.carbon, residue)

    xyz = np.zeros((n_frames, n_atoms, 3), dtype=np.float32)
    # Make easily identifiable coordinates: frame index in x component
    for i in range(n_frames):
        xyz[i, :, 0] = i
        xyz[i, :, 1] = i * 10
        xyz[i, :, 2] = i * 100
    traj = md.Trajectory(xyz, top)

    pdb_path = tmp_path / "topology.pdb"
    dcd_path = tmp_path / "traj.dcd"
    traj[0].save_pdb(pdb_path)
    traj.save_dcd(dcd_path)
    return traj, pdb_path, dcd_path


def test_probe_length_and_iter_frames(tmp_path: Path):
    traj, pdb_path, dcd_path = _make_tiny_traj(tmp_path, n_frames=6, n_atoms=2)

    reader = MDTrajReader(topology_path=str(pdb_path))
    n = reader.probe_length(str(dcd_path))
    assert n == traj.n_frames

    # Full range, stride=1
    frames = list(reader.iter_frames(str(dcd_path), start=0, stop=n, stride=1))
    assert len(frames) == n
    assert all(isinstance(f, np.ndarray) and f.shape == (2, 3) for f in frames)
    # Contents match x/y/z pattern
    for i, arr in enumerate(frames):
        assert np.allclose(arr[:, 0], i)
        assert np.allclose(arr[:, 1], i * 10)
        assert np.allclose(arr[:, 2], i * 100)

    # Subrange with stride=2 (frames 2 and 4)
    sub = list(reader.iter_frames(str(dcd_path), start=2, stop=5, stride=2))
    assert len(sub) == 2
    assert np.allclose(sub[0][:, 0], 2)
    assert np.allclose(sub[1][:, 0], 4)


def test_missing_topology_raises(tmp_path: Path):
    _, _pdb, dcd_path = _make_tiny_traj(tmp_path, n_frames=3, n_atoms=1)

    # DCD requires topology; reader without topology should raise
    reader = MDTrajReader(topology_path=None)
    with pytest.raises(TrajectoryMissingTopologyError):
        _ = reader.probe_length(str(dcd_path))
    with pytest.raises(TrajectoryMissingTopologyError):
        list(reader.iter_frames(str(dcd_path), start=0, stop=2, stride=1))
