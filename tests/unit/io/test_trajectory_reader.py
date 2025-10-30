from __future__ import annotations

from pathlib import Path
import sys

import mdtraj as md
import numpy as np
import pytest

import types

from pmarlo.io.trajectory_reader import (
    MDAnalysisReader,
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


def test_mdanalysis_reader_respects_stride(monkeypatch):
    frames = [
        np.full((1, 3), fill_value=float(i)) for i in range(6)
    ]  # shape (n_atoms=1, 3)

    class DummyTimestep:
        def __init__(self, arr: np.ndarray) -> None:
            self.positions = arr

    class DummyTrajectory:
        def __init__(self) -> None:
            self._frames = [DummyTimestep(arr) for arr in frames]
            self.requested_slice = None

        def __getitem__(self, key):
            self.requested_slice = key
            if isinstance(key, slice):
                start, stop, step = key.indices(len(self._frames))
                indices = range(start, stop, step)
            else:  # pragma: no cover - defensive fallback
                indices = [int(key)]
            return (self._frames[i] for i in indices)

    module = types.ModuleType("MDAnalysis")

    class DummyUniverse:
        def __init__(self, topology_path: str, traj_path: str) -> None:
            self.topology_path = topology_path
            self.traj_path = traj_path
            self.trajectory = DummyTrajectory()
            module.last_universe = self

    module.Universe = DummyUniverse  # type: ignore[attr-defined]
    module.last_universe = None  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "MDAnalysis", module)

    reader = MDAnalysisReader(topology_path="top.pdb")
    out = list(reader.iter_frames("traj.dcd", start=1, stop=6, stride=2))

    assert module.last_universe is not None
    requested = module.last_universe.trajectory.requested_slice
    assert isinstance(requested, slice)
    assert (requested.start, requested.stop, requested.step) == (1, 6, 2)
    assert len(out) == 3
    assert all(arr.shape == (1, 3) for arr in out)
    assert np.allclose(out[0], frames[1])
    assert np.allclose(out[1], frames[3])
    assert np.allclose(out[2], frames[5])
