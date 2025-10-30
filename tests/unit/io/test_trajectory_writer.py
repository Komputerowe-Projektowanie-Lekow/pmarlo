from __future__ import annotations

import builtins
from pathlib import Path
import sys

import mdtraj as md
import numpy as np
import pytest

from pmarlo.io.trajectory_reader import MDTrajReader
from pmarlo.io.trajectory_writer import (
    MDAnalysisDCDWriter,
    MDTrajDCDWriter,
    TrajectoryWriteError,
)


def _make_topology(tmp_path: Path, n_atoms: int = 3):
    top = md.Topology()
    chain = top.add_chain()
    residue = top.add_residue("ALA", chain)
    for _ in range(n_atoms):
        top.add_atom("C", md.element.carbon, residue)
    pdb_path = tmp_path / "topology.pdb"
    # Save a single-frame PDB as topology
    xyz0 = np.zeros((1, n_atoms, 3), dtype=np.float32)
    md.Trajectory(xyz0, top).save_pdb(pdb_path)
    return str(pdb_path)


def test_writer_append_like_and_reader_roundtrip(tmp_path: Path):
    n_atoms = 2
    top_path = _make_topology(tmp_path, n_atoms=n_atoms)
    out_path = tmp_path / "out.dcd"

    # Prepare two chunks with easily verifiable content
    chunk1 = np.zeros((2, n_atoms, 3), dtype=np.float32)
    chunk2 = np.zeros((3, n_atoms, 3), dtype=np.float32)
    for i in range(chunk1.shape[0]):
        chunk1[i, :, 0] = i
        chunk1[i, :, 1] = i + 10
    for i in range(chunk2.shape[0]):
        v = i + chunk1.shape[0]
        chunk2[i, :, 0] = v
        chunk2[i, :, 1] = v + 10

    writer = MDTrajDCDWriter(rewrite_threshold=2).open(
        str(out_path), top_path, overwrite=True
    )
    writer.write_frames(chunk1)
    writer.write_frames(chunk2)
    writer.close()

    # Read back using streaming reader
    reader = MDTrajReader(topology_path=top_path)
    n = reader.probe_length(str(out_path))
    assert n == chunk1.shape[0] + chunk2.shape[0]

    frames = list(reader.iter_frames(str(out_path), start=0, stop=n, stride=1))
    assert len(frames) == n
    # Validate coordinates
    for i, arr in enumerate(frames):
        assert arr.shape == (n_atoms, 3)
        assert np.allclose(arr[:, 0], i)
        assert np.allclose(arr[:, 1], i + 10)


def test_writer_overwrite_flag(tmp_path: Path):
    top_path = _make_topology(tmp_path, n_atoms=1)
    out_path = tmp_path / "out.dcd"

    # Initial write
    w = MDTrajDCDWriter(rewrite_threshold=10).open(
        str(out_path), top_path, overwrite=True
    )
    w.write_frames(np.zeros((1, 1, 3), dtype=np.float32))
    w.close()

    # Overwrite allowed
    w = MDTrajDCDWriter(rewrite_threshold=10).open(
        str(out_path), top_path, overwrite=True
    )
    w.write_frames(np.zeros((2, 1, 3), dtype=np.float32))
    w.close()

    # Now length should be 2
    reader = MDTrajReader(topology_path=top_path)
    assert reader.probe_length(str(out_path)) == 2


def test_mdanalysis_writer_missing_dependency(monkeypatch, tmp_path: Path):
    monkeypatch.delitem(sys.modules, "MDAnalysis", raising=False)

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "MDAnalysis" or name.startswith("MDAnalysis."):
            raise ModuleNotFoundError("No module named 'MDAnalysis'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    writer = MDAnalysisDCDWriter()
    with pytest.raises(TrajectoryWriteError, match="MDAnalysis is required"):
        writer.open(str(tmp_path / "out.dcd"), topology_path=str(tmp_path / "top.pdb"))
