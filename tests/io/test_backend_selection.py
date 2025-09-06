from __future__ import annotations

import importlib

import pytest

from pmarlo.io.trajectory_reader import get_reader, MDTrajReader, TrajectoryIOError
from pmarlo.io.trajectory_writer import get_writer, MDTrajDCDWriter, TrajectoryWriteError


def _has_mdanalysis() -> bool:
    try:
        importlib.import_module("MDAnalysis")
        return True
    except Exception:
        return False


def test_get_reader_mdtraj_default():
    r = get_reader("mdtraj", topology_path=None)
    assert isinstance(r, MDTrajReader)


def test_get_writer_mdtraj_default():
    w = get_writer("mdtraj", topology_path=None)
    assert isinstance(w, MDTrajDCDWriter)


def test_get_reader_mdanalysis_selection_or_error():
    have = _has_mdanalysis()
    if have:
        r = get_reader("mdanalysis", topology_path="dummy.pdb")
        assert type(r).__name__.lower().startswith("mdanalysis")
    else:
        with pytest.raises(TrajectoryIOError):
            get_reader("mdanalysis", topology_path="dummy.pdb")


def test_get_writer_mdanalysis_selection_or_error():
    have = _has_mdanalysis()
    if have:
        w = get_writer("mdanalysis", topology_path="dummy.pdb")
        assert type(w).__name__.lower().startswith("mdanalysis")
    else:
        with pytest.raises(TrajectoryWriteError):
            get_writer("mdanalysis", topology_path="dummy.pdb")


def test_get_reader_unknown_backend_raises():
    with pytest.raises(TrajectoryIOError):
        get_reader("unknown_backend", topology_path=None)


def test_get_writer_unknown_backend_raises():
    with pytest.raises(TrajectoryWriteError):
        get_writer("unknown_backend", topology_path=None)

