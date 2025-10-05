from __future__ import annotations

from pathlib import Path

import mdtraj as md
import numpy as np
import pytest

pytest.importorskip("sklearn")

from pmarlo.demultiplexing.demux_engine import demux_streaming
from pmarlo.demultiplexing.demux_plan import DemuxPlan, DemuxSegmentPlan
from pmarlo.io.trajectory_reader import MDTrajReader, TrajectoryIOError
from pmarlo.io.trajectory_writer import MDTrajDCDWriter, TrajectoryWriteError
from pmarlo.utils.errors import DemuxWriterError


class FailingWriter(MDTrajDCDWriter):
    def write_frames(self, coords: np.ndarray, box: np.ndarray | None = None) -> None:  # type: ignore[override]
        raise TrajectoryWriteError("injected writer failure")


class FailingReader(MDTrajReader):
    def iter_frames(self, path: str, start: int, stop: int, stride: int = 1):  # type: ignore[override]
        raise TrajectoryIOError("injected reader failure")


def _make_replica(tmp: Path, label: int, n_frames: int = 2):
    top = md.Topology()
    chain = top.add_chain()
    residue = top.add_residue("ALA", chain)
    top.add_atom("CA", md.element.carbon, residue)
    xyz = np.zeros((n_frames, 1, 3), dtype=np.float32)
    for i in range(n_frames):
        xyz[i, 0, 0] = i + 100 * label
    traj = md.Trajectory(xyz, top)
    pdb = tmp / f"replica_{label}.pdb"
    dcd = tmp / f"replica_{label}.dcd"
    traj[0].save_pdb(pdb)
    traj.save_dcd(dcd)
    return str(pdb), str(dcd)


def test_demux_engine_writer_error_raises(tmp_path: Path):
    top, dcd = _make_replica(tmp_path, 1, n_frames=2)
    plan = DemuxPlan(
        segments=[DemuxSegmentPlan(0, 0, dcd, 0, 2, 2, False)],
        target_temperature=300.0,
        frames_per_segment=2,
        total_expected_frames=2,
    )
    writer = FailingWriter(rewrite_threshold=1).open(
        str(tmp_path / "out.dcd"), top, overwrite=True
    )
    reader = MDTrajReader(topology_path=top)
    with pytest.raises(DemuxWriterError):
        demux_streaming(plan, top, reader, writer, fill_policy="repeat")


def test_demux_engine_reader_error_is_handled(tmp_path: Path):
    top, dcd = _make_replica(tmp_path, 1, n_frames=1)
    # Expect 1 frame but reader fails => no last frame available, so skipped
    plan = DemuxPlan(
        segments=[DemuxSegmentPlan(0, 0, dcd, 0, 1, 1, False)],
        target_temperature=300.0,
        frames_per_segment=1,
        total_expected_frames=1,
    )
    writer = MDTrajDCDWriter(rewrite_threshold=1).open(
        str(tmp_path / "out.dcd"), top, overwrite=True
    )
    reader = FailingReader(topology_path=top)
    res = demux_streaming(plan, top, reader, writer, fill_policy="repeat")
    writer.close()
    assert res.total_frames_written == 0
    assert 0 in res.skipped_segments
