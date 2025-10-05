from __future__ import annotations

from pathlib import Path

import mdtraj as md
import numpy as np

from pmarlo.demultiplexing.demux_engine import demux_streaming
from pmarlo.demultiplexing.demux_plan import DemuxPlan, DemuxSegmentPlan
from pmarlo.io.trajectory_reader import MDTrajReader
from pmarlo.io.trajectory_writer import MDTrajDCDWriter


class CountingWriter:
    def __init__(self, inner: MDTrajDCDWriter):
        self.inner = inner
        self.flush_count = 0

    def open(self, *args, **kwargs):
        self.inner.open(*args, **kwargs)
        return self

    def write_frames(self, coords, box=None):
        self.inner.write_frames(coords, box)

    def close(self):
        self.inner.close()

    def flush(self):
        self.flush_count += 1
        self.inner.flush()


def _make_replica(tmp: Path, label: int, n_frames: int = 3):
    top = md.Topology()
    chain = top.add_chain()
    residue = top.add_residue("ALA", chain)
    top.add_atom("CA", md.element.carbon, residue)
    xyz = np.zeros((n_frames, 1, 3), dtype=np.float32)
    for i in range(n_frames):
        xyz[i, 0, 0] = i + 100 * label
    traj = md.Trajectory(xyz, top)
    pdb = tmp / f"model_{label}.pdb"
    dcd = tmp / f"replica_{label}.dcd"
    traj[0].save_pdb(pdb)
    traj.save_dcd(dcd)
    return str(pdb), str(dcd)


def test_checkpoint_every_segment_triggers_flush(tmp_path: Path):
    top0, dcd0 = _make_replica(tmp_path, 1, n_frames=2)
    top1, dcd1 = _make_replica(tmp_path, 2, n_frames=2)
    top = top0
    reader = MDTrajReader(topology_path=top)
    out = tmp_path / "out_ckpt.dcd"
    inner = MDTrajDCDWriter(rewrite_threshold=9999)
    writer = CountingWriter(inner).open(str(out), top, overwrite=True)

    plan = DemuxPlan(
        segments=[
            DemuxSegmentPlan(0, 0, dcd0, 0, 1, 1, False),
            DemuxSegmentPlan(1, 1, dcd1, 0, 1, 1, False),
            DemuxSegmentPlan(2, 0, dcd0, 1, 2, 1, False),
        ],
        target_temperature=300.0,
        frames_per_segment=1,
        total_expected_frames=3,
    )

    demux_streaming(
        plan,
        top,
        reader,
        writer,
        fill_policy="repeat",
        progress_callback=None,
        checkpoint_interval_segments=1,
        flush_between_segments=False,
    )
    writer.close()
    # Expect a flush after each segment
    assert writer.flush_count >= 3
    # Validate final output has all frames
    traj = md.load(str(out), top=top)
    assert traj.n_frames == 3
