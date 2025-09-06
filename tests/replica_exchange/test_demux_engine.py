from __future__ import annotations

from pathlib import Path

import mdtraj as md
import numpy as np

from pmarlo.io.trajectory_reader import MDTrajReader
from pmarlo.io.trajectory_writer import MDTrajDCDWriter
from pmarlo.demultiplexing.demux_engine import demux_streaming
from pmarlo.demultiplexing.demux_plan import DemuxPlan, DemuxSegmentPlan


def _make_replica_traj(tmp: Path, label: int, n_frames: int = 4, n_atoms: int = 2):
    top = md.Topology()
    chain = top.add_chain()
    residue = top.add_residue("ALA", chain)
    for _ in range(n_atoms):
        top.add_atom("C", md.element.carbon, residue)
    xyz = np.zeros((n_frames, n_atoms, 3), dtype=np.float32)
    # Embed trajectory index and replica label for verification
    for i in range(n_frames):
        xyz[i, :, 0] = i + 100 * label
        xyz[i, :, 1] = i + 10 + 100 * label
    traj = md.Trajectory(xyz, top)
    pdb = tmp / f"replica_{label}_top.pdb"
    dcd = tmp / f"replica_{label}.dcd"
    traj[0].save_pdb(pdb)
    traj.save_dcd(dcd)
    return str(pdb), str(dcd)


def _read_out(path: str, top_path: str):
    r = MDTrajReader(topology_path=top_path)
    n = r.probe_length(path)
    frames = list(r.iter_frames(path, start=0, stop=n, stride=1))
    return frames


def test_demux_engine_normal(tmp_path: Path):
    top0, dcd0 = _make_replica_traj(tmp_path, label=1, n_frames=4)
    top1, dcd1 = _make_replica_traj(tmp_path, label=2, n_frames=4)

    plan = DemuxPlan(
        segments=[
            DemuxSegmentPlan(0, 0, dcd0, 0, 2, 2, False),
            DemuxSegmentPlan(1, 1, dcd1, 0, 2, 2, False),
        ],
        target_temperature=300.0,
        frames_per_segment=2,
        total_expected_frames=4,
    )

    out = tmp_path / "out_normal.dcd"
    writer = MDTrajDCDWriter(rewrite_threshold=10).open(str(out), top0, overwrite=True)
    reader = MDTrajReader(topology_path=top0)
    res = demux_streaming(plan, top0, reader, writer, fill_policy="repeat")
    writer.close()

    assert res.total_frames_written == 4
    frames = _read_out(str(out), top0)
    # Values should be [100,101,200,201] on x channel
    xs = [f[0, 0] for f in frames]  # first atom x
    assert np.isclose(xs[0], 100)
    assert np.isclose(xs[1], 101)
    assert np.isclose(xs[2], 200)
    assert np.isclose(xs[3], 201)
    assert not res.repaired_segments and not res.skipped_segments


def test_demux_engine_partial_and_fill(tmp_path: Path):
    top0, dcd0 = _make_replica_traj(tmp_path, label=1, n_frames=3)
    top1, dcd1 = _make_replica_traj(tmp_path, label=2, n_frames=1)

    plan = DemuxPlan(
        segments=[
            DemuxSegmentPlan(0, 0, dcd0, 0, 3, 3, False),
            DemuxSegmentPlan(1, 1, dcd1, 0, 1, 3, True),  # need 2 fills
        ],
        target_temperature=300.0,
        frames_per_segment=0,
        total_expected_frames=6,
    )

    out = tmp_path / "out_partial.dcd"
    writer = MDTrajDCDWriter(rewrite_threshold=4).open(str(out), top0, overwrite=True)
    reader = MDTrajReader(topology_path=top0)
    res = demux_streaming(plan, top0, reader, writer, fill_policy="repeat")
    writer.close()

    frames = _read_out(str(out), top0)
    # Expect 3 + (1 real + 2 repeats of last real from segment 1) = 6 frames total
    assert len(frames) == 6
    # Segment 0 x: 100, 101, 102
    assert np.isclose(frames[0][0, 0], 100)
    assert np.isclose(frames[1][0, 0], 101)
    assert np.isclose(frames[2][0, 0], 102)
    # Segment 1: first real 200 then repeats 200, 200
    assert np.isclose(frames[3][0, 0], 200)
    assert np.isclose(frames[4][0, 0], 200)
    assert np.isclose(frames[5][0, 0], 200)
    assert 1 in res.repaired_segments


def test_demux_engine_missing_first_segment_skip(tmp_path: Path):
    top0, dcd0 = _make_replica_traj(tmp_path, label=1, n_frames=2)
    # First segment has no source
    plan = DemuxPlan(
        segments=[
            DemuxSegmentPlan(0, -1, "", 0, 0, 2, True),
            DemuxSegmentPlan(1, 0, dcd0, 0, 2, 2, False),
        ],
        target_temperature=300.0,
        frames_per_segment=0,
        total_expected_frames=4,
    )
    out = tmp_path / "out_missing_first.dcd"
    writer = MDTrajDCDWriter(rewrite_threshold=4).open(str(out), top0, overwrite=True)
    reader = MDTrajReader(topology_path=top0)
    res = demux_streaming(plan, top0, reader, writer, fill_policy="repeat")
    writer.close()

    frames = _read_out(str(out), top0)
    # Only second segment should be written (2 frames)
    assert len(frames) == 2
    assert 0 in res.skipped_segments


def test_demux_engine_interpolate_between_segments(tmp_path: Path):
    top0, dcd0 = _make_replica_traj(tmp_path, label=1, n_frames=1)
    top1, dcd1 = _make_replica_traj(tmp_path, label=2, n_frames=2)

    # Segment 0 has only 1 frame but expects 2 -> interpolate 1 frame between last (100) and next seg first (200)
    plan = DemuxPlan(
        segments=[
            DemuxSegmentPlan(0, 0, dcd0, 0, 1, 2, True),
            DemuxSegmentPlan(1, 1, dcd1, 0, 2, 2, False),
        ],
        target_temperature=300.0,
        frames_per_segment=0,
        total_expected_frames=4,
    )
    out = tmp_path / "out_interp.dcd"
    writer = MDTrajDCDWriter(rewrite_threshold=2).open(str(out), top0, overwrite=True)
    reader = MDTrajReader(topology_path=top0)
    res = demux_streaming(plan, top0, reader, writer, fill_policy="interpolate")
    writer.close()

    frames = _read_out(str(out), top0)
    # Expected sequence: 100 (real), 150 (interp), 200 (next seg real 0), 201 (next seg real 1)
    assert len(frames) == 4
    xs = [f[0, 0] for f in frames]
    assert np.isclose(xs[0], 100)
    assert np.isclose(xs[1], (100 + 200) / 2.0)
    assert np.isclose(xs[2], 200)
    assert np.isclose(xs[3], 201)
    assert 0 in res.repaired_segments
