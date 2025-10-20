from __future__ import annotations

from typing import Dict, Iterator, List, Optional

import numpy as np
import pytest

from pmarlo.demultiplexing.demux_engine import demux_streaming
from pmarlo.demultiplexing.demux_plan import DemuxPlan, DemuxSegmentPlan
from pmarlo.io.trajectory_writer import TrajectoryWriteError


class _DictTrajectoryReader:
    """In-memory trajectory reader for deterministic unit tests."""

    def __init__(self, frames: Dict[str, np.ndarray]):
        self._frames = {k: np.asarray(v) for k, v in frames.items()}

    def iter_frames(self, path: str, start: int, stop: int, stride: int = 1) -> Iterator[np.ndarray]:
        data = self._frames.get(path)
        if data is None:
            return iter(())
        start = max(0, int(start))
        stop = max(start, min(int(stop), data.shape[0]))
        stride = 1 if stride <= 0 else int(stride)
        return (
            np.array(data[idx], copy=True)
            for idx in range(start, stop, stride)
        )

    def probe_length(self, path: str) -> int:
        arr = self._frames.get(path)
        return 0 if arr is None else int(arr.shape[0])


class _RecordingWriter:
    """Minimal writer that records frames for assertions."""

    def __init__(self) -> None:
        self.opened = False
        self.path: Optional[str] = None
        self.topology: Optional[str] = None
        self.frames: List[np.ndarray] = []
        self.flush_calls = 0

    def open(self, path: str, topology_path: str | None, overwrite: bool = False) -> "_RecordingWriter":
        self.opened = True
        self.path = path
        self.topology = topology_path
        return self

    def write_frames(self, coords: np.ndarray, box: np.ndarray | None = None) -> None:
        if not self.opened:
            raise TrajectoryWriteError("Writer is not open")
        arr = np.asarray(coords)
        if arr.ndim != 3:
            raise TrajectoryWriteError(
                f"coords must have shape (n_frames, n_atoms, 3); got {arr.shape}"
            )
        for frame in arr:
            self.frames.append(np.array(frame, copy=True))

    def flush(self) -> None:
        self.flush_calls += 1

    def close(self) -> None:
        self.opened = False


def _make_plan(segments: List[DemuxSegmentPlan]) -> DemuxPlan:
    total = sum(int(seg.expected_frames) for seg in segments)
    return DemuxPlan(
        segments=segments,
        target_temperature=300.0,
        frames_per_segment=0,
        total_expected_frames=total,
    )


def _frame(value: float) -> np.ndarray:
    return np.full((1, 1, 3), float(value), dtype=np.float32)


def _expected_frame(value: float) -> np.ndarray:
    return np.full((1, 3), float(value), dtype=np.float32)


def test_demux_streaming_repeat_fill_records_segment_counts() -> None:
    reader = _DictTrajectoryReader(
        {
            "replica0": np.concatenate([_frame(0.0), _frame(1.0)], axis=0),
            "replica1": _frame(10.0),
        }
    )
    writer = _RecordingWriter().open("unused.dcd", topology_path=None)

    plan = _make_plan(
        [
            DemuxSegmentPlan(0, 0, "replica0", 0, 2, 2, False),
            DemuxSegmentPlan(1, 1, "replica1", 0, 3, 3, True),
        ]
    )

    result = demux_streaming(
        plan,
        topology_path=None,
        reader=reader,
        writer=writer,
        fill_policy="repeat",
        chunk_size=2,
    )

    written = np.stack(writer.frames, axis=0)
    assert written.shape == (5, 1, 3)
    # First two frames come from replica0
    assert np.allclose(written[0], _expected_frame(0.0))
    assert np.allclose(written[1], _expected_frame(1.0))
    # Remaining three frames repeat the last real frame from replica1
    assert np.allclose(written[2], _expected_frame(10.0))
    assert np.allclose(written[3], _expected_frame(10.0))
    assert np.allclose(written[4], _expected_frame(10.0))

    assert result.total_frames_written == 5
    assert result.segment_real_frames == [2, 1]
    assert result.repaired_segments == [1]
    assert result.skipped_segments == []
    assert not result.warnings


def test_demux_streaming_skip_policy_tracks_skipped_segments() -> None:
    reader = _DictTrajectoryReader({})
    writer = _RecordingWriter().open("unused.dcd", topology_path=None)

    plan = _make_plan(
        [
            DemuxSegmentPlan(0, -1, "", 0, 0, 2, True),
        ]
    )

    result = demux_streaming(
        plan,
        topology_path=None,
        reader=reader,
        writer=writer,
        fill_policy="skip",
    )

    assert writer.frames == []
    assert result.total_frames_written == 0
    assert result.segment_real_frames == [0]
    assert result.skipped_segments and all(idx == 0 for idx in result.skipped_segments)
    assert result.repaired_segments == []
    assert result.warnings, "Expected warning recorded for skipped segment"


def test_demux_streaming_interpolate_uses_next_segment_frame() -> None:
    reader = _DictTrajectoryReader(
        {
            "seg0": _frame(0.0),
            "seg1": _frame(10.0),
            "seg2": _frame(20.0),
        }
    )
    writer = _RecordingWriter().open("unused.dcd", topology_path=None)

    plan = _make_plan(
        [
            DemuxSegmentPlan(0, 0, "seg0", 0, 1, 1, False),
            DemuxSegmentPlan(1, 1, "seg1", 0, 3, 3, True),
            DemuxSegmentPlan(2, 2, "seg2", 0, 1, 1, False),
        ]
    )

    result = demux_streaming(
        plan,
        topology_path=None,
        reader=reader,
        writer=writer,
        fill_policy="interpolate",
    )

    values = [float(frame[0, 0]) for frame in writer.frames]
    assert values == pytest.approx([0.0, 10.0, 13.333333, 16.666667, 20.0])
    assert result.total_frames_written == 5
    assert result.segment_real_frames == [1, 1, 1]
    assert result.repaired_segments == [1]
    assert result.skipped_segments == []
