"""Streaming demux engine that builds output one segment at a time.

Consumes a :class:`DemuxPlan` and uses the streaming reader/writer abstractions
to construct a demultiplexed trajectory with minimal memory usage.
"""

from __future__ import annotations

import logging
from concurrent.futures import Future
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional

import numpy as np

from ..io.trajectory_reader import TrajectoryIOError, TrajectoryReader
from ..io.trajectory_writer import TrajectoryWriteError, TrajectoryWriter
from ..transform.progress import ProgressCB, ProgressReporter
from ..utils.errors import DemuxWriterError
from .demux_plan import DemuxPlan, DemuxSegmentPlan

logger = logging.getLogger("pmarlo")


FillPolicy = Literal["repeat", "skip", "interpolate"]


@dataclass
class DemuxResult:
    """Outcome of streaming demultiplexing.

    Attributes
    ----------
    total_frames_written : int
        Total number of frames written to the output trajectory.
    repaired_segments : list[int]
        Indices of segments where gap filling or interpolation occurred.
    skipped_segments : list[int]
        Indices of segments that were skipped entirely (e.g., first segment with
        no prior frame to repeat and policy not permitting fill).
    warnings : list[str]
        Human-readable warnings encountered during demux.
    """

    total_frames_written: int
    repaired_segments: List[int] = field(default_factory=list)
    skipped_segments: List[int] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    # Number of real (non-filled) frames obtained per segment, aligned with plan.segments
    segment_real_frames: List[int] = field(default_factory=list)


@dataclass
class _DemuxContext:
    plan: DemuxPlan
    reader: TrajectoryReader
    writer: TrajectoryWriter
    fill_policy: FillPolicy
    checkpoint_interval_segments: Optional[int]
    flush_between_segments: bool
    reporter: ProgressReporter
    topology_path: str | None


@dataclass
class _DemuxState:
    total_written: int = 0
    last_written_frame: Optional[np.ndarray] = None
    repaired_segments: List[int] = field(default_factory=list)
    skipped_segments: List[int] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    segment_real_frames: List[int] = field(default_factory=list)


def _repeat_frames(frame: np.ndarray, count: int) -> np.ndarray:
    # frame: (n_atoms, 3) -> (count, n_atoms, 3)
    if count <= 0:
        return np.empty((0,) + frame.shape, dtype=frame.dtype)
    return np.repeat(frame[np.newaxis, ...], count, axis=0)


def _interpolate_frames(last: np.ndarray, nxt: np.ndarray, count: int) -> np.ndarray:
    if count <= 0:
        return np.empty((0,) + last.shape, dtype=last.dtype)
    # create linear interpolation excluding endpoints
    # t in (1/(count+1), ..., count/(count+1))
    t_vals = (np.arange(1, count + 1, dtype=np.float32) / float(count + 1)).reshape(
        (-1, 1, 1)
    )
    return (1.0 - t_vals) * last[np.newaxis, ...] + t_vals * nxt[np.newaxis, ...]


def _peek_next_first_frame(
    plan: DemuxPlan, idx: int, reader: TrajectoryReader
) -> Optional[np.ndarray]:
    # Return the first available frame of the next segment if readable.
    if idx + 1 >= len(plan.segments):
        return None
    nxt = plan.segments[idx + 1]
    if (
        nxt.replica_index < 0
        or nxt.stop_frame <= nxt.start_frame
        or not nxt.source_path
    ):
        return None
    it = reader.iter_frames(
        nxt.source_path,
        start=int(nxt.start_frame),
        stop=int(nxt.start_frame + 1),
        stride=1,
    )
    for f in it:
        return f
    return None


def _read_segment_frames_worker(
    path: str, start: int, stop: int, stride: int, topology_path: str | None
) -> np.ndarray:
    """Worker function to read a segment's frames as a single ndarray.

    Returns an array of shape (n_frames, n_atoms, 3). Returns an empty
    array with shape (0, 0, 3) when no frames are available.
    """
    import numpy as _np

    from pmarlo.io.trajectory_reader import MDTrajReader as _MDTR

    rdr = _MDTR(topology_path=topology_path)
    acc: list[_np.ndarray] = []
    for xyz in rdr.iter_frames(
        path, start=int(start), stop=int(stop), stride=int(stride)
    ):
        acc.append(_np.asarray(xyz))
    if acc:
        return _np.stack(acc, axis=0)
    return _np.empty((0, 0, 3), dtype=_np.float32)


def _canonical_topology_path(requested: str | None, plan: DemuxPlan) -> str | None:
    """Pick a deterministic topology path when not provided.

    If ``requested`` is not None, return it. Otherwise, attempt to derive a
    candidate PDB by replacing the suffix of segment source paths with ``.pdb``
    and pick the lexicographically smallest existing file. If none exist,
    return ``requested`` (None).
    """
    if requested:
        return requested
    candidates: list[str] = []
    for s in plan.segments:
        source_path = s.source_path
        if not source_path:
            continue
        cand = Path(source_path).with_suffix(".pdb")
        if cand.exists():
            candidates.append(str(cand))
    return min(candidates) if candidates else requested


def demux_streaming(
    plan: DemuxPlan,
    topology_path: str | None,
    reader: TrajectoryReader,
    writer: TrajectoryWriter,
    *,
    fill_policy: Literal["repeat", "skip", "interpolate"] = "repeat",
    checkpoint_interval_segments: int | None = None,
    flush_between_segments: bool = False,
    parallel_read_workers: int | None = None,
    chunk_size: int = 1000,
    progress_callback: Optional[ProgressCB] = None,
) -> DemuxResult:
    """Stream and write demultiplexed frames according to a plan."""

    reporter = ProgressReporter(progress_callback)
    total_segments = len(plan.segments)
    reporter.emit(
        "demux_begin",
        {
            "segments": int(total_segments),
            "current": 0,
            "total": int(max(1, total_segments)),
        },
    )

    normalized_topology = _canonical_topology_path(topology_path, plan)
    context = _DemuxContext(
        plan=plan,
        reader=reader,
        writer=writer,
        fill_policy=fill_policy,
        checkpoint_interval_segments=checkpoint_interval_segments,
        flush_between_segments=flush_between_segments,
        reporter=reporter,
        topology_path=normalized_topology,
    )
    state = _DemuxState(segment_real_frames=[0 for _ in plan.segments])

    if parallel_read_workers is not None and int(parallel_read_workers) > 1:
        _demux_parallel(
            context,
            state,
            max_workers=max(1, int(parallel_read_workers)),
            window_multiplier=2,
        )
    else:
        _demux_sequential(context, state, max(1, int(chunk_size)))

    _finalize_demux(context, state)
    return DemuxResult(
        total_frames_written=int(state.total_written),
        repaired_segments=state.repaired_segments,
        skipped_segments=state.skipped_segments,
        warnings=state.warnings,
        segment_real_frames=state.segment_real_frames,
    )


def _demux_sequential(
    context: _DemuxContext, state: _DemuxState, write_chunk: int
) -> None:
    for index, segment in enumerate(context.plan.segments):
        planned = max(0, int(segment.expected_frames))

        handled = _handle_missing_source_segment(
            context,
            state,
            index,
            segment,
            planned,
        )
        if handled is not None:
            if handled >= 0:
                _emit_segment_progress(context, index, handled)
                _flush_after_segment(context, index)
            continue

        got = _stream_segment_frames(context, state, index, segment, write_chunk)
        state.segment_real_frames[index] = got
        frames_written = _handle_post_read_gap(
            context,
            state,
            index,
            segment,
            planned,
            got,
        )
        if frames_written is not None:
            _emit_segment_progress(context, index, frames_written)
            _flush_after_segment(context, index)


def _demux_parallel(
    context: _DemuxContext,
    state: _DemuxState,
    *,
    max_workers: int,
    window_multiplier: int,
) -> None:
    _ParallelDemuxer(context, state, max_workers, window_multiplier).run()


class _ParallelDemuxer:
    def __init__(
        self,
        context: _DemuxContext,
        state: _DemuxState,
        max_workers: int,
        window_multiplier: int,
    ) -> None:
        self.context = context
        self.state = state
        self.plan = context.plan
        self.max_workers = max(1, max_workers)
        self.window = self.max_workers * window_multiplier
        self.next_to_submit = 0
        self.next_to_consume = 0
        self.pending: dict[Future[np.ndarray | None], int] = {}
        self.buffered: dict[int, Optional[np.ndarray]] = {}

    def run(self) -> None:
        import concurrent.futures as fut

        with fut.ProcessPoolExecutor(max_workers=self.max_workers) as pool:
            while self._has_work():
                self._schedule(pool)
                self._drain_ready()
                if not self._has_pending():
                    continue
                self._collect_completed()
                self._drain_ready()

    def _has_work(self) -> bool:
        return self.next_to_consume < len(self.plan.segments)

    def _has_pending(self) -> bool:
        return bool(self.pending) and self._has_work()

    def _schedule(self, pool) -> None:
        while len(self.pending) < self.window and self.next_to_submit < len(
            self.plan.segments
        ):
            seg_idx = self.next_to_submit
            segment = self.plan.segments[seg_idx]
            self.next_to_submit += 1
            if not self._segment_has_source(segment):
                self.buffered[seg_idx] = None
                continue
            future = pool.submit(
                _read_segment_frames_worker,
                segment.source_path,
                int(segment.start_frame),
                int(segment.stop_frame),
                1,
                self.context.topology_path,
            )
            self.pending[future] = seg_idx

    def _segment_has_source(self, segment: DemuxSegmentPlan) -> bool:
        return (
            segment.replica_index >= 0
            and bool(segment.source_path)
            and segment.stop_frame > segment.start_frame
        )

    def _collect_completed(self) -> None:
        import concurrent.futures as fut

        if not self.pending:
            return
        pending_futures = list(self.pending.keys())
        done, _ = fut.wait(pending_futures, return_when=fut.FIRST_COMPLETED)
        for future in done:
            seg_idx = self.pending.pop(future)
            self.buffered[seg_idx] = self._resolve_future_result(seg_idx, future)

    def _resolve_future_result(
        self, seg_idx: int, future: Future[np.ndarray | None]
    ) -> Optional[np.ndarray]:
        arr = future.result()
        return arr if isinstance(arr, np.ndarray) else None

    def _drain_ready(self) -> None:
        while self.next_to_consume in self.buffered:
            frames = self.buffered.pop(self.next_to_consume)
            segment = self.plan.segments[self.next_to_consume]
            planned = max(0, int(segment.expected_frames))
            if frames is not None and isinstance(frames, np.ndarray):
                self.state.segment_real_frames[self.next_to_consume] = int(
                    frames.shape[0]
                )
            progress_frames = _consume_parallel_segment(
                self.context,
                self.state,
                self.next_to_consume,
                segment,
                planned,
                frames,
            )
            _emit_segment_progress(self.context, self.next_to_consume, progress_frames)
            _flush_after_segment(self.context, self.next_to_consume)
            self.next_to_consume += 1


def _finalize_demux(context: _DemuxContext, state: _DemuxState) -> None:
    context.writer.flush()
    total_segments = len(context.plan.segments)
    context.reporter.emit(
        "demux_end",
        {
            "frames": int(state.total_written),
            "repaired": int(len(state.repaired_segments)),
            "skipped": int(len(state.skipped_segments)),
            "current": int(total_segments),
            "total": int(max(1, total_segments)),
        },
    )


def _handle_missing_source_segment(
    context: _DemuxContext,
    state: _DemuxState,
    index: int,
    segment: DemuxSegmentPlan,
    planned: int,
) -> Optional[int]:
    if (
        segment.replica_index >= 0
        and segment.source_path
        and segment.stop_frame > segment.start_frame
    ):
        return None
    if planned <= 0:
        return 0
    if context.fill_policy == "skip":
        msg = (
            f"Segment {index} has no source frames; skipping {planned} planned frame(s)"
        )
        logger.warning(msg)
        state.warnings.append(msg)
        state.skipped_segments.append(index)
        return 0
    if state.last_written_frame is None:
        msg = f"Segment {index} lacks source and no previous frame to repeat; skipping {planned} frame(s)"
        logger.warning(msg)
        state.warnings.append(msg)
        state.skipped_segments.append(index)
        return 0

    fill, warning = _build_fill_frames(
        context,
        state,
        index,
        planned,
        segment,
    )
    if warning:
        logger.warning(warning)
        state.warnings.append(warning)
    _write_frames_safe(
        context.writer,
        fill,
        f"Writer failed when filling source-less segment {index}",
    )
    state.last_written_frame = np.array(fill[-1], copy=True)
    state.total_written += int(fill.shape[0])
    state.repaired_segments.append(index)
    return planned


def _emit_segment_progress(context: _DemuxContext, index: int, frames: int) -> None:
    context.reporter.emit(
        "demux_segment",
        {
            "index": int(index),
            "frames": int(frames),
            "current": int(index + 1),
            "total": int(max(1, len(context.plan.segments))),
        },
    )


def _flush_after_segment(context: _DemuxContext, index: int) -> None:
    if context.flush_between_segments:
        context.writer.flush()
    checkpoint = context.checkpoint_interval_segments
    if checkpoint and (index + 1) % int(checkpoint) == 0:
        context.writer.flush()


def _stream_segment_frames(
    context: _DemuxContext,
    state: _DemuxState,
    index: int,
    segment: DemuxSegmentPlan,
    write_chunk: int,
) -> int:
    if not segment.source_path:
        return 0
    acc: List[np.ndarray] = []
    got = 0
    try:
        for frame in context.reader.iter_frames(
            segment.source_path,
            start=int(segment.start_frame),
            stop=int(segment.stop_frame),
            stride=1,
        ):
            arr = np.asarray(frame)
            acc.append(arr)
            got += 1
            if len(acc) >= write_chunk:
                state.total_written += _flush_batch(
                    context,
                    state,
                    acc,
                    index,
                    "flushing batch",
                )
            state.last_written_frame = np.array(arr, copy=True)
    except TrajectoryIOError as exc:
        state.total_written += _flush_batch(
            context,
            state,
            acc,
            index,
            "flushing batch",
        )
        message = (
            f"Segment {index} reader failed after {got} frame(s): {exc}"
        )
        logger.warning(message)
        state.warnings.append(message)
        return got
    state.total_written += _flush_batch(
        context,
        state,
        acc,
        index,
        "flushing batch",
    )
    return got


def _handle_post_read_gap(
    context: _DemuxContext,
    state: _DemuxState,
    index: int,
    segment: DemuxSegmentPlan,
    planned: int,
    got: int,
) -> Optional[int]:
    missing = max(0, planned - got)
    if missing <= 0:
        return planned if context.fill_policy != "skip" else got
    if context.fill_policy == "skip":
        msg = f"Segment {index} missing {missing} frame(s); skipping due to policy=skip"
        logger.warning(msg)
        state.warnings.append(msg)
        state.skipped_segments.append(index)
        return got
    if state.last_written_frame is None:
        msg = f"Segment {index} has no frames and cannot fill; skipping {missing}"
        logger.warning(msg)
        state.warnings.append(msg)
        state.skipped_segments.append(index)
        return None

    fill, warning = _build_fill_frames(context, state, index, missing, segment)
    if warning:
        logger.warning(warning)
        state.warnings.append(warning)
    _write_frames_safe(
        context.writer,
        fill,
        f"Writer failed when filling {missing} frame(s) at segment {index}",
    )
    state.last_written_frame = np.array(fill[-1], copy=True)
    state.total_written += int(fill.shape[0])
    state.repaired_segments.append(index)
    return planned


def _consume_parallel_segment(
    context: _DemuxContext,
    state: _DemuxState,
    index: int,
    segment: DemuxSegmentPlan,
    planned: int,
    frames: Optional[np.ndarray],
) -> int:
    got = 0
    if frames is not None and isinstance(frames, np.ndarray) and frames.size > 0:
        _write_frames_safe(
            context.writer,
            frames,
            f"Writer failed when writing segment {index} frames",
        )
        got = int(frames.shape[0])
        state.total_written += got
        state.last_written_frame = np.array(frames[-1], copy=True)
    result = _handle_post_read_gap(context, state, index, segment, planned, got)
    return 0 if result is None else result


def _write_frames_safe(
    writer: TrajectoryWriter, frames: np.ndarray, error_message: str
) -> None:
    try:
        writer.write_frames(frames)
    except TrajectoryWriteError as exc:
        raise DemuxWriterError(error_message) from exc


def _flush_batch(
    context: _DemuxContext,
    state: _DemuxState,
    batch: List[np.ndarray],
    index: int,
    action: str,
) -> int:
    if not batch:
        return 0
    stacked = np.stack(batch, axis=0)
    _write_frames_safe(
        context.writer,
        stacked,
        f"Writer failed when {action} for segment {index}",
    )
    written = int(stacked.shape[0])
    state.last_written_frame = np.array(stacked[-1], copy=True)
    batch.clear()
    return written


def _build_fill_frames(
    context: _DemuxContext,
    state: _DemuxState,
    index: int,
    count: int,
    segment: DemuxSegmentPlan,
) -> tuple[np.ndarray, Optional[str]]:
    assert state.last_written_frame is not None
    if context.fill_policy == "interpolate":
        nxt = _peek_next_first_frame(context.plan, index, context.reader)
        if nxt is not None:
            return _interpolate_frames(state.last_written_frame, nxt, count), None
        warning = f"Segment {index} cannot interpolate (no next frame); repeating last frame for {count}"
        return _repeat_frames(state.last_written_frame, count), warning
    return _repeat_frames(state.last_written_frame, count), None
