"""Streaming demux engine that builds output one segment at a time.

Consumes a :class:`DemuxPlan` and uses the streaming reader/writer abstractions
to construct a demultiplexed trajectory with minimal memory usage.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional, Tuple

import numpy as np

from ..io.trajectory_reader import TrajectoryIOError, TrajectoryReader
from ..io.trajectory_writer import TrajectoryWriteError, TrajectoryWriter
from ..progress import ProgressCB, ProgressReporter
from ..utils.errors import DemuxWriterError
from .demux_plan import DemuxPlan, DemuxSegmentPlan

logger = logging.getLogger("pmarlo")


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


def _repeat_frames(frame: np.ndarray, count: int) -> np.ndarray:
    # frame: (n_atoms, 3) -> (count, n_atoms, 3)
    if count <= 0:
        return np.zeros((0,) + frame.shape, dtype=frame.dtype)
    return np.broadcast_to(frame, (count,) + frame.shape).copy()


def _interpolate_frames(last: np.ndarray, nxt: np.ndarray, count: int) -> np.ndarray:
    if count <= 0:
        return np.zeros((0,) + last.shape, dtype=last.dtype)
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
    try:
        it = reader.iter_frames(
            nxt.source_path,
            start=int(nxt.start_frame),
            stop=int(nxt.start_frame + 1),
            stride=1,
        )
        for f in it:
            return f
    except Exception:
        return None
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
        return _np.concatenate(acc, axis=0)
    return _np.zeros((0, 0, 3), dtype=_np.float32)


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
    try:
        for s in plan.segments:
            p = Path(getattr(s, "source_path", ""))
            if not str(p):
                continue
            cand = p.with_suffix(".pdb")
            if cand.exists():
                candidates.append(str(cand))
    except Exception:
        return requested
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
    """Stream and write demultiplexed frames according to a plan.

    Parameters
    ----------
    plan : DemuxPlan
        Validated plan with per-segment frame windows and expected frame counts.
    topology_path : str or None
        Topology path for backends that require it. The provided ``reader``
        already encapsulates any backend requirement; this parameter is kept
        for symmetry and parallel workers.
    reader : TrajectoryReader
        Streaming reader abstraction used to fetch frames from source files.
    writer : TrajectoryWriter
        Append-like writer abstraction used to persist frames in order. Must be
        opened by the caller and will not be closed by this function.
    fill_policy : {"repeat", "skip", "interpolate"}, optional
        How to handle missing frames per segment. ``"repeat"`` duplicates the
        last written frame; ``"skip"`` omits missing frames; ``"interpolate"``
        attempts linear interpolation between the last written and the first
        available frame of the next segment (falls back to repeat).
    parallel_read_workers : int or None, optional
        If >1, use a process pool to read segments concurrently while writing in
        order. ``None`` (default) reads sequentially.

    Returns
    -------
    DemuxResult
        Summary including total frames written, segments repaired/skipped,
        warnings, and per-segment real frame counts.

    Examples
    --------
    Minimal in-memory example using fake reader/writer (no I/O)::

        class FakeReader:
            def iter_frames(self, path, start, stop, stride=1):
                import numpy as np
                for i in range(start, stop):
                    yield np.zeros((1, 3), dtype=float)
            def probe_length(self, path):
                return 10

        class CollectWriter:
            def __init__(self):
                self.frames = []
            def open(self, *a, **k):
                return self
            def write_frames(self, coords, box=None):
                self.frames.append(coords)
            def close(self):
                pass

        res = demux_streaming(plan, None, FakeReader(), CollectWriter().open("", None, True))
        assert res.total_frames_written >= 0
    """

    total_written = 0
    repaired_segments: List[int] = []
    skipped_segments: List[int] = []
    warnings: List[str] = []

    last_written_frame: Optional[np.ndarray] = None

    def _flush_accum(acc: List[np.ndarray]) -> int:
        if not acc:
            return 0
        try:
            batch = np.stack(acc, axis=0)
            try:
                writer.write_frames(batch)
            except TrajectoryWriteError as exc:
                raise DemuxWriterError(
                    f"Writer failed when flushing batch of {batch.shape[0]} frame(s)"
                ) from exc
            return int(batch.shape[0])
        finally:
            acc.clear()

    def _consume_segment(
        i: int, seg: DemuxSegmentPlan, arr: Optional[np.ndarray]
    ) -> None:
        nonlocal total_written, last_written_frame
        planned = int(max(0, seg.expected_frames))
        got = (
            int(arr.shape[0])
            if (arr is not None and arr.size > 0 and arr.ndim == 3)
            else 0
        )
        # Write available frames in chunks
        if got > 0 and arr is not None:
            chunk = 1024
            for ofs in range(0, got, chunk):
                end = min(got, ofs + chunk)
                try:
                    writer.write_frames(arr[ofs:end])
                except TrajectoryWriteError as exc:
                    raise DemuxWriterError(
                        f"Writer failed when writing segment {i} frames {ofs}:{end}"
                    ) from exc
                total_written += end - ofs
                last_written_frame = np.array(arr[end - 1], copy=True)
        missing = max(0, planned - got)
        if missing > 0:
            if fill_policy == "skip":
                warnings.append(
                    f"Segment {i} missing {missing} frame(s); skipping due to policy=skip"
                )
                skipped_segments.append(i)
                return
            if last_written_frame is None:
                warnings.append(
                    f"Segment {i} has no frames and cannot fill; skipping {missing}"
                )
                skipped_segments.append(i)
                return
            if fill_policy == "interpolate":
                nxt = _peek_next_first_frame(plan, i, reader)
                if nxt is not None:
                    fill = _interpolate_frames(last_written_frame, nxt, missing)
                else:
                    warnings.append(
                        f"Segment {i} cannot interpolate (no next frame); repeating last frame for {missing}"
                    )
                    fill = _repeat_frames(last_written_frame, missing)
            else:
                fill = _repeat_frames(last_written_frame, missing)
            try:
                writer.write_frames(fill)
            except TrajectoryWriteError as exc:
                raise DemuxWriterError(
                    f"Writer failed when filling {missing} frame(s) at segment {i}"
                ) from exc
            last_written_frame = np.array(fill[-1], copy=True)
            total_written += int(fill.shape[0])
            repaired_segments.append(i)

    seg_real_counts: List[int] = [0 for _ in plan.segments]
    n_segments = len(plan.segments)
    reporter = ProgressReporter(progress_callback)
    reporter.emit(
        "demux_begin",
        {"segments": int(n_segments), "current": 0, "total": int(max(1, n_segments))},
    )
    # Normalize topology path if not provided
    topology_path = _canonical_topology_path(topology_path, plan)

    # Parallel path: read segments concurrently but write in order
    if parallel_read_workers is not None and int(parallel_read_workers) > 1:
        import concurrent.futures as _fut

        max_workers = max(1, int(parallel_read_workers))
        expected_idx = 0
        results: dict[int, Optional[np.ndarray]] = {}
        pending: dict[_fut.Future, int] = {}
        window = max_workers * 2

        def _drain_ready() -> None:
            nonlocal expected_idx
            while expected_idx in results:
                arr = results.pop(expected_idx)
                # consume and emit per-segment progress
                _consume_segment(expected_idx, plan.segments[expected_idx], arr)
                # Optional forced flush between segments or checkpoints
                if flush_between_segments:
                    try:
                        writer.flush()
                    except Exception:
                        pass
                if (
                    checkpoint_interval_segments
                    and (expected_idx + 1) % int(checkpoint_interval_segments) == 0
                ):
                    try:
                        writer.flush()
                    except Exception:
                        pass
                frames_written = int(
                    (arr.shape[0] if isinstance(arr, np.ndarray) else 0)
                )
                # Include fills when policy is not skip
                exp = int(plan.segments[expected_idx].expected_frames)
                if frames_written < exp and fill_policy != "skip":
                    frames_written = exp
                reporter.emit(
                    "demux_segment",
                    {
                        "index": int(expected_idx),
                        "frames": int(frames_written),
                        "current": int(expected_idx + 1),
                        "total": int(max(1, n_segments)),
                    },
                )
                expected_idx += 1

        with _fut.ProcessPoolExecutor(max_workers=max_workers) as ex:
            for i, seg in enumerate(plan.segments):
                planned = int(max(0, seg.expected_frames))
                have = int(max(0, seg.stop_frame - seg.start_frame))

                if seg.replica_index < 0 or not seg.source_path or have <= 0:
                    # No source for this segment
                    results[i] = None if planned > 0 else np.zeros((0, 0, 3))
                    _drain_ready()
                    continue

                # Backpressure on submissions
                while len(pending) >= window:
                    done_iter = _fut.as_completed(list(pending.keys()), timeout=None)
                    fut = next(done_iter)
                    seg_idx = pending.pop(fut)
                    try:
                        arr = fut.result()
                    except Exception as exc:  # worker failed; treat as missing
                        msg = (
                            f"Segment {seg_idx} parallel read error for replica={plan.segments[seg_idx].replica_index} "
                            f"path={plan.segments[seg_idx].source_path} window=[{plan.segments[seg_idx].start_frame},"
                            f"{plan.segments[seg_idx].stop_frame})]: {exc}"
                        )
                        logger.warning(msg)
                        warnings.append(msg)
                        arr = None
                    # Record real frame count if result present
                    if arr is not None and isinstance(arr, np.ndarray):
                        seg_real_counts[seg_idx] = int(arr.shape[0])
                    results[seg_idx] = arr
                    _drain_ready()

                fut = ex.submit(
                    _read_segment_frames_worker,
                    seg.source_path,
                    int(seg.start_frame),
                    int(seg.stop_frame),
                    1,
                    topology_path,
                )
                pending[fut] = i

            # Drain remaining
            for fut in _fut.as_completed(list(pending.keys())):
                seg_idx = pending.pop(fut)
                try:
                    arr = fut.result()
                except Exception as exc:
                    msg = (
                        f"Segment {seg_idx} parallel read error for replica={plan.segments[seg_idx].replica_index} "
                        f"path={plan.segments[seg_idx].source_path} window=[{plan.segments[seg_idx].start_frame},"
                        f"{plan.segments[seg_idx].stop_frame})]: {exc}"
                    )
                    logger.warning(msg)
                    warnings.append(msg)
                    arr = None
                if arr is not None and isinstance(arr, np.ndarray):
                    seg_real_counts[seg_idx] = int(arr.shape[0])
                results[seg_idx] = arr
                _drain_ready()

        # Finalize
        try:
            writer.flush()
        except Exception:
            pass
        reporter.emit(
            "demux_end",
            {
                "frames": int(total_written),
                "repaired": int(len(repaired_segments)),
                "skipped": int(len(skipped_segments)),
                "current": int(n_segments),
                "total": int(max(1, n_segments)),
            },
        )
        return DemuxResult(
            total_frames_written=int(total_written),
            repaired_segments=repaired_segments,
            skipped_segments=skipped_segments,
            warnings=warnings,
            segment_real_frames=seg_real_counts,
        )

    # Sequential path
    for i, seg in enumerate(plan.segments):
        planned = int(max(0, seg.expected_frames))
        have = int(max(0, seg.stop_frame - seg.start_frame))

        if seg.replica_index < 0 or not seg.source_path or have <= 0:
            # Entire segment must be filled or skipped
            if planned == 0:
                continue
            if fill_policy == "skip":
                msg = f"Segment {i} has no source frames; skipping {planned} planned frame(s)"
                logger.warning(msg)
                warnings.append(msg)
                skipped_segments.append(i)
                continue
            if last_written_frame is None:
                msg = f"Segment {i} lacks source and no previous frame to repeat; skipping {planned} frame(s)"
                logger.warning(msg)
                warnings.append(msg)
                skipped_segments.append(i)
                continue
            # repeat or interpolate with no next available -> repeat
            if fill_policy == "interpolate":
                nxt = _peek_next_first_frame(plan, i, reader)
                if nxt is not None:
                    fill = _interpolate_frames(last_written_frame, nxt, planned)
                else:
                    msg = f"Segment {i} cannot interpolate (no next frame); repeating last frame for {planned}"
                    logger.warning(msg)
                    warnings.append(msg)
                    fill = _repeat_frames(last_written_frame, planned)
            else:
                fill = _repeat_frames(last_written_frame, planned)
            writer.write_frames(fill)
            last_written_frame = np.array(fill[-1], copy=True)
            total_written += planned
            repaired_segments.append(i)
            # Emit per-segment progress (skipped or filled from previous)
            reporter.emit(
                "demux_segment",
                {
                    "index": int(i),
                    "frames": int(0 if fill_policy == "skip" else planned),
                    "current": int(i + 1),
                    "total": int(max(1, n_segments)),
                },
            )
            continue

        # Stream available frames
        acc: List[np.ndarray] = []
        got = 0
        try:
            for xyz in reader.iter_frames(
                seg.source_path,
                start=int(seg.start_frame),
                stop=int(seg.stop_frame),
                stride=1,
            ):
                acc.append(np.asarray(xyz))
                got += 1
                # Write in moderate batches to limit memory; ~1k frames is typical
                if len(acc) >= 1024:
                    total_written += _flush_accum(acc)
                last_written_frame = np.array(xyz, copy=True)
        except TrajectoryIOError as exc:
            msg = (
                f"Segment {i} read error for replica={seg.replica_index} path={seg.source_path} "
                f"window=[{seg.start_frame},{seg.stop_frame}): {exc}; will fill remaining if policy allows"
            )
            logger.warning(msg)
            warnings.append(msg)
        finally:
            total_written += _flush_accum(acc)
            seg_real_counts[i] = int(got)

        # Fill remainder if needed
        missing = max(0, planned - got)
        if missing > 0:
            if fill_policy == "skip":
                msg = f"Segment {i} missing {missing} frame(s); skipping due to policy=skip"
                logger.warning(msg)
                warnings.append(msg)
                skipped_segments.append(i)
                reporter.emit(
                    "demux_segment",
                    {
                        "index": int(i),
                        "frames": int(got),
                        "current": int(i + 1),
                        "total": int(max(1, n_segments)),
                    },
                )
                continue
            if last_written_frame is None:
                msg = f"Segment {i} has no frames and cannot fill; skipping {missing}"
                logger.warning(msg)
                warnings.append(msg)
                skipped_segments.append(i)
                continue
            if fill_policy == "interpolate":
                nxt = _peek_next_first_frame(plan, i, reader)
                if nxt is not None:
                    fill = _interpolate_frames(last_written_frame, nxt, missing)
                else:
                    msg = f"Segment {i} cannot interpolate (no next frame); repeating last frame for {missing}"
                    logger.warning(msg)
                    warnings.append(msg)
                    fill = _repeat_frames(last_written_frame, missing)
            else:  # repeat
                fill = _repeat_frames(last_written_frame, missing)
            try:
                writer.write_frames(fill)
            except TrajectoryWriteError as exc:
                raise DemuxWriterError(
                    f"Writer failed when filling {missing} frame(s) at segment {i}"
                ) from exc
            last_written_frame = np.array(fill[-1], copy=True)
            total_written += int(fill.shape[0])
            repaired_segments.append(i)
        # Emit per-segment completion with frames written (real + fills when not skip)
        written_now = int(planned if fill_policy != "skip" else got)
        reporter.emit(
            "demux_segment",
            {
                "index": int(i),
                "frames": int(written_now),
                "current": int(i + 1),
                "total": int(max(1, n_segments)),
            },
        )
        # Optional flush controls
        if flush_between_segments:
            try:
                writer.flush()
            except Exception:
                pass
        if (
            checkpoint_interval_segments
            and (i + 1) % int(checkpoint_interval_segments) == 0
        ):
            try:
                writer.flush()
            except Exception:
                pass

    # Finalize
    try:
        writer.flush()
    except Exception:
        pass
    reporter.emit(
        "demux_end",
        {
            "frames": int(total_written),
            "repaired": int(len(repaired_segments)),
            "skipped": int(len(skipped_segments)),
            "current": int(n_segments),
            "total": int(max(1, n_segments)),
        },
    )
    return DemuxResult(
        total_frames_written=int(total_written),
        repaired_segments=repaired_segments,
        skipped_segments=skipped_segments,
        warnings=warnings,
        segment_real_frames=seg_real_counts,
    )


