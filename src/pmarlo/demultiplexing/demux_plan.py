"""Demultiplexing plan builder for REMD trajectories (pure planning layer).

This module constructs a validated, deterministic plan for demultiplexing
trajectories without loading or touching trajectory data. It maps exchange
history and configuration to per‑segment frame slices sourced from replica
files.

Design goals
------------
- Pure and testable: no I/O; inputs are simple types; outputs are dataclasses.
- Deterministic: no random choices; stable with identical inputs.
- Resilient: validates and auto‑corrects minor inconsistencies, logs warnings,
  and never raises for routine issues.

Usage
-----
- Call :func:`build_demux_plan` with exchange history, temperatures, target
  temperature, frequency/stride information, equilibration offset, and per‑
  replica path/length metadata. Receive a :class:`DemuxPlan` with per‑segment
  slicing instructions and global expectations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

logger = logging.getLogger("pmarlo")


@dataclass(frozen=True)
class DemuxSegmentPlan:
    """Plan for a single demultiplexing segment.

    Parameters
    ----------
    segment_index : int
        Index of the segment (0-based) corresponding to an exchange interval.
    replica_index : int
        Replica id that holds frames at the target temperature for this segment.
        When ``needs_fill`` is True due to missing source or frames, this is the
        intended source replica index; ``-1`` indicates no replica was at the
        target temperature for this segment.
    source_path : str
        Filesystem path to the intended source trajectory for ``replica_index``.
        May point to a missing file; planning does not check the filesystem.
    start_frame : int
        Start frame index within ``source_path`` (inclusive), adjusted to avoid
        backward jumps (non-negative).
    stop_frame : int
        Stop frame index within ``source_path`` (exclusive), truncated to the
        available frame count to avoid out-of-range access.
    expected_frames : int
        Planned number of frames for this segment based on MD steps and stride.
        This forms the contribution to the global output timeline length.
    needs_fill : bool, default=False
        True when this segment cannot be fully sourced from ``source_path`` and
        requires gap filling (e.g., repeating/interpolating frames) during demux.
    """

    segment_index: int
    replica_index: int
    source_path: str
    start_frame: int
    stop_frame: int
    expected_frames: int
    needs_fill: bool = False


@dataclass(frozen=True)
class DemuxPlan:
    """Complete demultiplexing plan.

    Attributes
    ----------
    segments : list of DemuxSegmentPlan
        Per-segment slicing instructions.
    target_temperature : float
        Target temperature in Kelvin for demultiplexing.
    frames_per_segment : int
        Common planned frames per segment when consistent across all segments;
        0 when variable due to mixed strides.
    total_expected_frames : int
        Sum of ``expected_frames`` across all segments.
    """

    segments: List[DemuxSegmentPlan]
    target_temperature: float
    frames_per_segment: int
    total_expected_frames: int


def _to_indexed_mapping(
    values: Union[Sequence[int], Mapping[int, int]], default: int = 0
) -> Dict[int, int]:
    if isinstance(values, Mapping):
        return {int(k): int(v) for k, v in values.items()}
    return {i: int(v) for i, v in enumerate(values)}


def _to_path_mapping(values: Union[Sequence[str], Mapping[int, str]]) -> Dict[int, str]:
    if isinstance(values, Mapping):
        return {int(k): str(v) for k, v in values.items()}
    return {i: str(v) for i, v in enumerate(values)}


def _closest_temperature_index(
    temperatures: Sequence[float], target_temperature: float
) -> int:
    import math

    diffs = [abs(float(t) - float(target_temperature)) for t in temperatures]
    best = 0
    best_diff = math.inf
    for i, d in enumerate(diffs):
        if d < best_diff:
            best = i
            best_diff = d
    return int(best)


def build_demux_plan(
    *,
    exchange_history: Sequence[Sequence[int]],
    temperatures: Sequence[float],
    target_temperature: float,
    exchange_frequency: int,
    equilibration_offset: int,
    replica_paths: Union[Sequence[str], Mapping[int, str]],
    replica_frames: Union[Sequence[int], Mapping[int, int]],
    default_stride: int,
    replica_strides: Optional[Union[Sequence[int], Mapping[int, int]]] = None,
) -> DemuxPlan:
    """Build a validated demultiplexing plan without touching trajectory data.

    Parameters
    ----------
    exchange_history : sequence of sequence of int
        For each segment (0..S-1) a sequence of temperature-state indices per
        replica (length R). ``state == k`` means the replica was at
        ``temperatures[k]`` during that segment.
    temperatures : sequence of float
        Temperature schedule (length R). Used to pick the closest index to
        ``target_temperature``.
    target_temperature : float
        Temperature in Kelvin to demultiplex to.
    exchange_frequency : int
        MD steps between exchange attempts; sets the segment duration in MD steps.
    equilibration_offset : int
        MD steps elapsed before production begins; forms the initial time offset.
    replica_paths : sequence[str] or mapping[int, str]
        Mapping or sequence from replica index -> trajectory path.
    replica_frames : sequence[int] or mapping[int, int]
        Mapping or sequence from replica index -> available frame count in its
        trajectory file (0 when missing).
    default_stride : int
        Default reporter stride in MD steps per saved frame.
    replica_strides : sequence[int] or mapping[int, int], optional
        Optional per-replica stride. When provided, this maps MD steps to frame
        indices for the corresponding replica; when absent, ``default_stride``
        is used for all replicas.

    Returns
    -------
    DemuxPlan
        Validated plan containing per-segment frame slices and global summary.

    Notes
    -----
    - Never performs file I/O or trajectory loading.
    - Auto-corrects minor inconsistencies (e.g., negative/zero stride, truncated
      segments due to insufficient frames) and logs warnings. It does not raise
      for these conditions.
    - Uses a global output timeline based on planned frames per segment:
      ``expected_frames = floor((stop_md-1)/stride)+1 - floor(start_md/stride)``.
    - ``frames_per_segment`` in the returned plan is 0 when the expected frame
      count varies across segments (mixed per-replica strides).

    Examples
    --------
    Plan for a tiny two-replica, two-segment exchange history (no I/O)::

        plan = build_demux_plan(
            exchange_history=[[0, 1], [1, 0]],
            temperatures=[300.0, 310.0],
            target_temperature=300.0,
            exchange_frequency=10,
            equilibration_offset=0,
            replica_paths=["replica_00.dcd", "replica_01.dcd"],
            replica_frames=[100, 100],
            default_stride=2,
        )
    """

    if exchange_frequency <= 0:
        logger.warning("exchange_frequency <= 0; coercing to 1 for planning")
        exchange_frequency = 1
    if equilibration_offset < 0:
        logger.warning("equilibration_offset < 0; coercing to 0 for planning")
        equilibration_offset = 0
    if default_stride <= 0:
        logger.warning("default_stride <= 0; coercing to 1 for planning")
        default_stride = 1

    paths = _to_path_mapping(replica_paths)
    frames = _to_indexed_mapping(replica_frames, default=0)
    strides = (
        _to_indexed_mapping(replica_strides, default=default_stride)
        if replica_strides is not None
        else {k: int(default_stride) for k in paths.keys()}
    )

    target_idx = _closest_temperature_index(temperatures, target_temperature)
    n_segments = len(exchange_history)

    segments: List[DemuxSegmentPlan] = []
    expected_global_start = 0  # position in the output timeline (planned frames)
    common_fps: Optional[int] = None
    varied_fps: bool = False

    prev_stop_md = int(equilibration_offset)
    prev_stop_frame: int = 0

    for s, states in enumerate(exchange_history):
        # Find which replica is at the target temperature during this segment
        replica_idx = -1
        for r_idx, temp_state in enumerate(states):
            if int(temp_state) == int(target_idx):
                replica_idx = int(r_idx)
                break

        start_md = int(equilibration_offset + s * exchange_frequency)
        stop_md = int(equilibration_offset + (s + 1) * exchange_frequency)
        if start_md < prev_stop_md:
            logger.warning(
                "Non-monotonic segment times detected at segment %d: %d < %d",
                s,
                start_md,
                prev_stop_md,
            )
        prev_stop_md = stop_md

        # Determine stride for the selected replica (or default when missing)
        stride = (
            int(strides.get(replica_idx, default_stride))
            if replica_idx >= 0
            else int(default_stride)
        )
        if stride <= 0:
            logger.warning(
                "Stride <= 0 for replica %d; using default_stride=%d",
                replica_idx,
                default_stride,
            )
            stride = int(default_stride if default_stride > 0 else 1)

        # Planned frames from MD steps mapped by stride (half-open, ceil mapping)
        # Use ceil for both boundaries to avoid boundary duplication and backtracks
        start_frame = max(0, (start_md + stride - 1) // stride)
        stop_frame = max(0, (stop_md + stride - 1) // stride)
        needs_fill = False
        # Enforce monotonicity across segments even with variable per-replica stride
        if start_frame < prev_stop_frame:
            logger.warning(
                "Backward frame index at segment %d (start=%d < expected=%d); adjusting",
                s,
                start_frame,
                prev_stop_frame,
            )
            start_frame = prev_stop_frame
            if stop_frame < start_frame:
                stop_frame = start_frame
            needs_fill = True
        expected_frames = max(0, stop_frame - start_frame)

        if common_fps is None:
            common_fps = expected_frames
        elif expected_frames != common_fps:
            # Mixed per-replica stride leads to variable frames per segment
            logger.warning(
                "Variable frames per segment detected (segment %d has %d, expected %d)",
                s,
                expected_frames,
                common_fps,
            )
            varied_fps = True

        # If there is no replica at target temperature for this segment,
        # mark fill and leave indices as planned; source_path becomes empty.
        if replica_idx < 0:
            logger.warning(
                "No replica at target temperature for segment %d; will require fill",
                s,
            )
            needs_fill = True
            source_path = ""
            available = 0
        else:
            source_path = paths.get(replica_idx, "")
            available = int(frames.get(replica_idx, 0))

        # Truncate stop frame to available frames to avoid out-of-range access
        if stop_frame > available and replica_idx >= 0:
            if replica_idx >= 0:
                logger.warning(
                    "Segment %d truncated by source frames (replica=%d, have=%d, want stop=%d)",
                    s,
                    replica_idx,
                    available,
                    stop_frame,
                )
            stop_frame = max(start_frame, available)
            needs_fill = True

        segments.append(
            DemuxSegmentPlan(
                segment_index=int(s),
                replica_index=int(replica_idx),
                source_path=str(source_path),
                start_frame=int(start_frame),
                stop_frame=int(stop_frame),
                expected_frames=int(expected_frames),
                needs_fill=bool(needs_fill),
            )
        )

        # Advance the global expected output timeline by the planned frames
        expected_global_start += int(expected_frames)
        prev_stop_frame = int(stop_frame)

    total_expected = int(sum(seg.expected_frames for seg in segments))
    frames_per_segment = (
        int(common_fps) if (common_fps is not None and not varied_fps) else 0
    )

    # If frames-per-segment varied, return 0 as sentinel and log once
    if frames_per_segment == 0 and n_segments > 0:
        logger.warning(
            "frames_per_segment is variable across segments; returning 0 in plan"
        )

    return DemuxPlan(
        segments=segments,
        target_temperature=float(target_temperature),
        frames_per_segment=frames_per_segment,
        total_expected_frames=total_expected,
    )


def build_demux_frame_windows(
    *,
    total_md_steps: int,
    equilibration_steps_pre: int,
    equilibration_steps_post: int,
    stride_steps: int,
    exchange_frequency_steps: int,
    n_segments: int | None = None,
) -> list[tuple[int, int]]:
    """Plan half-open frame windows per segment using only integers.

    Parameters
    ----------
    total_md_steps : int
        Total MD steps in the run (including equilibration).
    equilibration_steps_pre : int
        MD steps for the first equilibration phase before production.
    equilibration_steps_post : int
        MD steps for the second equilibration phase before production.
    stride_steps : int
        MD steps per saved frame (reporter stride).
    exchange_frequency_steps : int
        MD steps between exchange attempts; each segment spans this length in MD steps.
    n_segments : int or None, optional
        Optional segment count override. When None, infer from totals by iterating
        segments until the start MD step reaches ``total_md_steps``.

    Returns
    -------
    list[tuple[int, int]]
        List of (start_frame, stop_frame) pairs per segment, half-open, with a
        consistent ceil mapping for both boundaries and trimmed to ``total_md_steps``.

    Notes
    -----
    - Uses a single rounding convention: ``ceil(x / stride)`` for both start and stop.
    - Enforces strict monotonicity: if a computed start would be less than the
      previous stop, it is clamped up to the previous stop.
    - Drops empty segments (where stop_frame <= start_frame).
    """
    import math

    tot = int(max(0, total_md_steps))
    eq_pre = int(max(0, equilibration_steps_pre))
    eq_post = int(max(0, equilibration_steps_post))
    stride = int(stride_steps) if int(stride_steps) > 0 else 1
    exch = int(exchange_frequency_steps) if int(exchange_frequency_steps) > 0 else 1

    eq_total = int(eq_pre + eq_post)
    windows: list[tuple[int, int]] = []

    # Determine number of segments if requested; otherwise iterate until exhausted
    if n_segments is None:
        seg_count = 0
        start_md = eq_total
        while start_md < tot:
            seg_count += 1
            start_md += exch
    else:
        seg_count = int(max(0, n_segments))

    prev_stop: int | None = None
    for s in range(seg_count):
        seg_start_md = eq_total + s * exch
        seg_stop_md = min(eq_total + (s + 1) * exch, tot)
        if seg_start_md >= seg_stop_md:
            continue
        # ceil mapping for both boundaries
        start_frame = (seg_start_md + stride - 1) // stride
        stop_frame = (seg_stop_md + stride - 1) // stride
        if prev_stop is not None and start_frame < prev_stop:
            start_frame = prev_stop
        if stop_frame <= start_frame:
            # Drop empty/degenerate segment
            continue
        windows.append((int(start_frame), int(stop_frame)))
        prev_stop = int(stop_frame)

    return windows
