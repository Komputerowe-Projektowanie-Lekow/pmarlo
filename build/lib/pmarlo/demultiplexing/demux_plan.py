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
from typing import Dict, List, Mapping, Optional, Sequence, Union

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


@dataclass
class _PlanState:
    prev_stop_md: int
    prev_stop_frame: int
    common_fps: Optional[int] = None
    varied_fps: bool = False


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
    """Build a validated demultiplexing plan without touching trajectory data."""

    exchange_frequency, equilibration_offset, default_stride = _normalize_plan_inputs(
        exchange_frequency,
        equilibration_offset,
        default_stride,
    )
    paths, frames, strides = _prepare_replica_metadata(
        replica_paths,
        replica_frames,
        replica_strides,
        default_stride,
    )

    target_idx = _closest_temperature_index(temperatures, target_temperature)
    state = _PlanState(prev_stop_md=equilibration_offset, prev_stop_frame=0)
    segments: List[DemuxSegmentPlan] = []

    for segment_index, states in enumerate(exchange_history):
        replica_idx = _replica_at_target(states, target_idx)
        start_md, stop_md = _segment_md_window(
            segment_index,
            exchange_frequency,
            equilibration_offset,
        )
        stride = _resolve_stride(replica_idx, strides, default_stride)
        start_frame, stop_frame, needs_fill, state.prev_stop_md = _compute_frame_window(
            segment_index,
            start_md,
            stop_md,
            stride,
            state.prev_stop_md,
            state.prev_stop_frame,
        )

        expected_frames = max(0, stop_frame - start_frame)
        state = _update_common_fps(state, expected_frames, segment_index)

        source_path, available, needs_fill = _resolve_source_metadata(
            replica_idx,
            paths,
            frames,
            needs_fill,
            segment_index,
        )
        start_frame, stop_frame, needs_fill = _truncate_to_available(
            start_frame,
            stop_frame,
            available,
            needs_fill,
            segment_index,
            replica_idx,
        )

        segments.append(
            DemuxSegmentPlan(
                segment_index=int(segment_index),
                replica_index=int(replica_idx),
                source_path=source_path,
                start_frame=int(start_frame),
                stop_frame=int(stop_frame),
                expected_frames=int(expected_frames),
                needs_fill=bool(needs_fill),
            )
        )

        state.prev_stop_frame = int(stop_frame)

    total_expected = int(sum(seg.expected_frames for seg in segments))
    frames_per_segment = (
        int(state.common_fps)
        if (state.common_fps is not None and not state.varied_fps)
        else 0
    )
    if frames_per_segment == 0 and segments:
        logger.warning(
            "frames_per_segment is variable across segments; returning 0 in plan"
        )

    return DemuxPlan(
        segments=segments,
        target_temperature=float(target_temperature),
        frames_per_segment=frames_per_segment,
        total_expected_frames=total_expected,
    )


def _normalize_plan_inputs(
    exchange_frequency: int,
    equilibration_offset: int,
    default_stride: int,
) -> tuple[int, int, int]:
    if exchange_frequency <= 0:
        logger.warning("exchange_frequency <= 0; coercing to 1 for planning")
        exchange_frequency = 1
    if equilibration_offset < 0:
        logger.warning("equilibration_offset < 0; coercing to 0 for planning")
        equilibration_offset = 0
    if default_stride <= 0:
        logger.warning("default_stride <= 0; coercing to 1 for planning")
        default_stride = 1
    return int(exchange_frequency), int(equilibration_offset), int(default_stride)


def _prepare_replica_metadata(
    replica_paths: Union[Sequence[str], Mapping[int, str]],
    replica_frames: Union[Sequence[int], Mapping[int, int]],
    replica_strides: Optional[Union[Sequence[int], Mapping[int, int]]],
    default_stride: int,
) -> tuple[Dict[int, str], Dict[int, int], Dict[int, int]]:
    paths = _to_path_mapping(replica_paths)
    frames = _to_indexed_mapping(replica_frames, default=0)
    if replica_strides is not None:
        strides = _to_indexed_mapping(replica_strides, default=default_stride)
    else:
        strides = {idx: int(default_stride) for idx in paths.keys()}
    return paths, frames, strides


def _replica_at_target(states: Sequence[int], target_idx: int) -> int:
    for replica_idx, temp_state in enumerate(states):
        if int(temp_state) == int(target_idx):
            return int(replica_idx)
    return -1


def _segment_md_window(
    segment_index: int,
    exchange_frequency: int,
    equilibration_offset: int,
) -> tuple[int, int]:
    start_md = int(equilibration_offset + segment_index * exchange_frequency)
    stop_md = int(equilibration_offset + (segment_index + 1) * exchange_frequency)
    return start_md, stop_md


def _resolve_stride(
    replica_idx: int,
    strides: Dict[int, int],
    default_stride: int,
) -> int:
    stride = int(strides.get(replica_idx, default_stride))
    if stride <= 0:
        logger.warning(
            "Stride <= 0 for replica %d; using default_stride=%d",
            replica_idx,
            default_stride,
        )
        stride = int(default_stride if default_stride > 0 else 1)
    return stride


def _compute_frame_window(
    segment_index: int,
    start_md: int,
    stop_md: int,
    stride: int,
    prev_stop_md: int,
    prev_stop_frame: int,
) -> tuple[int, int, bool, int]:
    if start_md < prev_stop_md:
        logger.warning(
            "Non-monotonic segment times detected at segment %d: %d < %d",
            segment_index,
            start_md,
            prev_stop_md,
        )
    start_frame = max(0, (start_md + stride - 1) // stride)
    stop_frame = max(0, (stop_md + stride - 1) // stride)
    needs_fill = False
    if start_frame < prev_stop_frame:
        logger.warning(
            "Backward frame index at segment %d (start=%d < expected=%d); adjusting",
            segment_index,
            start_frame,
            prev_stop_frame,
        )
        start_frame = prev_stop_frame
        if stop_frame < start_frame:
            stop_frame = start_frame
        needs_fill = True
    return start_frame, stop_frame, needs_fill, stop_md


def _update_common_fps(
    state: _PlanState,
    expected_frames: int,
    segment_index: int,
) -> _PlanState:
    if state.common_fps is None:
        state.common_fps = expected_frames
    elif expected_frames != state.common_fps:
        logger.warning(
            "Variable frames per segment detected (segment %d has %d, expected %d)",
            segment_index,
            expected_frames,
            state.common_fps,
        )
        state.varied_fps = True
    return state


def _resolve_source_metadata(
    replica_idx: int,
    paths: Dict[int, str],
    frames: Dict[int, int],
    needs_fill: bool,
    segment_index: int,
) -> tuple[str, int, bool]:
    if replica_idx < 0:
        logger.warning(
            "No replica at target temperature for segment %d; will require fill",
            segment_index,
        )
        return "", 0, True
    source_path = paths.get(replica_idx, "")
    available = int(frames.get(replica_idx, 0))
    return source_path, available, needs_fill


def _truncate_to_available(
    start_frame: int,
    stop_frame: int,
    available: int,
    needs_fill: bool,
    segment_index: int,
    replica_idx: int,
) -> tuple[int, int, bool]:
    if replica_idx >= 0 and stop_frame > available:
        logger.warning(
            "Segment %d truncated by source frames (replica=%d, have=%d, want stop=%d)",
            segment_index,
            replica_idx,
            available,
            stop_frame,
        )
        stop_frame = max(start_frame, available)
        needs_fill = True
    return start_frame, stop_frame, needs_fill


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
