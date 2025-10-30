from __future__ import annotations

import logging

from pmarlo.demultiplexing.demux_plan import (
    DemuxPlan,
    DemuxSegmentPlan,
    build_demux_plan,
)


def test_build_demux_plan_basic_constant_stride(caplog) -> None:
    caplog.set_level(logging.WARNING)

    # Two replicas, three segments, target temperature is index 0
    exchange_history = [
        [0, 1],  # segment 0: replica 0 at target
        [1, 0],  # segment 1: replica 1 at target
        [0, 1],  # segment 2: replica 0 at target
    ]
    temperatures = [300.0, 310.0]
    target_temperature = 300.0
    exchange_frequency = 10
    equilibration_offset = 0
    default_stride = 2  # MD steps per saved frame
    replica_strides = [2, 2]
    # Sufficient frames in both files
    replica_frames = [100, 100]
    replica_paths = ["/tmp/replica_00.dcd", "/tmp/replica_01.dcd"]

    plan = build_demux_plan(
        exchange_history=exchange_history,
        temperatures=temperatures,
        target_temperature=target_temperature,
        exchange_frequency=exchange_frequency,
        equilibration_offset=equilibration_offset,
        replica_paths=replica_paths,
        replica_frames=replica_frames,
        default_stride=default_stride,
        replica_strides=replica_strides,
    )

    assert isinstance(plan, DemuxPlan)
    assert plan.target_temperature == 300.0
    # Frames per segment should be constant: (stop-1)//stride + 1 - start//stride
    # For stride=2 and exchange_frequency=10 -> 5 frames per segment
    assert plan.frames_per_segment == 5
    assert plan.total_expected_frames == 3 * 5

    # Segment 0
    s0 = plan.segments[0]
    assert isinstance(s0, DemuxSegmentPlan)
    assert s0.replica_index == 0
    assert s0.source_path.endswith("replica_00.dcd")
    assert (s0.start_frame, s0.stop_frame, s0.expected_frames) == (0, 5, 5)
    assert not s0.needs_fill

    # Segment 1
    s1 = plan.segments[1]
    assert s1.replica_index == 1
    assert s1.source_path.endswith("replica_01.dcd")
    assert (s1.start_frame, s1.stop_frame, s1.expected_frames) == (5, 10, 5)
    assert not s1.needs_fill

    # Segment 2
    s2 = plan.segments[2]
    assert s2.replica_index == 0
    assert s2.source_path.endswith("replica_00.dcd")
    assert (s2.start_frame, s2.stop_frame, s2.expected_frames) == (10, 15, 5)
    assert not s2.needs_fill
    # No warnings expected in ideal constant case
    assert not caplog.records


def test_build_demux_plan_truncation_and_fill(caplog) -> None:
    caplog.set_level(logging.WARNING)

    exchange_history = [
        [0, 1],  # segment 0 -> replica 0
        [1, 0],  # segment 1 -> replica 1
    ]
    temperatures = [300.0, 310.0]
    target_temperature = 300.0
    exchange_frequency = 10
    equilibration_offset = 0
    default_stride = 2
    replica_strides = [2, 2]
    # Replica 1 has only 7 frames, so segment 1 (frames 5..10) must truncate
    replica_frames = [100, 7]
    replica_paths = ["/tmp/replica_00.dcd", "/tmp/replica_01.dcd"]

    plan = build_demux_plan(
        exchange_history=exchange_history,
        temperatures=temperatures,
        target_temperature=target_temperature,
        exchange_frequency=exchange_frequency,
        equilibration_offset=equilibration_offset,
        replica_paths=replica_paths,
        replica_frames=replica_frames,
        default_stride=default_stride,
        replica_strides=replica_strides,
    )

    # Segment 0 as usual
    s0 = plan.segments[0]
    assert (s0.start_frame, s0.stop_frame, s0.expected_frames) == (0, 5, 5)
    assert not s0.needs_fill
    # Segment 1 truncated to available frames 7
    s1 = plan.segments[1]
    assert s1.start_frame == 5
    assert s1.stop_frame == 7
    assert s1.expected_frames == 5  # still planned 5
    assert s1.needs_fill
    # Warning should be emitted
    assert any("truncated" in r.getMessage() for r in caplog.records)


def test_build_demux_plan_missing_target_replica(caplog) -> None:
    caplog.set_level(logging.WARNING)

    # No replica at target temperature in segment 0
    exchange_history = [
        [1, 1],  # segment 0 -> no 0 present
        [0, 1],  # segment 1 -> ok
    ]
    temperatures = [300.0, 310.0]
    target_temperature = 300.0
    exchange_frequency = 8
    equilibration_offset = 0
    default_stride = 2
    replica_frames = [100, 100]
    replica_paths = ["/tmp/replica_00.dcd", "/tmp/replica_01.dcd"]

    plan = build_demux_plan(
        exchange_history=exchange_history,
        temperatures=temperatures,
        target_temperature=target_temperature,
        exchange_frequency=exchange_frequency,
        equilibration_offset=equilibration_offset,
        replica_paths=replica_paths,
        replica_frames=replica_frames,
        default_stride=default_stride,
    )

    # For stride=2 and freq=8 -> expected frames per segment = 4
    assert plan.total_expected_frames == 8 // 2 + 8 // 2 == 8
    s0 = plan.segments[0]
    assert s0.replica_index == -1  # no replica at target
    assert s0.source_path == ""
    assert s0.expected_frames == 4
    # start/stop planned from default stride
    assert (s0.start_frame, s0.stop_frame) == (0, 4)
    assert s0.needs_fill
    # Segment 1 normal
    s1 = plan.segments[1]
    assert s1.replica_index == 0
    assert s1.expected_frames == 4
    assert (s1.start_frame, s1.stop_frame) == (4, 8)
    assert not s1.needs_fill
    # Warnings logged
    assert any(
        "No replica at target temperature" in r.getMessage() for r in caplog.records
    )


def test_build_demux_plan_backward_adjustment_variable_stride(caplog) -> None:
    caplog.set_level(logging.WARNING)

    # Alternate replicas; segment 0 uses stride=2 (5 frames), segment 1 uses stride=10 (1 frame)
    exchange_history = [
        [0, 1],  # seg 0 -> replica 0
        [1, 0],  # seg 1 -> replica 1
    ]
    temperatures = [300.0, 310.0]
    target_temperature = 300.0
    exchange_frequency = 10
    equilibration_offset = 0
    default_stride = 2
    replica_strides = [2, 10]
    replica_frames = [100, 100]
    replica_paths = ["/tmp/replica_00.dcd", "/tmp/replica_01.dcd"]

    plan = build_demux_plan(
        exchange_history=exchange_history,
        temperatures=temperatures,
        target_temperature=target_temperature,
        exchange_frequency=exchange_frequency,
        equilibration_offset=equilibration_offset,
        replica_paths=replica_paths,
        replica_frames=replica_frames,
        default_stride=default_stride,
        replica_strides=replica_strides,
    )

    s0 = plan.segments[0]
    assert (s0.start_frame, s0.stop_frame, s0.expected_frames) == (0, 5, 5)
    assert not s0.needs_fill
    s1 = plan.segments[1]
    # Planned start using stride=10 is 1, but expected_global_start is 5
    # The planner adjusts start_frame to 5 and marks needs_fill
    assert s1.start_frame == 5
    # stop_frame computed with stride=10 -> (20-1)//10 + 1 = 1 + 1 = 2, but must be >= start
    assert s1.stop_frame >= s1.start_frame
    assert s1.needs_fill
    # frames_per_segment becomes 0 (variable across segments)
    assert plan.frames_per_segment == 0
    assert any("Backward frame index" in r.getMessage() for r in caplog.records)
