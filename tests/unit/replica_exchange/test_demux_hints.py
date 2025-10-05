from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("sklearn")

from pmarlo.demultiplexing.demux_engine import DemuxResult
from pmarlo.demultiplexing.demux_hints import load_demux_hints
from pmarlo.demultiplexing.demux_metadata import serialize_metadata
from pmarlo.demultiplexing.demux_plan import DemuxPlan, DemuxSegmentPlan


def test_contiguous_blocks_with_repairs(tmp_path: Path) -> None:
    # Plan: three segments of 2 frames each
    plan = DemuxPlan(
        segments=[
            DemuxSegmentPlan(0, 0, "r0.dcd", 0, 2, 2, False),
            DemuxSegmentPlan(1, 1, "r1.dcd", 0, 2, 2, True),  # repaired partially
            DemuxSegmentPlan(2, 0, "r0.dcd", 2, 4, 2, False),
        ],
        target_temperature=300.0,
        frames_per_segment=2,
        total_expected_frames=6,
    )
    # Result: first seg 2 real, second seg 1 real (1 filled), third seg 2 real
    result = DemuxResult(
        total_frames_written=6,
        repaired_segments=[1],
        skipped_segments=[],
        warnings=[],
        segment_real_frames=[2, 1, 2],
    )
    runtime_info = {
        "exchange_frequency_steps": 1,
        "integration_timestep_ps": 0.002,
        "fill_policy": "repeat",
        "temperature_schedule": {},
        "frames_per_segment": 2,
    }
    meta = serialize_metadata(result, plan, runtime_info)
    # Write/read round-trip
    meta_path = tmp_path / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    hints = load_demux_hints(meta_path)

    # Blocks of real frames: [0, 2) then [4, 6)
    assert hints.contiguous_blocks == [(0, 2), (4, 6)]
    assert hints.fill_policy == "repeat"
    assert hints.repaired_segments == [1]
    assert hints.total_expected_frames == 6
