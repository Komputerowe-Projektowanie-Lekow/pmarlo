from __future__ import annotations

from pathlib import Path
import json

from pmarlo.replica_exchange.demux_metadata import (
    DemuxMetadata,
    serialize_metadata,
)
from pmarlo.replica_exchange.demux_plan import DemuxPlan, DemuxSegmentPlan
from pmarlo.replica_exchange.demux_engine import DemuxResult


def test_serialize_metadata_contains_required_keys(tmp_path: Path) -> None:
    plan = DemuxPlan(
        segments=[
            DemuxSegmentPlan(0, 0, str(tmp_path / "r0.dcd"), 0, 2, 2, False),
            DemuxSegmentPlan(1, 1, str(tmp_path / "r1.dcd"), 2, 4, 2, False),
        ],
        target_temperature=300.0,
        frames_per_segment=2,
        total_expected_frames=4,
    )
    result = DemuxResult(total_frames_written=4, repaired_segments=[1], skipped_segments=[], warnings=["ok"])
    runtime_info = {
        "exchange_frequency_steps": 10,
        "integration_timestep_ps": 0.002,
        "fill_policy": "repeat",
        "temperature_schedule": {"0": {"0": 300.0}},
    }

    meta_dict = serialize_metadata(result, plan, runtime_info)
    # Required v2 keys
    for key in (
        "schema_version",
        "segment_count",
        "repaired_segments",
        "skipped_segments",
        "fill_policy",
        "time_per_frame_ps",
        "source_files_checksum",
        "plan_checksum",
    ):
        assert key in meta_dict
    assert meta_dict["schema_version"] == 2
    assert meta_dict["segment_count"] == 2
    # Backward-compatible fields
    assert meta_dict["exchange_frequency_steps"] == 10
    assert meta_dict["frames_per_segment"] == 2


def test_demuxmetadata_roundtrip_and_json_schema_key(tmp_path: Path) -> None:
    meta = DemuxMetadata(
        exchange_frequency_steps=5,
        integration_timestep_ps=0.004,
        frames_per_segment=5,
        temperature_schedule={"0": {"0": 300.0}},
        segment_count=10,
        repaired_segments=[2, 5],
        skipped_segments=[7],
        fill_policy="repeat",
        time_per_frame_ps=0.004 * (5 / 5),
        source_files_checksum={"a": "deadbeef"},
        plan_checksum="cafebabe",
        schema_version=2,
    )
    path = tmp_path / "meta.json"
    meta.to_json(path)
    # File contains schema_version: 2
    as_text = path.read_text()
    data = json.loads(as_text)
    assert data.get("schema_version") == 2
    # Roundtrip via from_json
    loaded = DemuxMetadata.from_json(path)
    assert loaded.schema_version == 2
    # v1 fields still parsed
    assert loaded.exchange_frequency_steps == 5
    assert loaded.frames_per_segment == 5
