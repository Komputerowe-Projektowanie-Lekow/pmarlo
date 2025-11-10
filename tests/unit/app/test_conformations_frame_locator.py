from pathlib import Path

import pytest

from pmarlo_webapp.app.backend.conformations import (
    _derive_frame_range_from_metadata,
)


def _make_shard_meta(start: int, stop: int) -> dict:
    return {
        "start": start,
        "stop": stop,
        "frames_loaded": stop - start,
    }


def _make_source_meta(
    run_id: str, replica_id: int, segment_id: int, traj_name: str
) -> dict:
    return {
        "run_id": run_id,
        "replica_id": replica_id,
        "segment_id": segment_id,
        "traj_files": [f"/runs/{run_id}/{traj_name}"],
    }


def test_derive_frame_range_tracks_segments_per_run():
    tracker = {}
    shard_path = Path("T300K_run-seg0000.json")

    shard_meta = _make_shard_meta(0, 100)
    source_meta = _make_source_meta("run-A", 0, 0, "demux_A.dcd")
    start, stop = _derive_frame_range_from_metadata(
        shard_path, shard_meta, source_meta, tracker, stride=1
    )
    assert (start, stop) == (0, 100)
    assert source_meta["range"] == [0, 100]

    shard_meta_b = _make_shard_meta(100, 200)
    source_meta_b = _make_source_meta("run-A", 0, 1, "demux_A.dcd")
    start_b, stop_b = _derive_frame_range_from_metadata(
        shard_path, shard_meta_b, source_meta_b, tracker, stride=1
    )
    assert (start_b, stop_b) == (100, 200)
    assert source_meta_b["range"] == [100, 200]


def test_derive_frame_range_resets_for_new_run_and_replica():
    tracker = {}
    shard_path = Path("T300K_other.json")

    shard_meta = _make_shard_meta(250, 350)
    source_meta = _make_source_meta("run-B", 1, 7, "demux_B_rep1.dcd")
    start, stop = _derive_frame_range_from_metadata(
        shard_path, shard_meta, source_meta, tracker, stride=1
    )
    assert (start, stop) == (250, 350)

    shard_meta_new = _make_shard_meta(25, 125)
    source_meta_new = _make_source_meta("run-C", 0, 0, "demux_C_rep0.dcd")
    start_new, stop_new = _derive_frame_range_from_metadata(
        shard_path, shard_meta_new, source_meta_new, tracker, stride=1
    )
    assert (start_new, stop_new) == (25, 125)


def test_derive_frame_range_errors_when_no_frames():
    tracker = {}
    shard_path = Path("invalid.json")
    shard_meta = {"start": 10, "stop": 10, "frames_loaded": 0}
    source_meta = _make_source_meta("run-Z", 0, 0, "traj.dcd")

    with pytest.raises(ValueError):
        _derive_frame_range_from_metadata(
            shard_path, shard_meta, source_meta, tracker, stride=1
        )


def test_derive_frame_range_applies_stride_multiplier():
    tracker = {}
    shard_path = Path("T300K_stride.json")
    shard_meta = _make_shard_meta(0, 10)
    source_meta = _make_source_meta("run-D", 0, 0, "demux_D.dcd")

    start, stop = _derive_frame_range_from_metadata(
        shard_path, shard_meta, source_meta, tracker, stride=5
    )
    assert (start, stop) == (0, 50)

    shard_meta_next = _make_shard_meta(10, 20)
    source_meta_next = _make_source_meta("run-D", 0, 1, "demux_D.dcd")
    start_next, stop_next = _derive_frame_range_from_metadata(
        shard_path, shard_meta_next, source_meta_next, tracker, stride=5
    )
    assert (start_next, stop_next) == (50, 100)
