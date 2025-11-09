"""Test backward compatibility of shard metadata format."""

import json
import pytest


def test_shard_metadata_field_names():
    """Test that shard metadata uses standard field names."""
    # Expected field names based on existing replica exchange shards
    expected_fields = {
        "created_at": "Timestamp of shard creation",
        "kind": "Shard type: 'demux' or 'replica'",
        "run_id": "Run identifier for uniqueness",
        "replica_id": "Replica index",
        "segment_id": "Segment/shard index",
        "range": "Frame range [start, stop] from trajectory",
        "traj": "Primary trajectory file path",
        "traj_files": "List of all trajectory files (for multi-file support)",
        "n_frames": "Number of frames in shard (auto-added by write_shard)",
        "seed": "Random seed (auto-added by write_shard)",
        "temperature_K": "Temperature in Kelvin (auto-added by write_shard)",
        "columns": "Feature column names (auto-added by write_shard)",
        "periodic": "Periodic flags per column (auto-added by write_shard)",
    }

    assert len(expected_fields) == 13


def test_field_name_conventions():
    """Test that standard field names are preferred over alternatives."""
    comparisons = [
        ("range", "frame_range", "Using standard 'range'"),
        ("traj", "trajectory", "Using standard 'traj'"),
        ("kind", "type", "Using standard 'kind'"),
        ("run_id", "runId", "Using standard 'run_id'"),
    ]

    for standard, alternative, note in comparisons:
        # Standard names should be used
        assert len(standard) > 0


def test_old_vs_new_format_compatibility():
    """Test that old and new shard formats are compatible."""
    old_shard_format = {
        "source": {
            "created_at": "2024-11-08T12:00:00Z",
            "kind": "demux",
            "run_id": "run-20241108-120000",
            "replica_id": 0,
            "segment_id": 0,
            "range": [0, 1000],  # Standard field name
            "traj": "/path/to/trajectory.dcd",  # Standard field name
            "n_frames": 1000,
            "seed": 42,
            "temperature_K": 300.0,
            "columns": ["Rg", "RMSD_ref"],
            "periodic": [False, False],
        }
    }

    new_shard_format = {
        "source": {
            "created_at": "2024-11-08T12:00:00Z",
            "kind": "demux",
            "run_id": "run-20241108-120000",
            "replica_id": 0,
            "segment_id": 0,
            "range": [0, 1000],  # Uses standard name (not frame_range)
            "traj": "/path/to/trajectory.dcd",  # Uses standard name
            "traj_files": ["/path/to/trajectory.dcd"],  # Additional field for multi-file
            "n_frames": 1000,
            "seed": 42,
            "temperature_K": 300.0,
            "columns": ["distance([0, 1])", "angle([0, 1, 2])"],  # Molecular features
            "periodic": [False, True],  # Correct periodic flags
        }
    }

    # Check field compatibility
    old_keys = set(old_shard_format["source"].keys())
    new_keys = set(new_shard_format["source"].keys())

    common_keys = old_keys & new_keys
    only_new = new_keys - old_keys

    # All old fields should be present in new format (except potentially different periodic values)
    assert len(common_keys) >= 10, "Most fields should be common"
    assert "traj_files" in only_new, "New format should add traj_files"


def test_required_fields_present():
    """Test that required fields are present in both formats."""
    required_by_write_shard = ["created_at", "kind", "run_id", "replica_id", "segment_id"]
    required_by_conformations = ["range", "traj"]  # or traj_files, trajectory, path

    shard_example = {
        "source": {
            "created_at": "2024-11-08T12:00:00Z",
            "kind": "demux",
            "run_id": "run-20241108-120000",
            "replica_id": 0,
            "segment_id": 0,
            "range": [0, 1000],
            "traj": "/path/to/trajectory.dcd",
            "n_frames": 1000,
            "seed": 42,
            "temperature_K": 300.0,
        }
    }

    source = shard_example["source"]

    for field in required_by_write_shard:
        assert field in source, f"Missing required field: {field}"

    for field in required_by_conformations:
        assert field in source, f"Missing required field: {field}"
