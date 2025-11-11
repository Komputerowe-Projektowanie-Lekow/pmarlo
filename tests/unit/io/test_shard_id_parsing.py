"""
Tests for shard ID parsing and canonical identification.

This module tests the ShardId dataclass and parse_shard_id function
to ensure collision-free identification across multiple runs and file types.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pmarlo.io.shard_id import ShardId, parse_shard_id


class TestShardId:
    """Test ShardId dataclass functionality."""

    def test_canonical_format(self):
        """Test canonical string representation format."""
        shard_id = ShardId(
            run_id="run-20250906-170155",
            source_kind="demux",
            temperature_K=300,
            replica_index=None,
            local_index=0,
            source_path=Path("/data/run-20250906-170155/demux_T300K.dcd"),
            dataset_hash="abc123",
        )

        canonical = shard_id.canonical()
        expected = "run-20250906-170155:demux:T300:0"
        assert canonical == expected

    def test_canonical_format_replica(self):
        """Test canonical format for replica files."""
        shard_id = ShardId(
            run_id="run-20250906-170155",
            source_kind="replica",
            temperature_K=None,
            replica_index=5,
            local_index=2,
            source_path=Path("/data/run-20250906-170155/replica_05.dcd"),
            dataset_hash="def456",
        )

        canonical = shard_id.canonical()
        expected = "run-20250906-170155:replica:R5:2"
        assert canonical == expected

    def test_from_canonical_roundtrip(self):
        """Test roundtrip conversion from canonical string back to ShardId."""
        original = ShardId(
            run_id="run-20250906-170155",
            source_kind="demux",
            temperature_K=350,
            replica_index=None,
            local_index=1,
            source_path=Path("/data/run-20250906-170155/demux_T350K.dcd"),
            dataset_hash="ghi789",
        )

        canonical = original.canonical()
        reconstructed = ShardId.from_canonical(
            canonical, original.source_path, original.dataset_hash
        )

        assert reconstructed.run_id == original.run_id
        assert reconstructed.source_kind == original.source_kind
        assert reconstructed.temperature_K == original.temperature_K
        assert reconstructed.replica_index == original.replica_index
        assert reconstructed.local_index == original.local_index
        assert reconstructed.source_path == original.source_path
        assert reconstructed.dataset_hash == original.dataset_hash

    def test_invalid_canonical_format(self):
        """Test error handling for invalid canonical formats."""
        with pytest.raises(ValueError, match="Invalid canonical format"):
            ShardId.from_canonical("invalid", Path("/tmp/test.dcd"), "")

        with pytest.raises(ValueError, match="Invalid canonical format"):
            ShardId.from_canonical("too:few:parts", Path("/tmp/test.dcd"), "")

        with pytest.raises(ValueError, match="Invalid source_kind"):
            ShardId.from_canonical("run:invalid:T300:0", Path("/tmp/test.dcd"), "")

        with pytest.raises(ValueError, match="Invalid temp/replica format"):
            ShardId.from_canonical("run:demux:invalid:0", Path("/tmp/test.dcd"), "")


class TestParseShardId:
    """Test parse_shard_id function with various file patterns."""

    def test_parse_demux_file(self, tmp_path):
        """Test parsing demux_T*.dcd files."""
        # Create a mock run directory structure
        run_dir = tmp_path / "run-20250906-170155"
        run_dir.mkdir()

        # Create demux files
        demux_300k = run_dir / "demux_T300K.dcd"
        demux_300k.touch()

        demux_350k = run_dir / "demux_T350K.dcd"
        demux_350k.touch()

        # Test parsing
        shard_id = parse_shard_id(demux_300k)

        assert shard_id.run_id == "run-20250906-170155"
        assert shard_id.source_kind == "demux"
        assert shard_id.temperature_K == 300
        assert shard_id.replica_index is None
        assert shard_id.local_index == 0  # First in alphabetical order
        assert shard_id.source_path == demux_300k

        # Test second file
        shard_id_350 = parse_shard_id(demux_350k)
        assert shard_id_350.temperature_K == 350
        assert shard_id_350.local_index == 1  # Second in alphabetical order

    def test_parse_replica_file(self, tmp_path):
        """Test parsing replica_*.dcd files."""
        run_dir = tmp_path / "run-20250906-170155"
        run_dir.mkdir()

        # Create replica files
        replica_00 = run_dir / "replica_00.dcd"
        replica_00.touch()

        replica_05 = run_dir / "replica_05.dcd"
        replica_05.touch()

        # Test parsing
        shard_id = parse_shard_id(replica_00)

        assert shard_id.run_id == "run-20250906-170155"
        assert shard_id.source_kind == "replica"
        assert shard_id.temperature_K is None
        assert shard_id.replica_index == 0
        assert shard_id.local_index == 0  # replica_00 comes first alphabetically
        assert shard_id.source_path == replica_00

        # Test second file
        shard_id_05 = parse_shard_id(replica_05)
        assert shard_id_05.replica_index == 5
        assert shard_id_05.local_index == 1  # replica_05 comes second

    def test_collision_free_across_runs(self, tmp_path):
        """Test that canonical IDs differ across runs even with same shard names."""
        # Create two different run directories
        run1_dir = tmp_path / "run-20250906-170155"
        run1_dir.mkdir()

        run2_dir = tmp_path / "run-20250906-170156-seed123"
        run2_dir.mkdir()

        # Create identical shard files in both runs
        shard1 = run1_dir / "demux_T300K.dcd"
        shard1.touch()

        shard2 = run2_dir / "demux_T300K.dcd"
        shard2.touch()

        # Parse both
        id1 = parse_shard_id(shard1)
        id2 = parse_shard_id(shard2)

        # Canonical IDs should be different
        assert id1.canonical() != id2.canonical()
        assert id1.run_id != id2.run_id
        assert id1.canonical() == "run-20250906-170155:demux:T300:0"
        assert id2.canonical() == "run-20250906-170156-seed123:demux:T300:0"

    def test_mixed_file_types_same_run(self, tmp_path):
        """Test parsing mixed demux and replica files in same run."""
        run_dir = tmp_path / "run-20250906-170155"
        run_dir.mkdir()

        # Create both types of files
        demux_file = run_dir / "demux_T300K.dcd"
        demux_file.touch()

        replica_file = run_dir / "replica_00.dcd"
        replica_file.touch()

        # Parse both
        demux_id = parse_shard_id(demux_file)
        replica_id = parse_shard_id(replica_file)

        # Should have same run_id but different source_kind
        assert demux_id.run_id == replica_id.run_id
        assert demux_id.source_kind == "demux"
        assert replica_id.source_kind == "replica"

        # Canonical IDs should be different
        assert demux_id.canonical() != replica_id.canonical()

    def test_nonexistent_file(self):
        """Test error handling for nonexistent files."""
        with pytest.raises(Exception):  # PMarloError
            parse_shard_id(Path("/nonexistent/file.dcd"))

    def test_invalid_filename_pattern(self, tmp_path):
        """Test error handling for unrecognized filename patterns."""
        run_dir = tmp_path / "run-20250906-170155"
        run_dir.mkdir()

        invalid_file = run_dir / "invalid_filename.dcd"
        invalid_file.touch()

        with pytest.raises(Exception):  # PMarloError
            parse_shard_id(invalid_file)

    def test_nested_run_directory(self, tmp_path):
        """Test parsing files in nested directory structures."""
        # Create nested structure: base/experiment/run-.../
        base_dir = tmp_path / "base"
        base_dir.mkdir()

        exp_dir = base_dir / "experiment"
        exp_dir.mkdir()

        run_dir = exp_dir / "run-20250906-170155"
        run_dir.mkdir()

        shard_file = run_dir / "demux_T300K.dcd"
        shard_file.touch()

        # Should still find the run directory
        shard_id = parse_shard_id(shard_file)
        assert shard_id.run_id == "run-20250906-170155"
        assert shard_id.canonical() == "run-20250906-170155:demux:T300:0"

    def test_run_directory_not_found(self, tmp_path):
        """Test behavior when run directory cannot be found."""
        # Create file not in a run-* directory
        regular_dir = tmp_path / "regular_directory"
        regular_dir.mkdir()

        shard_file = regular_dir / "demux_T300K.dcd"
        shard_file.touch()

        with pytest.raises(Exception):  # PMarloError
            parse_shard_id(shard_file)
