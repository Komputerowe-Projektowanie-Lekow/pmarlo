"""
Tests for catalog deduplication and validation.

This module tests the ShardCatalog class and validation functions
to ensure proper handling of mixed file types and collision-free identification.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pmarlo.io.catalog import (
    ShardCatalog,
    build_catalog_from_paths,
    validate_shard_usage,
)
from pmarlo.io.shard_id import ShardId


class TestShardCatalog:
    """Test ShardCatalog functionality."""

    def test_empty_catalog(self):
        """Test empty catalog initialization."""
        catalog = ShardCatalog()
        assert len(catalog.shards) == 0
        assert len(catalog.source_kinds) == 0
        assert len(catalog.run_ids) == 0

    def test_add_shard(self, tmp_path):
        """Test adding shards to catalog."""
        catalog = ShardCatalog()

        # Create a mock shard
        shard_path = tmp_path / "run-20250906-170155" / "demux_T300K.dcd"
        shard_path.parent.mkdir(parents=True)
        shard_path.touch()

        shard_id = ShardId(
            run_id="run-20250906-170155",
            source_kind="demux",
            temperature_K=300,
            replica_index=None,
            local_index=0,
            source_path=shard_path,
            dataset_hash="test123",
        )

        catalog.add_shard(shard_id)

        assert len(catalog.shards) == 1
        assert "run-20250906-170155:demux:T300:0" in catalog.shards
        assert "demux" in catalog.source_kinds
        assert "run-20250906-170155" in catalog.run_ids

    def test_add_duplicate_shard_same_path(self, tmp_path):
        """Test adding duplicate shard with same path (should not error)."""
        catalog = ShardCatalog()

        shard_path = tmp_path / "run-20250906-170155" / "demux_T300K.dcd"
        shard_path.parent.mkdir(parents=True)
        shard_path.touch()

        shard_id1 = ShardId(
            run_id="run-20250906-170155",
            source_kind="demux",
            temperature_K=300,
            replica_index=None,
            local_index=0,
            source_path=shard_path,
            dataset_hash="test123",
        )

        shard_id2 = ShardId(
            run_id="run-20250906-170155",
            source_kind="demux",
            temperature_K=300,
            replica_index=None,
            local_index=0,
            source_path=shard_path,
            dataset_hash="different_hash",
        )

        catalog.add_shard(shard_id1)
        catalog.add_shard(shard_id2)  # Should not raise error

        assert len(catalog.shards) == 1  # Still only one entry

    def test_add_duplicate_shard_different_path(self, tmp_path):
        """Test adding duplicate shard with different path (should error)."""
        catalog = ShardCatalog()

        path1 = tmp_path / "run1" / "demux_T300K.dcd"
        path1.parent.mkdir(parents=True)
        path1.touch()

        path2 = tmp_path / "run2" / "demux_T300K.dcd"
        path2.parent.mkdir(parents=True)
        path2.touch()

        shard_id1 = ShardId(
            run_id="run-20250906-170155",
            source_kind="demux",
            temperature_K=300,
            replica_index=None,
            local_index=0,
            source_path=path1,
            dataset_hash="test123",
        )

        shard_id2 = ShardId(
            run_id="run-20250906-170155",  # Same canonical ID
            source_kind="demux",
            temperature_K=300,
            replica_index=None,
            local_index=0,
            source_path=path2,  # Different path
            dataset_hash="test123",
        )

        catalog.add_shard(shard_id1)

        with pytest.raises(Exception):  # PMarloError
            catalog.add_shard(shard_id2)

    def test_add_from_paths(self, tmp_path):
        """Test adding shards from file paths."""
        # Create mock directory structure
        run_dir = tmp_path / "run-20250906-170155"
        run_dir.mkdir()

        demux_file = run_dir / "demux_T300K.dcd"
        demux_file.touch()

        replica_file = run_dir / "replica_00.dcd"
        replica_file.touch()

        catalog = ShardCatalog()
        catalog.add_from_paths([demux_file, replica_file])

        assert len(catalog.shards) == 2
        assert len(catalog.source_kinds) == 2  # demux and replica
        assert "demux" in catalog.source_kinds
        assert "replica" in catalog.source_kinds

    def test_mixed_source_kinds_warning(self, tmp_path):
        """Test warning generation for mixed source kinds."""
        # Create mock files
        run_dir = tmp_path / "run-20250906-170155"
        run_dir.mkdir()

        demux_file = run_dir / "demux_T300K.dcd"
        demux_file.touch()

        replica_file = run_dir / "replica_00.dcd"
        replica_file.touch()

        catalog = ShardCatalog()
        catalog.add_from_paths([demux_file, replica_file])

        used_ids = {
            "run-20250906-170155:demux:T300:0",
            "run-20250906-170155:replica:R0:0",
        }
        validation = catalog.validate_against_used(used_ids)

        assert "Mixed source kinds detected" in " ".join(validation["warnings"])

    def test_multiple_runs_warning(self, tmp_path):
        """Test warning generation for multiple runs."""
        # Create files in different runs
        run1_dir = tmp_path / "run-20250906-170155"
        run1_dir.mkdir()

        run2_dir = tmp_path / "run-20250906-170156"
        run2_dir.mkdir()

        file1 = run1_dir / "demux_T300K.dcd"
        file1.touch()

        file2 = run2_dir / "demux_T300K.dcd"
        file2.touch()

        catalog = ShardCatalog()
        catalog.add_from_paths([file1, file2])

        used_ids = set(catalog.get_canonical_ids())
        validation = catalog.validate_against_used(used_ids)

        assert "Multiple runs detected" in " ".join(validation["warnings"])

    def test_temperature_gap_detection(self, tmp_path):
        """Test detection of missing temperatures in demux data."""
        run_dir = tmp_path / "run-20250906-170155"
        run_dir.mkdir()

        # Create files with gap (300K and 400K, missing 350K)
        demux_300 = run_dir / "demux_T300K.dcd"
        demux_300.touch()

        demux_400 = run_dir / "demux_T400K.dcd"
        demux_400.touch()

        catalog = ShardCatalog()
        catalog.add_from_paths([demux_300, demux_400])

        used_ids = set(catalog.get_canonical_ids())
        validation = catalog.validate_against_used(used_ids)

        assert any(
            "Missing temperatures" in warning for warning in validation["warnings"]
        )

    def test_replica_continuity_check(self, tmp_path):
        """Test detection of non-contiguous replica indices."""
        run_dir = tmp_path / "run-20250906-170155"
        run_dir.mkdir()

        # Create replica files with gap (00 and 02, missing 01)
        replica_00 = run_dir / "replica_00.dcd"
        replica_00.touch()

        replica_02 = run_dir / "replica_02.dcd"
        replica_02.touch()

        catalog = ShardCatalog()
        catalog.add_from_paths([replica_00, replica_02])

        used_ids = set(catalog.get_canonical_ids())
        validation = catalog.validate_against_used(used_ids)

        assert any("not contiguous" in warning for warning in validation["warnings"])


class TestBuildCatalogFromPaths:
    """Test build_catalog_from_paths function."""

    def test_build_catalog_single_run(self, tmp_path):
        """Test building catalog from paths in single run."""
        run_dir = tmp_path / "run-20250906-170155"
        run_dir.mkdir()

        files = []
        for temp in [300, 350, 400]:
            f = run_dir / f"demux_T{temp}K.dcd"
            f.touch()
            files.append(f)

        catalog = build_catalog_from_paths(files)

        assert len(catalog.shards) == 3
        assert len(catalog.run_ids) == 1
        assert len(catalog.source_kinds) == 1

        canonical_ids = catalog.get_canonical_ids()
        assert len(canonical_ids) == 3
        assert all("run-20250906-170155:demux:" in cid for cid in canonical_ids)

    def test_build_catalog_multiple_runs(self, tmp_path):
        """Test building catalog from paths across multiple runs."""
        files = []

        # Run 1
        run1_dir = tmp_path / "run-20250906-170155"
        run1_dir.mkdir()
        f1 = run1_dir / "demux_T300K.dcd"
        f1.touch()
        files.append(f1)

        # Run 2
        run2_dir = tmp_path / "run-20250906-170156"
        run2_dir.mkdir()
        f2 = run2_dir / "demux_T300K.dcd"
        f2.touch()
        files.append(f2)

        catalog = build_catalog_from_paths(files)

        assert len(catalog.shards) == 2
        assert len(catalog.run_ids) == 2

        canonical_ids = catalog.get_canonical_ids()
        assert len(set(canonical_ids)) == 2  # All unique


class TestValidateShardUsage:
    """Test validate_shard_usage function."""

    def test_perfect_match(self, tmp_path):
        """Test validation with perfect match between available and used."""
        run_dir = tmp_path / "run-20250906-170155"
        run_dir.mkdir()

        files = []

        for temp in [300, 350]:
            f = run_dir / f"demux_T{temp}K.dcd"
            f.touch()
            files.append(f)

        # Build catalog to get actual canonical IDs
        from pmarlo.io.catalog import build_catalog_from_paths

        catalog = build_catalog_from_paths(files)
        used_ids = set(catalog.get_canonical_ids())
        validation = validate_shard_usage(files, used_ids)

        assert len(validation["missing"]) == 0
        assert len(validation["extra"]) == 0
        assert len(validation["warnings"]) == 0

    def test_missing_shards(self, tmp_path):
        """Test validation with missing shards."""
        run_dir = tmp_path / "run-20250906-170155"
        run_dir.mkdir()

        files = []
        for temp in [300, 350]:
            f = run_dir / f"demux_T{temp}K.dcd"
            f.touch()
            files.append(f)

        # Build catalog to get actual canonical IDs, but only use one
        from pmarlo.io.catalog import build_catalog_from_paths

        catalog = build_catalog_from_paths(files)
        all_ids = catalog.get_canonical_ids()
        used_ids = {all_ids[0]}  # Only use the first one
        validation = validate_shard_usage(files, used_ids)

        assert len(validation["missing"]) == 1
        # The missing one should be the other ID in the catalog
        missing_id = (set(all_ids) - used_ids).pop()
        assert missing_id in validation["missing"]
        assert len(validation["extra"]) == 0

    def test_extra_shards(self, tmp_path):
        """Test validation with extra shards (used but not available)."""
        run_dir = tmp_path / "run-20250906-170155"
        run_dir.mkdir()

        f = run_dir / "demux_T300K.dcd"
        f.touch()

        # Use both available and non-existent shard
        used_ids = {
            "run-20250906-170155:demux:T300:0",  # Available
            "run-20250906-170155:demux:T350:0",  # Not available
        }
        validation = validate_shard_usage([f], used_ids)

        assert len(validation["missing"]) == 0
        assert len(validation["extra"]) == 1
        assert "run-20250906-170155:demux:T350:0" in validation["extra"]
