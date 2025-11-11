"""
Tests for workflow validation functionality.

This module tests the validation functions for build results, FES quality,
and diagnostic message formatting.
"""

from __future__ import annotations

import numpy as np
import pytest

from pmarlo.workflow.validation import (
    format_validation_report,
    validate_build_result,
    validate_fes_quality,
)


class TestValidateBuildResult:
    """Test validate_build_result function."""

    def test_valid_build_result(self, tmp_path):
        """Test validation of a valid build result."""
        # Create mock shard files
        run_dir = tmp_path / "run-20250906-170155"
        run_dir.mkdir()

        shard_files = []
        for temp in [300, 350]:
            f = run_dir / f"demux_T{temp}K.dcd"
            f.touch()
            shard_files.append(f)

        # Build catalog to get actual canonical IDs
        from pmarlo.io.catalog import build_catalog_from_paths

        catalog = build_catalog_from_paths(shard_files)
        canonical_ids = catalog.get_canonical_ids()

        # Mock build result with actual canonical IDs
        build_result = {"artifacts": {"shards_used": canonical_ids}}

        validation = validate_build_result(build_result, shard_files)

        assert validation["is_valid"] is True
        assert "Build used all 2 available shards" in validation["messages"]
        # Some warnings are expected for simple test cases (single run, narrow temp range, etc.)
        assert len(validation["errors"]) == 0

    def test_missing_shards_warning(self, tmp_path):
        """Test validation with missing shards."""
        run_dir = tmp_path / "run-20250906-170155"
        run_dir.mkdir()

        # Create 3 shard files
        shard_files = []
        for temp in [300, 350, 400]:
            f = run_dir / f"demux_T{temp}K.dcd"
            f.touch()
            shard_files.append(f)

        # Build catalog to get actual canonical IDs, but only use 2
        from pmarlo.io.catalog import build_catalog_from_paths

        catalog = build_catalog_from_paths(shard_files)
        all_ids = catalog.get_canonical_ids()

        # But only use 2 in build result
        build_result = {"artifacts": {"shards_used": all_ids[:2]}}  # Use first 2

        validation = validate_build_result(build_result, shard_files)

        assert validation["is_valid"] is True  # Not invalid, just warnings
        assert any("Missing shards in build" in msg for msg in validation["warnings"])
        # The missing shard ID will be the one not in all_ids[:2]
        missing_id = (set(all_ids) - set(all_ids[:2])).pop()
        assert missing_id in " ".join(validation["warnings"])

    def test_extra_shards_error(self, tmp_path):
        """Test validation with extra shards (used but not available)."""
        run_dir = tmp_path / "run-20250906-170155"
        run_dir.mkdir()

        # Create only 1 shard file
        f = run_dir / "demux_T300K.dcd"
        f.touch()

        # But use 2 in build result
        build_result = {
            "artifacts": {
                "shards_used": [
                    "run-20250906-170155:demux:T300:0",
                    "run-20250906-170155:demux:T350:0",  # This doesn't exist
                ]
            }
        }

        validation = validate_build_result(build_result, [f])

        assert validation["is_valid"] is False
        assert any(
            "references shards not present" in msg for msg in validation["errors"]
        )

    def test_non_canonical_shard_ids_raise(self, tmp_path):
        """Non-canonical shard identifiers should raise immediately."""
        run_dir = tmp_path / "run-20250906-170155"
        run_dir.mkdir()

        f = run_dir / "demux_T300K.dcd"
        f.touch()

        build_result = {"artifacts": {"shards_used": ["demux_T300K"]}}

        with pytest.raises(ValueError):
            validate_build_result(build_result, [f])

    def test_mixed_source_kinds(self, tmp_path):
        """Test validation with mixed demux and replica files."""
        run_dir = tmp_path / "run-20250906-170155"
        run_dir.mkdir()

        # Create both demux and replica files
        demux_file = run_dir / "demux_T300K.dcd"
        demux_file.touch()

        replica_file = run_dir / "replica_00.dcd"
        replica_file.touch()

        build_result = {
            "artifacts": {
                "shards_used": [
                    "run-20250906-170155:demux:T300:0",
                    "run-20250906-170155:replica:R0:0",
                ]
            }
        }

        validation = validate_build_result(build_result, [demux_file, replica_file])

        assert validation["is_valid"] is True
        assert any(
            "Mixed source kinds detected" in msg for msg in validation["warnings"]
        )

    def test_multiple_runs(self, tmp_path):
        """Test validation with multiple runs."""
        # Create files in two runs
        run1_dir = tmp_path / "run-20250906-170155"
        run1_dir.mkdir()

        run2_dir = tmp_path / "run-20250906-170156"
        run2_dir.mkdir()

        f1 = run1_dir / "demux_T300K.dcd"
        f1.touch()

        f2 = run2_dir / "demux_T300K.dcd"
        f2.touch()

        build_result = {
            "artifacts": {
                "shards_used": [
                    "run-20250906-170155:demux:T300:0",
                    "run-20250906-170156:demux:T300:0",
                ]
            }
        }

        validation = validate_build_result(build_result, [f1, f2])

        assert validation["is_valid"] is True
        assert any("Multiple runs detected" in msg for msg in validation["warnings"])

    def test_shard_table_generation(self, tmp_path):
        """Test that shard table is generated correctly."""
        run_dir = tmp_path / "run-20250906-170155"
        run_dir.mkdir()

        f = run_dir / "demux_T300K.dcd"
        f.touch()

        build_result = {
            "artifacts": {"shards_used": ["run-20250906-170155:demux:T300:0"]}
        }

        validation = validate_build_result(build_result, [f])

        assert "shard_table" in validation
        assert len(validation["shard_table"]) > 0

        table_entry = validation["shard_table"][0]
        assert "canonical_id" in table_entry
        assert "run_id" in table_entry
        assert "source_kind" in table_entry
        assert table_entry["canonical_id"] == "run-20250906-170155:demux:T300:0"


class TestValidateFesQuality:
    """Test validate_fes_quality function."""

    def test_valid_fes_data(self):
        """Test validation of valid FES data."""
        fes_data = {
            "fes": np.random.rand(10, 10),
            "values": np.random.rand(10, 10) * 10,
        }

        validation = validate_fes_quality(fes_data)

        assert validation["is_valid"] is True
        assert len(validation["errors"]) == 0

    def test_missing_fes_data(self):
        """Test validation with missing FES data."""
        fes_data = {}

        validation = validate_fes_quality(fes_data)

        assert validation["is_valid"] is False
        assert any("No FES values found" in msg for msg in validation["errors"])

    def test_fes_with_nans(self):
        """Test validation of FES data containing NaN values."""
        fes_values = np.random.rand(10, 10)
        fes_values[5, 5] = np.nan
        fes_values[2, 3] = np.nan

        fes_data = {"fes": fes_values}

        validation = validate_fes_quality(fes_data)

        assert validation["is_valid"] is True  # NaNs are warnings, not errors
        assert any("NaN values" in msg for msg in validation["warnings"])
        assert "2" in " ".join(validation["warnings"])  # Should mention count

    def test_empty_bins_detection(self):
        """Test detection of empty FES bins."""
        # Create FES with many very high values (indicating empty bins)
        fes_values = np.random.rand(20, 20) * 5  # Normal values
        fes_values[5:15, 5:15] = 200  # Very high values = empty bins (50% of bins)

        fes_data = {"fes": fes_values}

        validation = validate_fes_quality(fes_data)

        assert validation["is_valid"] is True
        assert any("empty FES bins" in msg for msg in validation["warnings"])
        assert "empty_bins_ratio" in validation["metrics"]

    def test_narrow_fes_range(self):
        """Test detection of narrow FES energy range."""
        # Create FES with very small range
        fes_values = np.ones((10, 10)) * 2.5  # Almost constant values
        fes_values += np.random.rand(10, 10) * 0.01  # Tiny variations

        fes_data = {"fes": fes_values}

        validation = validate_fes_quality(fes_data)

        assert validation["is_valid"] is True
        assert any("Narrow FES range" in msg for msg in validation["warnings"])

    def test_accepts_msm_artifact_key(self):
        """FES validation should handle MSM artifact dictionaries."""

        fes_values = np.arange(100, dtype=float).reshape(10, 10)
        fes_data = {"F": fes_values}

        validation = validate_fes_quality(fes_data)

        assert validation["is_valid"] is True
        assert validation["errors"] == []
        assert "fes_range" in validation["metrics"]


class TestFormatValidationReport:
    """Test format_validation_report function."""

    def test_format_valid_report(self):
        """Test formatting of valid validation report."""
        validation_results = {
            "is_valid": True,
            "messages": ["Build completed successfully", "All shards used"],
            "warnings": [],
            "errors": [],
            "summary": {"available_shard_count": 5, "used_shard_count": 5},
            "shard_table": [
                {
                    "canonical_id": "run1:demux:T300:0",
                    "run_id": "run1",
                    "source_kind": "demux",
                    "temp_or_replica": "T300K",
                    "local_index": "0",
                    "source_path": "/path/to/file.dcd",
                }
            ],
        }

        report = format_validation_report(validation_results)

        assert "VALID" in report
        assert "Build completed successfully" in report
        assert "All shards used" in report
        assert "Available shards: 5" in report
        assert "Used shards: 5" in report

    def test_format_invalid_report(self):
        """Test formatting of invalid validation report."""
        validation_results = {
            "is_valid": False,
            "messages": [],
            "warnings": ["Missing some data"],
            "errors": ["Critical error occurred"],
            "summary": {"available_shard_count": 3, "used_shard_count": 5},
            "shard_table": [],
        }

        report = format_validation_report(validation_results)

        assert "INVALID" in report
        assert "Critical error occurred" in report
        assert "Missing some data" in report

    def test_format_with_warnings(self):
        """Test formatting report with warnings."""
        validation_results = {
            "is_valid": True,
            "messages": ["Basic validation passed"],
            "warnings": ["Mixed source kinds detected", "Narrow temperature range"],
            "errors": [],
            "summary": {},
            "shard_table": [],
        }

        report = format_validation_report(validation_results)

        assert "VALID" in report
        assert "Mixed source kinds detected" in report
        assert "Narrow temperature range" in report

    def test_format_shard_table(self):
        """Test formatting of shard table in report."""
        validation_results = {
            "is_valid": True,
            "messages": [],
            "warnings": [],
            "errors": [],
            "summary": {},
            "shard_table": [
                {
                    "canonical_id": "run1:demux:T300:0",
                    "run_id": "run1",
                    "source_kind": "demux",
                    "temp_or_replica": "T300K",
                    "local_index": "0",
                    "source_path": "/very/long/path/to/demux_T300K.dcd",
                },
                {
                    "canonical_id": "run1:replica:R0:1",
                    "run_id": "run1",
                    "source_kind": "replica",
                    "temp_or_replica": "R0",
                    "local_index": "1",
                    "source_path": "/very/long/path/to/replica_00.dcd",
                },
            ],
        }

        report = format_validation_report(validation_results)

        assert "Shard Information" in report
        assert "Canonical ID" in report
        assert "run1:demux:T300:0" in report
        assert "run1:replica:R0:1" in report
        assert "demux" in report
        assert "replica" in report
