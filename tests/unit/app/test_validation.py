"""Tests for the simulation run validation and discovery system."""

import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest

from pmarlo_webapp.app.backend import Backend, WorkspaceLayout
from pmarlo_webapp.app.backend.validation import RunStatus


@pytest.fixture
def temp_workspace():
    """Create a temporary workspace for testing."""
    temp_dir = Path(tempfile.mkdtemp())
    try:
        workspace = temp_dir / "app_output"
        workspace.mkdir()

        layout = WorkspaceLayout(
            app_root=temp_dir,
            inputs_dir=temp_dir / "app_input",
            workspace_dir=workspace,
            sims_dir=workspace / "sims",
            shards_dir=workspace / "shards",
            models_dir=workspace / "models",
            bundles_dir=workspace / "bundles",
            logs_dir=workspace / "logs",
            state_path=workspace / "state.json",
        )
        layout.ensure()

        yield layout
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def backend(temp_workspace):
    """Create a backend instance with temp workspace."""
    return Backend(temp_workspace)


def test_discover_empty_sims_directory(backend):
    """Test discovering runs in an empty sims directory."""
    validations = backend.discover_all_runs()
    assert len(validations) == 0


def test_discover_empty_run_directory(backend, temp_workspace):
    """Test discovering a run with empty replica_exchange directory."""
    run_dir = temp_workspace.sims_dir / "run-20251107-000000"
    run_dir.mkdir()
    (run_dir / "replica_exchange").mkdir()

    validations = backend.discover_all_runs()
    assert len(validations) == 1
    assert validations[0].run_id == "run-20251107-000000"
    assert validations[0].status == RunStatus.EMPTY
    assert not validations[0].is_valid


def test_discover_run_with_trajectories(backend, temp_workspace):
    """Test discovering a run with trajectory files."""
    run_dir = temp_workspace.sims_dir / "run-20251107-000001"
    remd_dir = run_dir / "replica_exchange"
    remd_dir.mkdir(parents=True)

    (remd_dir / "replica_00.dcd").touch()
    (remd_dir / "replica_01.dcd").touch()
    (remd_dir / "replica_02.dcd").touch()

    validations = backend.discover_all_runs()
    assert len(validations) == 1
    v = validations[0]
    assert v.run_id == "run-20251107-000001"
    assert v.metadata["trajectory_count"] == 3
    assert v.status in (RunStatus.MISSING_DEMUX, RunStatus.INCOMPLETE)


def test_discover_complete_run(backend, temp_workspace):
    """Test discovering a complete run with all required files."""
    run_dir = temp_workspace.sims_dir / "run-20251107-000002"
    remd_dir = run_dir / "replica_exchange"
    remd_dir.mkdir(parents=True)

    (remd_dir / "replica_00.dcd").touch()
    (remd_dir / "replica_01.dcd").touch()
    (remd_dir / "demux_T300K.dcd").touch()
    (remd_dir / "analysis_results.json").write_text("{}")

    validations = backend.discover_all_runs()
    assert len(validations) == 1
    v = validations[0]
    assert v.run_id == "run-20251107-000002"
    assert v.status == RunStatus.COMPLETE
    assert v.is_valid


def test_validation_summary(backend, temp_workspace):
    """Test validation summary statistics."""
    run1 = temp_workspace.sims_dir / "run-20251107-000003"
    remd1 = run1 / "replica_exchange"
    remd1.mkdir(parents=True)
    (remd1 / "replica_00.dcd").touch()
    (remd1 / "demux_T300K.dcd").touch()
    (remd1 / "analysis_results.json").write_text("{}")

    run2 = temp_workspace.sims_dir / "run-20251107-000004"
    remd2 = run2 / "replica_exchange"
    remd2.mkdir(parents=True)

    summary = backend.get_validation_summary()
    assert summary["total_runs"] == 2
    assert summary["not_in_state"] == 2
    assert summary["status_counts"]["complete"] >= 1
    assert summary["status_counts"]["empty"] >= 1


def test_get_missing_state_entries(backend, temp_workspace):
    """Test getting runs that are not in state."""
    run_dir = temp_workspace.sims_dir / "run-20251107-000005"
    remd_dir = run_dir / "replica_exchange"
    remd_dir.mkdir(parents=True)

    (remd_dir / "replica_00.dcd").touch()
    (remd_dir / "demux_T300K.dcd").touch()
    (remd_dir / "analysis_results.json").write_text("{}")

    missing = backend.get_missing_state_entries()
    assert len(missing) >= 1
    assert any(v.run_id == "run-20251107-000005" for v in missing)


def test_validate_specific_run(backend, temp_workspace):
    """Test validating a specific run by ID."""
    run_dir = temp_workspace.sims_dir / "run-20251107-000006"
    remd_dir = run_dir / "replica_exchange"
    remd_dir.mkdir(parents=True)
    (remd_dir / "replica_00.dcd").touch()

    validation = backend.validate_run("run-20251107-000006")
    assert validation is not None
    assert validation.run_id == "run-20251107-000006"
    assert validation.metadata["trajectory_count"] == 1


def test_validate_nonexistent_run(backend):
    """Test validating a run that doesn't exist."""
    validation = backend.validate_run("run-nonexistent")
    assert validation is None


def test_add_discovered_run_to_state(backend, temp_workspace):
    """Test adding a discovered run to state."""
    run_dir = temp_workspace.sims_dir / "run-20251107-000007"
    remd_dir = run_dir / "replica_exchange"
    remd_dir.mkdir(parents=True)
    (remd_dir / "replica_00.dcd").touch()
    (remd_dir / "demux_T300K.dcd").touch()
    (remd_dir / "analysis_results.json").write_text("{}")

    temp_workspace.inputs_dir.mkdir(parents=True, exist_ok=True)
    pdb_file = temp_workspace.inputs_dir / "test.pdb"
    pdb_file.write_text("test pdb content")

    assert len(backend.state.runs) == 0

    success = backend.add_run_to_state("run-20251107-000007")
    assert success
    assert len(backend.state.runs) == 1
    assert backend.state.runs[0]["run_id"] == "run-20251107-000007"


def test_run_validation_issues(backend, temp_workspace):
    """Test that validation detects and reports issues."""
    run_dir = temp_workspace.sims_dir / "run-20251107-000008"
    remd_dir = run_dir / "replica_exchange"
    remd_dir.mkdir(parents=True)
    (remd_dir / "replica_00.dcd").touch()

    validations = backend.discover_all_runs()
    v = next((v for v in validations if v.run_id == "run-20251107-000008"), None)
    assert v is not None
    assert len(v.issues) > 0
    assert any(
        "demux" in issue.message.lower() or "missing" in issue.message.lower()
        for issue in v.issues
    )


def test_run_validation_with_provenance(backend, temp_workspace):
    """Test that validation extracts metadata from provenance file."""
    import json

    run_dir = temp_workspace.sims_dir / "run-20251107-000009"
    remd_dir = run_dir / "replica_exchange"
    remd_dir.mkdir(parents=True)
    (remd_dir / "replica_00.dcd").touch()
    (remd_dir / "demux_T300K.dcd").touch()
    (remd_dir / "analysis_results.json").write_text("{}")

    provenance = {
        "total_steps": 50000,
        "temperatures": [300.0, 320.0, 340.0],
    }
    (remd_dir / "provenance.json").write_text(json.dumps(provenance))

    validation = backend.validate_run("run-20251107-000009")
    assert validation is not None
    assert validation.metadata.get("total_steps") == 50000
    assert validation.metadata.get("temperatures") == [300.0, 320.0, 340.0]
