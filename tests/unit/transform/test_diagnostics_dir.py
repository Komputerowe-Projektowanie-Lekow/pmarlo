from __future__ import annotations

from pathlib import Path

from pmarlo.transform.build import BuildOpts, _resolve_diagnostics_dir


def test_resolve_diagnostics_dir_prefers_example_programs(tmp_path, monkeypatch):
    project_root = tmp_path / "proj"
    project_root.mkdir()
    example_dir = project_root / "example_programs"
    example_dir.mkdir()
    source_file = project_root / "source.dcd"
    source_file.touch()

    monkeypatch.chdir(project_root)

    dataset = {"__shards__": [{"source_path": str(source_file)}]}
    opts = BuildOpts()

    diagnostics_dir = _resolve_diagnostics_dir(dataset, opts)

    assert diagnostics_dir == example_dir / "diagnostics"
    assert diagnostics_dir.exists()


def test_resolve_diagnostics_dir_prefers_top_level_folder(tmp_path, monkeypatch):
    project_root = tmp_path / "proj"
    fixtures_dir = project_root / "tests" / "fixtures"
    fixtures_dir.mkdir(parents=True)
    source_file = fixtures_dir / "traj.dcd"
    source_file.touch()

    monkeypatch.chdir(project_root)

    dataset = {"__shards__": [{"source_path": str(source_file)}]}
    opts = BuildOpts()

    diagnostics_dir = _resolve_diagnostics_dir(dataset, opts)

    assert diagnostics_dir == project_root / "tests" / "diagnostics"
    assert diagnostics_dir.exists()


def test_resolve_diagnostics_dir_defaults_to_experiments_output(tmp_path, monkeypatch):
    project_root = tmp_path / "proj"
    project_root.mkdir()
    source_file = project_root / "dataset.dcd"
    source_file.touch()

    monkeypatch.chdir(project_root)

    dataset = {"__shards__": [{"source_path": str(source_file)}]}
    opts = BuildOpts()

    diagnostics_dir = _resolve_diagnostics_dir(dataset, opts)

    expected = project_root / "experiments_output" / "diagnostics"
    assert diagnostics_dir == expected
    assert diagnostics_dir.exists()
