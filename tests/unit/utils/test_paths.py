from __future__ import annotations

import os
from pathlib import Path

from pmarlo.utils.path_utils import repository_root, resolve_project_path


def test_resolve_project_path_preserves_absolute_inputs(tmp_path):
    sample = tmp_path / "traj.dcd"
    sample.write_text("")

    resolved = resolve_project_path(sample)

    assert resolved == os.fspath(sample)


def test_resolve_project_path_prefers_current_working_directory(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    relative = Path("data.bin")
    relative.write_text("payload")

    resolved = resolve_project_path("data.bin")

    assert resolved == os.fspath(relative.resolve())


def test_resolve_project_path_falls_back_to_repository_root(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    resolved = resolve_project_path("pyproject.toml")

    assert resolved.endswith("pyproject.toml")
    assert Path(resolved).exists()
    assert Path(resolved).parent == repository_root()


def test_resolve_project_path_uses_additional_search_roots(monkeypatch, tmp_path):
    extra_root = tmp_path / "assets"
    extra_root.mkdir()
    target = extra_root / "topology.pdb"
    target.write_text("")

    monkeypatch.chdir(tmp_path)

    resolved = resolve_project_path("topology.pdb", search_roots=[extra_root])

    assert resolved == os.fspath(target.resolve())


def test_resolve_project_path_returns_original_for_missing_files(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    missing = "does/not/exist.ext"

    resolved = resolve_project_path(missing)

    assert resolved == missing
