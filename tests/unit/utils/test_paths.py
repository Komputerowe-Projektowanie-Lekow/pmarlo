from __future__ import annotations

import os
from pathlib import Path

from pmarlo.utils.path_utils import repository_root, resolve_project_path


def test_resolve_project_path_preserves_absolute_inputs(tmp_path):
    sample = tmp_path / "traj.dcd"
    sample.write_text("")

    resolved = resolve_project_path(sample)

    assert resolved == os.fspath(sample)


def test_resolve_project_path_prefers_current_working_directory(tmp_path):
    """Test path resolution with files in temporary working directory."""
    original_cwd = Path.cwd()
    try:
        os.chdir(tmp_path)
        relative = Path("data.bin")
        relative.write_text("payload")

        resolved = resolve_project_path("data.bin")

        assert resolved == os.fspath(relative.resolve())
    finally:
        os.chdir(original_cwd)


def test_resolve_project_path_falls_back_to_repository_root(tmp_path):
    """Test fallback to repository root for project files."""
    original_cwd = Path.cwd()
    try:
        os.chdir(tmp_path)

        resolved = resolve_project_path("pyproject.toml")

        assert resolved.endswith("pyproject.toml")
        assert Path(resolved).exists()
        assert Path(resolved).parent == repository_root()
    finally:
        os.chdir(original_cwd)


def test_resolve_project_path_uses_additional_search_roots(tmp_path):
    """Test path resolution with additional search roots."""
    extra_root = tmp_path / "assets"
    extra_root.mkdir()
    target = extra_root / "topology.pdb"
    target.write_text("")

    original_cwd = Path.cwd()
    try:
        os.chdir(tmp_path)

        resolved = resolve_project_path("topology.pdb", search_roots=[extra_root])

        assert resolved == os.fspath(target.resolve())
    finally:
        os.chdir(original_cwd)


def test_resolve_project_path_returns_original_for_missing_files(tmp_path):
    """Test that missing files return the original path string."""
    original_cwd = Path.cwd()
    try:
        os.chdir(tmp_path)
        missing = "does/not/exist.ext"

        resolved = resolve_project_path(missing)

        assert resolved == missing
    finally:
        os.chdir(original_cwd)
