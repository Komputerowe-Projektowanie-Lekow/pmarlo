from __future__ import annotations

import os
from pathlib import Path

import pytest

from pmarlo.utils.path_utils import repository_root, resolve_project_path


def test_resolve_project_path_preserves_absolute_inputs(tmp_path):
    sample = tmp_path / "traj.dcd"
    sample.write_text("")

    resolved = resolve_project_path(sample)

    assert resolved == os.fspath(sample.resolve())


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


def test_resolve_project_path_requires_explicit_search_roots(tmp_path):
    original_cwd = Path.cwd()
    try:
        os.chdir(tmp_path)

        root = repository_root()
        resolved = resolve_project_path(
            "pyproject.toml", search_roots=[root]
        )

        assert resolved == os.fspath((root / "pyproject.toml").resolve())
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


def test_resolve_project_path_raises_for_missing_files(tmp_path):
    """Missing files trigger a FileNotFoundError."""

    original_cwd = Path.cwd()
    try:
        os.chdir(tmp_path)
        missing = "does/not/exist.ext"

        with pytest.raises(FileNotFoundError):
            resolve_project_path(missing)
    finally:
        os.chdir(original_cwd)
