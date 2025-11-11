from __future__ import annotations

from pathlib import Path

import pytest

from pmarlo.transform.build import BuildOpts, _resolve_diagnostics_dir


def test_resolve_diagnostics_dir_prefers_opts(tmp_path: Path) -> None:
    custom_dir = tmp_path / "custom_diag"
    opts = BuildOpts(diagnostics_dir=str(custom_dir))
    resolved = _resolve_diagnostics_dir({}, opts)
    assert resolved == custom_dir
    assert resolved.is_dir()


def test_resolve_diagnostics_dir_uses_dataset_output_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    dataset_output = tmp_path / "analysis" / "msm"
    dataset = {"output_dir": str(dataset_output)}
    resolved = _resolve_diagnostics_dir(dataset, BuildOpts())
    assert resolved == dataset_output / "diagnostics"
    assert resolved.is_dir()


def test_resolve_diagnostics_dir_env_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    env_dir = tmp_path / "env_diag"
    monkeypatch.setenv("PMARLO_DIAGNOSTICS_DIR", str(env_dir))
    monkeypatch.delenv("PMARLO_OUTPUT_ROOT", raising=False)
    resolved = _resolve_diagnostics_dir({}, BuildOpts())
    assert resolved == env_dir
    assert resolved.is_dir()


def test_resolve_diagnostics_dir_example_programs_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("PMARLO_DIAGNOSTICS_DIR", raising=False)
    monkeypatch.delenv("PMARLO_OUTPUT_ROOT", raising=False)
    programs_output = tmp_path / "example_programs" / "programs_outputs"
    programs_output.mkdir(parents=True)
    resolved = _resolve_diagnostics_dir({}, BuildOpts())
    assert resolved == programs_output / "diagnostics"
    assert resolved.is_dir()


def test_resolve_diagnostics_dir_default_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("PMARLO_DIAGNOSTICS_DIR", raising=False)
    monkeypatch.delenv("PMARLO_OUTPUT_ROOT", raising=False)
    resolved = _resolve_diagnostics_dir({}, BuildOpts())
    expected = tmp_path / "experiments_output" / "diagnostics"
    assert resolved == expected
    assert resolved.is_dir()
