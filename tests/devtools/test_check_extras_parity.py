from __future__ import annotations

from pathlib import Path

from tools.check_extras_parity import check_extras_parity


def write_pyproject(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "pyproject.toml"
    path.write_text(content, encoding="utf-8")
    return path


def test_perfect_parity(tmp_path: Path) -> None:
    path = write_pyproject(
        tmp_path,
        """
[project]
optional-dependencies = {test = ["pytest"]}

[tool.poetry]
extras = {test = ["pytest"]}
""",
    )

    ok, problems = check_extras_parity(str(path))

    assert ok is True
    assert problems == []


def test_missing_in_poetry_section(tmp_path: Path) -> None:
    path = write_pyproject(
        tmp_path,
        """
[project]
optional-dependencies = {app = ["streamlit"]}

[tool.poetry]
extras = {}
""",
    )

    ok, problems = check_extras_parity(str(path))

    assert ok is False
    assert "app" in problems[0]
    assert "[tool.poetry.extras]" in problems[0]


def test_missing_in_project_section(tmp_path: Path) -> None:
    path = write_pyproject(
        tmp_path,
        """
[project]
optional-dependencies = {}

[tool.poetry]
extras = {analysis = ["pandas"]}
""",
    )

    ok, problems = check_extras_parity(str(path))

    assert ok is False
    assert "analysis" in problems[0]
    assert "[project.optional-dependencies]" in problems[-1]


def test_malformed_extras_table(tmp_path: Path) -> None:
    path = write_pyproject(
        tmp_path,
        """
[project]
optional-dependencies = ["not-a-table"]
""",
    )

    ok, problems = check_extras_parity(str(path))

    assert ok is False
    assert "Extras definitions must be provided" in problems[0]


def test_invalid_toml(tmp_path: Path) -> None:
    path = write_pyproject(
        tmp_path,
        """
[project
name = "invalid"
""",
    )

    ok, problems = check_extras_parity(str(path))

    assert ok is False
    assert "Failed to parse" in problems[0]
