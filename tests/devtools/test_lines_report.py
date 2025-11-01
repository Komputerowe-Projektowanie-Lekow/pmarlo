from __future__ import annotations

from pathlib import Path
from typing import Mapping

import pytest

from scripts.lines_report import (
    FolderSummary,
    LangTotals,
    build_language_rows,
    collect_language_statistics,
    format_folder_summary,
    format_table,
    generate_report,
)


@pytest.fixture()
def fake_runner() -> dict[str, Mapping[str, object]]:
    def runner(path: Path, suffixes: tuple[str, ...] | None) -> Mapping[str, object]:
        return {
            "summary": {
                "totalFileCount": 1,
                "totalCodeCount": 10,
                "totalDocumentationCount": 3,
            },
            "languages": [
                {
                    "language": f"lang-{path.name}",
                    "fileCount": 1,
                    "codeCount": 10,
                    "documentationCount": 3,
                    "sourceCount": 13,
                }
            ],
        }

    return runner


def test_collect_language_statistics_aggregates(fake_runner, tmp_path: Path) -> None:
    paths = [tmp_path / "alpha", tmp_path / "beta"]
    for path in paths:
        path.mkdir()
    aggregated, summaries = collect_language_statistics(paths, runner=fake_runner)

    assert {name for name in aggregated} == {"lang-alpha", "lang-beta"}
    assert all(isinstance(summary, FolderSummary) for summary in summaries)


def test_build_language_rows_appends_total() -> None:
    rows = build_language_rows(
        {
            "python": LangTotals(files=2, code=100, comment=5, source=105),
            "markdown": LangTotals(files=1, code=10, comment=50, source=60),
        }
    )

    assert rows[-1] == ("Sum", 3, 110, 55)
    assert rows[0][0] == "python"


def test_format_table_handles_empty_rows() -> None:
    table = format_table([])
    assert "â€”" in table


def test_format_folder_summary_round_trip() -> None:
    summary = format_folder_summary(
        [
            FolderSummary(path=Path("alpha"), files=1, code=10, comment=3),
        ]
    )

    assert "alpha" in summary
    assert "Files: 1" in summary


def test_generate_report_combines_sections(fake_runner) -> None:
    report = generate_report([Path("alpha")], runner=fake_runner)
    assert "ZBIORCZY RAPORT" in report
    assert "KRÓTKIE PODSUMOWANIE" in report
