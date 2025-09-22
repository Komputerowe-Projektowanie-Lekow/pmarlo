"""Generate developer-friendly reports based on ``pygount`` statistics."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, MutableMapping


class PygountError(RuntimeError):
    """Raised when invoking ``pygount`` fails."""


@dataclass(slots=True)
class LangTotals:
    """Aggregate counters reported by ``pygount`` for a single language."""

    files: int
    code: int
    comment: int
    source: int


@dataclass(slots=True)
class FolderSummary:
    """Summary statistics for a scanned folder."""

    path: Path
    files: int
    code: int
    comment: int


PygountRunner = Callable[[Path, Sequence[str] | None], Mapping[str, Any]]

DEFAULT_FOLDERS = (
    Path("tests"),
    Path("src"),
    Path("example_programs"),
)
DEFAULT_SUFFIXES = ("py", "ipynb")


def run_pygount_json(
    path: Path, suffixes: Sequence[str] | None = None
) -> Mapping[str, Any]:
    """Invoke ``pygount`` in JSON mode and return its decoded payload."""

    cmd: list[str] = ["pygount", "--format=json", "--out=STDOUT", str(path)]
    if suffixes:
        cmd.insert(1, f"--suffix={','.join(suffixes)}")
    try:
        completed = subprocess.run(  # noqa: S603, S607 - intentional CLI usage
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:  # pragma: no cover - depends on local env
        raise PygountError(
            "Command 'pygount' not found. Install it with 'pip install pygount'."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise PygountError(
            f"pygount failed for '{path}': {exc.stderr or exc.stdout}."
        ) from exc

    stdout = completed.stdout.strip()
    if not stdout:
        return {}
    try:
        return json.loads(stdout)
    except json.JSONDecodeError as exc:
        snippet = stdout[:200].replace("\n", " ")
        raise PygountError(
            f"Unable to decode JSON from pygount output for '{path}': {exc}. Snippet: {snippet}"
        ) from exc


def _merge_language_totals(
    aggregated: MutableMapping[str, LangTotals],
    languages: Iterable[Mapping[str, Any]],
) -> None:
    for entry in languages:
        if not isinstance(entry, Mapping):
            continue
        name = str(entry.get("language", "__unknown__"))
        totals = LangTotals(
            files=int(entry.get("fileCount", 0) or 0),
            code=int(entry.get("codeCount", 0) or 0),
            comment=int(entry.get("documentationCount", 0) or 0),
            source=int(entry.get("sourceCount", 0) or 0),
        )
        previous = aggregated.get(name)
        if previous is None:
            aggregated[name] = totals
        else:
            aggregated[name] = LangTotals(
                files=previous.files + totals.files,
                code=previous.code + totals.code,
                comment=previous.comment + totals.comment,
                source=previous.source + totals.source,
            )


def collect_language_statistics(
    paths: Sequence[Path],
    *,
    suffixes: Sequence[str] | None = None,
    runner: PygountRunner = run_pygount_json,
) -> tuple[Mapping[str, LangTotals], Sequence[FolderSummary]]:
    """Collect ``pygount`` statistics for each path and aggregate per-language totals."""

    aggregated: MutableMapping[str, LangTotals] = {}
    folder_summaries: list[FolderSummary] = []

    for path in paths:
        if not path.exists():
            folder_summaries.append(
                FolderSummary(path=path, files=0, code=0, comment=0)
            )
            continue
        data = runner(path, suffixes)
        summary: Mapping[str, Any]
        languages: Iterable[Mapping[str, Any]]
        if isinstance(data, Mapping):
            summary_obj = data.get("summary", {})
            summary = summary_obj if isinstance(summary_obj, Mapping) else {}
            languages_obj = data.get("languages", [])
            languages = languages_obj if isinstance(languages_obj, Iterable) else []
        else:
            summary = {}
            languages = []
        files = int(summary.get("totalFileCount", 0) or 0)
        code = int(summary.get("totalCodeCount", 0) or 0)
        comment = int(summary.get("totalDocumentationCount", 0) or 0)
        folder_summaries.append(
            FolderSummary(path=path, files=files, code=code, comment=comment)
        )
        if isinstance(languages, Iterable):
            _merge_language_totals(aggregated, languages)

    return aggregated, folder_summaries


def build_language_rows(
    aggregated: Mapping[str, LangTotals],
) -> list[tuple[str, int, int, int]]:
    """Convert aggregated language totals into a sorted table."""

    rows: list[tuple[str, int, int, int]] = []
    for name, totals in sorted(
        aggregated.items(), key=lambda kv: kv[1].code, reverse=True
    ):
        rows.append((name, totals.files, totals.code, totals.comment))
    if aggregated:
        total_files = sum(t.files for t in aggregated.values())
        total_code = sum(t.code for t in aggregated.values())
        total_comment = sum(t.comment for t in aggregated.values())
        rows.append(("Sum", total_files, total_code, total_comment))
    return rows


def format_table(rows: Sequence[tuple[str, int, int, int]]) -> str:
    """Render totals in a simple box-drawing table."""

    headers = ("Language", "Files", "Code", "Comment")
    rows = list(rows)
    if not rows:
        rows = [("—", 0, 0, 0)]
    col_widths = [
        max(len(headers[0]), max((len(str(r[0])) for r in rows), default=0)),
        max(len(headers[1]), max((len(str(r[1])) for r in rows), default=0)),
        max(len(headers[2]), max((len(str(r[2])) for r in rows), default=0)),
        max(len(headers[3]), max((len(str(r[3])) for r in rows), default=0)),
    ]

    def sep(ch: str = "─", cross: str = "┼", left: str = "├", right: str = "┤") -> str:
        parts = [ch * (width + 2) for width in col_widths]
        return left + cross.join(parts) + right

    def top() -> str:
        return sep(ch="─", cross="┬", left="┌", right="┐")

    def bottom() -> str:
        return sep(ch="─", cross="┴", left="└", right="┘")

    def fmt_row(values: Sequence[object]) -> str:
        cells = [
            f" {str(value).ljust(width)} " for value, width in zip(values, col_widths)
        ]
        return "│" + "│".join(cells) + "│"

    body = [fmt_row(row) for row in rows]
    return "\n".join([top(), fmt_row(headers), sep(), *body, bottom()])


def format_folder_summary(summaries: Sequence[FolderSummary]) -> str:
    """Render folder summaries similar to the legacy script."""

    lines: list[str] = []
    for summary in summaries:
        lines.append(f"- {summary.path}")
        lines.append(
            f"  Files: {summary.files} | Code: {summary.code} | Comment: {summary.comment}"
        )
    return "\n".join(lines)


def generate_report(
    paths: Sequence[Path],
    *,
    suffixes: Sequence[str] | None = None,
    runner: PygountRunner = run_pygount_json,
) -> str:
    """Generate a multi-section text report for the given paths."""

    aggregated, folder_summaries = collect_language_statistics(
        paths, suffixes=suffixes, runner=runner
    )
    language_rows = build_language_rows(aggregated)
    total_code = sum(t.code for t in aggregated.values())
    report_lines = [
        "=== ZBIORCZY RAPORT (wszystkie folderdly razem) ===",
        "",
        format_table(language_rows),
        "",
        "=== KRÓTKIE PODSUMOWANIE PER FOLDER ===",
        "",
        format_folder_summary(folder_summaries),
        "",
        f"ŁĄCZNA liczba linii kodu (Code) we wszystkich folderach: {total_code}",
    ]
    return "\n".join(report_lines)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=DEFAULT_FOLDERS,
        help="Folders to scan (default: tests src example_programs).",
    )
    parser.add_argument(
        "--suffix",
        action="append",
        dest="suffixes",
        help="Restrict pygount to the given suffix (can be provided multiple times).",
    )
    parser.add_argument(
        "--all-suffixes",
        action="store_true",
        help="Disable suffix filtering and scan all files that pygount supports.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point used by the ``pmarlo-lines-report`` console script."""

    args = _parse_args(argv)
    suffixes: Sequence[str] | None
    if args.all_suffixes:
        suffixes = None
    elif args.suffixes:
        suffixes = args.suffixes
    else:
        suffixes = DEFAULT_SUFFIXES

    try:
        report = generate_report(args.paths, suffixes=suffixes)
    except PygountError as exc:
        print(str(exc))
        return 1

    for path in args.paths:
        print(f"Analiza folderu: {path}")
    print("\n" + report)
    return 0


if __name__ == "__main__":  # pragma: no cover - manual invocation only
    raise SystemExit(main())
