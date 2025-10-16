"""Parity checks between Poetry extras and PEP 621 optional dependencies."""

from __future__ import annotations

import argparse
import tomllib
from pathlib import Path
from typing import Iterable, Sequence


class ExtrasParityError(RuntimeError):
    """Raised when the ``pyproject.toml`` file cannot be processed."""


def _load_pyproject(path: Path) -> dict:
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:  # pragma: no cover - handled by caller
        raise ExtrasParityError(f"Could not find pyproject file at {path!s}.") from exc
    except OSError as exc:  # pragma: no cover - filesystem issues are rare
        raise ExtrasParityError(f"Could not read {path!s}: {exc}.") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ExtrasParityError(f"Failed to parse {path!s}: {exc}.") from exc
    if not isinstance(data, dict):
        raise ExtrasParityError(
            "Parsed pyproject data is not a table; expected a dictionary structure."
        )
    return data


def _extract_extra_names(section: object) -> set[str]:
    if section is None:
        return set()
    if not isinstance(section, dict):
        raise ExtrasParityError(
            "Extras definitions must be provided as mappings of extra name to dependencies."
        )
    return {
        key
        for key, value in section.items()
        if isinstance(key, str) and value is not None
    }


def _describe_missing(
    extras: Iterable[str], origin: str, counterpart: str
) -> list[str]:
    missing = sorted(set(extras))
    if not missing:
        return []
    if len(missing) == 1:
        return [
            f"Extra '{missing[0]}' is declared in {origin} but missing from {counterpart}."
        ]
    return [
        f"Extras {', '.join(sorted(missing))} are declared in {origin} but missing from {counterpart}."
    ]


def check_extras_parity(
    pyproject_path: str = "pyproject.toml",
) -> tuple[bool, list[str]]:
    """Compare extra names declared for Poetry and PEP 621 metadata."""

    path = Path(pyproject_path)
    try:
        data = _load_pyproject(path)
    except ExtrasParityError as exc:
        return False, [str(exc)]

    project_optional = (
        data.get("project", {}).get("optional-dependencies")
        if isinstance(data.get("project"), dict)
        else {}
    )
    poetry_extras = (
        data.get("tool", {}).get("poetry", {}).get("extras")
        if isinstance(data.get("tool"), dict)
        else {}
    )

    try:
        project_extra_names = _extract_extra_names(project_optional)
        poetry_extra_names = _extract_extra_names(poetry_extras)
    except ExtrasParityError as exc:
        return False, [str(exc)]

    problems: list[str] = []
    problems.extend(
        _describe_missing(
            project_extra_names - poetry_extra_names,
            "[project.optional-dependencies]",
            "[tool.poetry.extras]",
        )
    )
    problems.extend(
        _describe_missing(
            poetry_extra_names - project_extra_names,
            "[tool.poetry.extras]",
            "[project.optional-dependencies]",
        )
    )

    ok = not problems
    return ok, problems


def _format_report(ok: bool, problems: Sequence[str]) -> str:
    if ok:
        return "Extras definitions are in parity between Poetry and PEP 621 metadata."
    bullet_list = "\n".join(f" - {problem}" for problem in problems)
    return "Extras parity check failed:\n" + bullet_list


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pyproject",
        default="pyproject.toml",
        help="Path to the pyproject.toml file to inspect (default: %(default)s).",
    )
    args = parser.parse_args(argv)

    ok, problems = check_extras_parity(args.pyproject)
    print(_format_report(ok, problems))
    return 0 if ok else 1


if __name__ == "__main__":  # pragma: no cover - manual invocation only
    raise SystemExit(main())
