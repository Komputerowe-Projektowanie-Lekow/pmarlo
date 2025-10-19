from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Sequence

StrPath = str | os.PathLike[str]

__all__ = ["repository_root", "resolve_project_path", "ensure_directory"]


@lru_cache(maxsize=1)
def repository_root() -> Path:
    """Return the repository root detected via marker files."""
    here = Path(__file__).resolve()
    markers = ("pyproject.toml", ".git", "tox.ini")
    for ancestor in here.parents:
        if any((ancestor / marker).exists() for marker in markers):
            return ancestor
    raise RuntimeError("Unable to locate repository root from path utils")


def resolve_project_path(
    path: StrPath | None,
    *,
    search_roots: Sequence[StrPath] | None = None,
) -> str | None:
    """Resolve a possibly-relative path against common project roots.

    The resolution strategy enforces deterministic lookups:

    1. Absolute paths are normalised and must exist on disk.
    2. Relative paths are checked against the current working directory.
    3. Additional ``search_roots`` (if provided) are consulted in order.

    If the path does not exist in any candidate location, a
    :class:`FileNotFoundError` is raised immediately so callers can handle the
    failure explicitly.
    """

    if path is None:
        return None

    raw = os.fspath(path)
    candidate_path = Path(raw)

    if candidate_path.is_absolute():
        if not candidate_path.exists():
            raise FileNotFoundError(f"Path '{raw}' does not exist")
        return os.fspath(candidate_path.resolve())

    root_candidates: list[Path] = [Path.cwd()]
    if search_roots:
        root_candidates.extend(Path(os.fspath(root)).resolve() for root in search_roots)

    checked: list[Path] = []
    for root in root_candidates:
        resolved = (root / candidate_path).expanduser()
        if resolved.exists():
            return os.fspath(resolved.resolve())
        checked.append(resolved)

    searched = ", ".join(str(path.parent) for path in checked) or str(Path.cwd())
    raise FileNotFoundError(f"Could not resolve '{raw}' in search roots: {searched}")


def ensure_directory(
    path: StrPath | Path,
    *,
    parents: bool = True,
    exist_ok: bool = True,
    mode: int | None = None,
) -> Path:
    """Create *path* as a directory if it does not already exist.

    Parameters
    ----------
    path:
        Directory path to create. May be a string or :class:`os.PathLike`.
    parents:
        Whether to create parent directories. Defaults to ``True`` to match the
        historical ``mkdir(parents=True, exist_ok=True)`` idiom used across the
        codebase.
    exist_ok:
        Passed through to :meth:`pathlib.Path.mkdir`. Defaults to ``True`` to
        maintain backwards compatibility while still allowing callers to opt in
        to stricter behaviour when desired.
    mode:
        Optional POSIX permission bits to apply after creation via
        :func:`os.chmod`.

    Returns
    -------
    :class:`pathlib.Path`
        The created (or pre-existing) directory path as a :class:`Path`
        instance.
    """

    directory = Path(path)
    try:
        directory.mkdir(parents=parents, exist_ok=exist_ok)
    except FileExistsError as exc:
        if exist_ok and directory.is_dir():
            # The directory already exists (possibly created concurrently);
            # treat this as success to mirror the previous idiom.
            pass
        else:  # pragma: no cover - rely on callers/tests to cover rare branch
            raise

    if mode is not None:
        directory.chmod(mode)

    return directory
