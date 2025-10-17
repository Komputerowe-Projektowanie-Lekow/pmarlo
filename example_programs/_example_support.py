"""Helper utilities shared by the example scripts.

These helpers make it possible to run the examples directly from a cloned
repository without requiring the project to be installed in editable mode.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC_PATH = _PROJECT_ROOT / "src"
_ASSETS_DIR = _PROJECT_ROOT / "tests" / "_assets"


def ensure_src_on_path() -> Path:
    """Ensure the project's ``src`` directory is available on ``sys.path``.

    Returns the path that was inserted so callers can reuse it if needed.
    """

    src_str = str(_SRC_PATH)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)
    return _SRC_PATH


def project_root() -> Path:
    """Return the root directory of the project checkout."""

    return _PROJECT_ROOT


def assets_path(*relative: str) -> Path:
    """Return a path under ``tests/_assets`` joined with ``relative`` parts."""

    return _ASSETS_DIR.joinpath(*relative)
