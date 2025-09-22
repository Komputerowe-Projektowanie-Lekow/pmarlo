"""Developer-facing utilities and thin CLI entry points.

This package hosts helper functions that are useful while maintaining
:mod:`pmarlo`. Core logic lives here, while the corresponding console
scripts are declared in :mod:`pyproject.toml` and keep their command-line
interfaces minimal.
"""

from __future__ import annotations

from . import lines_report
from .check_extras_parity import check_extras_parity

__all__ = ["check_extras_parity", "lines_report"]
