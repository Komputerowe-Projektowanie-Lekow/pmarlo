from __future__ import annotations

from typing import Any

__all__ = ["safe_float"]


def safe_float(value: Any, default: float = 0.0) -> float:
    """Attempt to convert ``value`` to float, returning ``default`` on failure."""

    try:
        return float(value)
    except Exception:
        return float(default)
