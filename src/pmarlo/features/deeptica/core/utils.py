from __future__ import annotations

from typing import Any

__all__ = ["safe_float"]


def safe_float(value: Any, default: float = 0.0) -> float:
    """Convert ``value`` to ``float`` falling back to ``default`` when needed."""

    try:
        return float(value)
    except Exception:
        try:
            return float(default)
        except Exception as exc:  # pragma: no cover - default conversion errors are rare
            raise ValueError(
                f"Cannot convert {value!r} to float and default {default!r} is invalid"
            ) from exc
