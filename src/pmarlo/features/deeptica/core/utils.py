from __future__ import annotations

from typing import Any

__all__ = ["safe_float"]


def safe_float(value: Any, default: float = 0.0) -> float:
    """Convert ``value`` to ``float`` or raise a ``ValueError``."""

    try:
        return float(value)
    except Exception as exc:  # pragma: no cover - conversion errors are rare
        raise ValueError(f"Cannot convert {value!r} to float") from exc
