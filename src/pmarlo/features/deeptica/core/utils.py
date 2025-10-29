from __future__ import annotations

from typing import Any

__all__ = ["safe_float"]


def safe_float(value: Any) -> float:
    """Convert ``value`` to ``float`` or raise a descriptive :class:`ValueError`."""

    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        if value is None:
            message = "Cannot convert None to float: value is missing."
        else:
            message = f"Cannot convert {value!r} to float."
        raise ValueError(message) from exc
