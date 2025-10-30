"""Utilities for consistently coercing primitive types."""

from __future__ import annotations

from typing import Any

import math

__all__ = ["coerce_finite_float", "coerce_finite_float_with_default"]


def coerce_finite_float(value: Any) -> float | None:
    """Return ``value`` as a finite ``float`` when possible."""

    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def coerce_finite_float_with_default(value: Any, *, default: float) -> float:
    """Return ``value`` coerced to ``float`` or ``default`` when conversion fails."""

    result = coerce_finite_float(value)
    if result is None:
        return float(default)
    return result
