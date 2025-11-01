"""Common validation helpers shared across PMARLO modules."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

__all__ = ["require", "all_finite", "any_finite", "finite_or_none"]


def require(condition: bool, message: str) -> None:
    """Raise ``ValueError`` when a required condition is not satisfied."""
    if not condition:
        raise ValueError(message)


def _to_ndarray(values: Any) -> np.ndarray:
    """Convert *values* to a :class:`~numpy.ndarray` without copying when possible."""

    if isinstance(values, np.ndarray):
        return values
    return np.asarray(values)


def all_finite(values: Any) -> bool:
    """Return ``True`` when all numeric entries are finite.

    Empty arrays are treated as finite to keep downstream shape handling simple.
    """

    arr = _to_ndarray(values)
    if arr.size == 0:
        return True
    return bool(np.isfinite(arr).all())


def any_finite(values: Any) -> bool:
    """Return ``True`` if at least one finite numeric entry exists in *values*."""

    arr = _to_ndarray(values)
    if arr.size == 0:
        return False
    return bool(np.isfinite(arr).any())


def finite_or_none(value: Any) -> Optional[float]:
    """Return the value coerced to ``float`` when finite, otherwise ``None``.

    Non-numeric inputs, NaNs and infinite values all return ``None``. This mirrors
    the semantics required by KPI helpers when persisting optional metrics.
    """

    if value is None:
        return None
    try:
        as_float = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(as_float):
        return None
    return as_float
