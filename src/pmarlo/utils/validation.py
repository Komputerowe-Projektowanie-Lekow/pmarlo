"""Common validation helpers shared across PMARLO modules."""

from __future__ import annotations

__all__ = ["require"]


def require(condition: bool, message: str) -> None:
    """Raise ``ValueError`` when a required condition is not satisfied."""
    if not condition:
        raise ValueError(message)
