"""Project-specific exception hierarchy."""

from __future__ import annotations


class TemperatureConsistencyError(ValueError):
    """Raised when trajectory metadata mixes incompatible temperatures."""
