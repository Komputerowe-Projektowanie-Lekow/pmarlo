"""Shared analysis-specific error types."""

from __future__ import annotations

__all__ = ["CountingLogicError", "PruningFailedError"]


class CountingLogicError(RuntimeError):
    """Raised when pair-counting invariants are violated."""


class PruningFailedError(RuntimeError):
    """Raised when microstate pruning fails to resolve singular transition rows."""
