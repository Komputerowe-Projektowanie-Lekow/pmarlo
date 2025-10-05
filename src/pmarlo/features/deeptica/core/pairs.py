"""Backwards-compatible re-export of lagged pair helpers."""

from __future__ import annotations

from pmarlo.pairs.core import PairInfo, build_pair_info

__all__ = ["PairInfo", "build_pair_info"]
