"""Samplers for balanced batch selection across temperatures."""

from __future__ import annotations

import warnings

from pmarlo.samplers import BalancedTempSampler

__all__ = ["BalancedTempSampler"]

warnings.warn(
    "pmarlo.features.samplers is deprecated; import BalancedTempSampler from "
    "pmarlo.samplers instead.",
    DeprecationWarning,
    stacklevel=2,
)
