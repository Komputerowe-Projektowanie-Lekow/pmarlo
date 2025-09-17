"""Compatibility module for the legacy ``pmarlo.pipeline`` import path."""

from __future__ import annotations

import warnings

from .transform.pipeline import Pipeline, run_pmarlo

__all__ = ["Pipeline", "run_pmarlo"]

warnings.warn(
    "`pmarlo.pipeline` is deprecated and will be removed in PMARLO 0.3. "
    "Please import from `pmarlo.transform.pipeline` instead.",
    DeprecationWarning,
    stacklevel=2,
)
