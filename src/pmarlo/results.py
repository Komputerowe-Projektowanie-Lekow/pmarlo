"""Compatibility shim for the legacy ``pmarlo.results`` module."""

from __future__ import annotations

import warnings

from .markov_state_model.free_energy import FESResult, PMFResult
from .markov_state_model.results import (
    BaseResult,
    CKResult,
    ClusteringResult,
    DemuxResult,
    ITSResult,
    MSMResult,
    REMDResult,
)

__all__ = [
    "BaseResult",
    "REMDResult",
    "DemuxResult",
    "ClusteringResult",
    "MSMResult",
    "CKResult",
    "ITSResult",
    "FESResult",
    "PMFResult",
]

warnings.warn(
    "`pmarlo.results` is deprecated and will be removed in PMARLO 0.3. "
    "Import from `pmarlo.markov_state_model.results` or "
    "`pmarlo.markov_state_model.free_energy` instead.",
    DeprecationWarning,
    stacklevel=2,
)
