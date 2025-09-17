"""Compatibility wrapper for ``pmarlo.reduce.reducers``."""

from __future__ import annotations

import warnings

from ..markov_state_model.reduction import (
    get_available_methods,
    pca_reduce,
    reduce_features,
    tica_reduce,
    vamp_reduce,
)

__all__ = [
    "pca_reduce",
    "tica_reduce",
    "vamp_reduce",
    "reduce_features",
    "get_available_methods",
]

warnings.warn(
    "`pmarlo.reduce.reducers` is deprecated and will be removed in PMARLO 0.3. "
    "Import from `pmarlo.markov_state_model.reduction` instead.",
    DeprecationWarning,
    stacklevel=2,
)
