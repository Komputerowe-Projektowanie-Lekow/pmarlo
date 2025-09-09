"""Dimensionality reduction utilities (PCA, TICA, VAMP).

.. deprecated::
   This module has been moved. Use :mod:`pmarlo.markov_state_model.reduction` for
   reduction functionality. This compatibility shim will be removed in a future release.
"""

import warnings

# Re-export from new location with deprecation warnings
from ..markov_state_model.reduction import (  # noqa: F401
    pca_reduce,
    tica_reduce,
    vamp_reduce,
    reduce_features,
    get_available_methods,
)

# Issue deprecation warning when this module is imported
warnings.warn(
    "pmarlo.reduce is deprecated. Use pmarlo.markov_state_model.reduction "
    "for reduction functionality. This compatibility shim will be "
    "removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "pca_reduce",
    "tica_reduce",
    "vamp_reduce",
    "reduce_features",
    "get_available_methods",
]
