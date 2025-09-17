"""Compatibility shim for legacy ``pmarlo.reduce`` imports."""

from __future__ import annotations

import sys
import types
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
    "`pmarlo.reduce` is deprecated and will be removed in a future release. "
    "Import from `pmarlo.markov_state_model.reduction` instead.",
    DeprecationWarning,
    stacklevel=2,
)

_NAMES = tuple(__all__)

class _ReducersModule(types.ModuleType):
    __all__ = list(_NAMES)

    def __getattr__(self, name: str):
        if name in _NAMES:
            warnings.warn(
                "`pmarlo.reduce.reducers` is deprecated; use "
                "`pmarlo.markov_state_model.reduction`.",
                DeprecationWarning,
                stacklevel=2,
            )
            value = globals()[name]
            setattr(self, name, value)
            return value
        raise AttributeError(name)

sys.modules.setdefault("pmarlo.reduce.reducers", _ReducersModule("pmarlo.reduce.reducers"))
