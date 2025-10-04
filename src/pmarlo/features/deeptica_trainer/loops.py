"""Legacy helpers retained for backwards compatibility."""

from __future__ import annotations

import warnings

__all__: list[str] = []

warnings.warn(
    "pmarlo.features.deeptica_trainer.loops is deprecated; use "
    "pmarlo.ml.deeptica.trainer instead.",
    DeprecationWarning,
    stacklevel=2,
)
