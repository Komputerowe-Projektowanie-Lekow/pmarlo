"""Deprecated scheduler helpers."""

from __future__ import annotations

import warnings

__all__: list[str] = []

warnings.warn(
    "pmarlo.features.deeptica_trainer.schedulers is deprecated; schedulers are "
    "managed by pmarlo.ml.deeptica.trainer.",
    DeprecationWarning,
    stacklevel=2,
)
