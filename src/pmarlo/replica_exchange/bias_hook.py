from __future__ import annotations

"""Protocol defining CV?bias hooks for replica-exchange simulations."""

from typing import Protocol

import numpy as np

__all__ = ["BiasHook"]


class BiasHook(Protocol):
    """Callable mapping CV values to per-frame bias potentials."""

    def __call__(self, cv_values: np.ndarray) -> np.ndarray:
        """Return bias potentials (shape: [frames] or [frames, 1]) for the supplied CV samples."""
        ...
