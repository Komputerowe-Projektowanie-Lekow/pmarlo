from __future__ import annotations

import numpy as np
import numpy.typing as npt

# Centralized NumPy typing aliases used across the codebase.

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]

__all__ = ["FloatArray", "IntArray", "BoolArray"]
