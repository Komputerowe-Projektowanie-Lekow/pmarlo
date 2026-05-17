from __future__ import annotations

from typing import TypeAlias

import numpy as np
import numpy.typing as npt

# Centralized NumPy typing aliases used across the codebase.
#
# These aliases describe runtime NumPy arrays after validation/conversion.
# For raw user inputs, prefer npt.ArrayLike.

FloatArray: TypeAlias = npt.NDArray[np.float64]
IntArray: TypeAlias = npt.NDArray[np.int64]
BoolArray: TypeAlias = npt.NDArray[np.bool_]

__all__ = [
    "FloatArray",
    "IntArray",
    "BoolArray",
]
