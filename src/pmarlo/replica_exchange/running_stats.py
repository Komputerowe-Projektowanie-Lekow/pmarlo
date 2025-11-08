# Copyright (c) 2025 PMARLO Development Team
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Running statistics tracking for vector inputs.

This module provides utilities for computing running mean and standard deviation
without storing all historical values.
"""

from typing import Sequence, Tuple

import numpy as np


class RunningStats:
    """Track running mean and variance for vector inputs in a configurable dtype."""

    def __init__(
        self, dim: int, *, dtype: np.dtype | type[np.floating] = np.float64
    ) -> None:
        if dim <= 0:
            raise ValueError("dim must be a positive integer")
        self._dim = int(dim)
        self._count = 0
        self._dtype = np.dtype(dtype)
        self._mean = np.zeros(self._dim, dtype=self._dtype)
        self._m2 = np.zeros(self._dim, dtype=self._dtype)

    @property
    def count(self) -> int:
        return self._count

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    def update(self, values: Sequence[float] | np.ndarray) -> None:
        arr = np.asarray(values).reshape(-1)
        if arr.size != self._dim:
            raise ValueError(
                f"Expected values of length {self._dim}, received {arr.size}"
            )
        if arr.dtype != self._dtype:
            arr = arr.astype(self._dtype, copy=False)
        self._count += 1
        delta = arr - self._mean
        self._mean += delta / self._count
        self._m2 += delta * (arr - self._mean)

    def summary(self, copy: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Get current mean and standard deviation.

        Args:
            copy: If True, return copies of internal arrays. If False (default),
                  return views for read-only access. Only use copy=True if the
                  caller needs to modify the returned arrays.

        Returns:
            Tuple of (mean, stddev) arrays.
        """
        if self._count == 0:
            mean = self._mean.copy() if copy else self._mean
            return mean, np.zeros(self._dim, dtype=self._dtype)
        if self._count == 1:
            mean = self._mean.copy() if copy else self._mean
            return mean, np.zeros(self._dim, dtype=self._dtype)
        variance = self._m2 / max(1, self._count - 1)
        variance = np.clip(variance, a_min=0.0, a_max=None)
        stddev = np.sqrt(variance)
        if copy:
            return self._mean.copy(), stddev
        else:
            # Return views - caller must not modify these arrays
            return self._mean, stddev
