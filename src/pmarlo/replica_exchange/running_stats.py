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
    """Track running mean and standard deviation for vector inputs."""

    def __init__(self, dim: int) -> None:
        if dim <= 0:
            raise ValueError("dim must be a positive integer")
        self._dim = int(dim)
        self._count = 0
        self._mean = np.zeros(self._dim, dtype=float)
        self._m2 = np.zeros(self._dim, dtype=float)

    @property
    def count(self) -> int:
        return self._count

    def update(self, values: Sequence[float] | np.ndarray) -> None:
        arr = np.asarray(values, dtype=float).reshape(-1)
        if arr.size != self._dim:
            raise ValueError(
                f"Expected values of length {self._dim}, received {arr.size}"
            )
        self._count += 1
        delta = arr - self._mean
        self._mean += delta / self._count
        self._m2 += delta * (arr - self._mean)

    def summary(self) -> Tuple[np.ndarray, np.ndarray]:
        if self._count == 0:
            return self._mean.copy(), np.zeros(self._dim, dtype=float)
        if self._count == 1:
            return self._mean.copy(), np.zeros(self._dim, dtype=float)
        variance = self._m2 / max(1, self._count - 1)
        variance = np.clip(variance, a_min=0.0, a_max=None)
        return self._mean.copy(), np.sqrt(variance)
