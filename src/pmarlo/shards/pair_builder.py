from __future__ import annotations

"""Pair construction utilities for time-lagged learning."""

import numpy as np

from .schema import Shard

__all__ = ["PairBuilder"]


class PairBuilder:
    """Build (t, t+t) index pairs inside a single shard."""

    def __init__(self, tau_steps: int) -> None:
        if tau_steps <= 0:
            raise ValueError("tau_steps must be > 0")
        self.tau = int(tau_steps)

    def make_pairs(self, shard: Shard) -> np.ndarray:
        """Return contiguous pairs within a shard with no boundary crossings."""

        n_frames = shard.meta.n_frames
        if n_frames <= self.tau:
            return np.empty((0, 2), dtype=np.int64)
        idx0 = np.arange(0, n_frames - self.tau, dtype=np.int64)
        idx1 = idx0 + self.tau
        pairs = np.stack([idx0, idx1], axis=1)
        return pairs
