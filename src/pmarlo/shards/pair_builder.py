from __future__ import annotations

"""Pair construction utilities for time-lagged learning."""

import numpy as np

from .schema import Shard

__all__ = ["PairBuilder"]


class PairBuilder:
    """Build ``(t, t+tau)`` index pairs inside a single shard."""

    def __init__(self, tau_steps: int) -> None:
        self._tau = 1
        self.set_tau(tau_steps)

    @property
    def tau(self) -> int:
        """Current lag (number of steps) used to build pairs."""

        return int(self._tau)

    def set_tau(self, tau_steps: int) -> None:
        """Update the lag used for pair construction."""

        tau_int = int(tau_steps)
        if tau_int <= 0:
            raise ValueError("tau_steps must be > 0")
        self._tau = tau_int

    def update_tau(self, tau_steps: int) -> None:
        """Alias for :meth:`set_tau` for backwards compatibility."""

        self.set_tau(tau_steps)

    def make_pairs(self, shard: Shard) -> np.ndarray:
        """Return contiguous pairs within a shard with no boundary crossings."""

        n_frames = shard.meta.n_frames
        tau = self.tau
        if n_frames <= tau:
            return np.empty((0, 2), dtype=np.int64)
        idx0 = np.arange(0, n_frames - tau, dtype=np.int64)
        idx1 = idx0 + tau
        pairs = np.stack([idx0, idx1], axis=1)
        return pairs

    def __repr__(self) -> str:  # pragma: no cover - simple debug helper
        return f"PairBuilder(tau={self.tau})"
