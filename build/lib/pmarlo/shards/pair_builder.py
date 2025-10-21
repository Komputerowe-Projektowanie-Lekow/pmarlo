from __future__ import annotations

"""Pair construction utilities for time-lagged learning."""

import numpy as np

from pmarlo.pairs.core import build_pair_info

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

        info = build_pair_info([np.asarray(shard.X)], (self.tau,))
        if info.idx_t.size == 0 or info.idx_tau.size == 0:
            return np.empty((0, 2), dtype=np.int64)
        pairs = np.stack((info.idx_t, info.idx_tau), axis=1)
        return pairs.astype(np.int64, copy=False)

    def __repr__(self) -> str:  # pragma: no cover - simple debug helper
        return f"PairBuilder(tau={self.tau})"
