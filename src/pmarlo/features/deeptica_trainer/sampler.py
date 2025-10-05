"""Sampling utilities for DeepTICA pair curricula."""

from __future__ import annotations

from typing import Iterator

import numpy as np

from pmarlo.features.deeptica.core.pairs import PairInfo
from pmarlo.samplers import BalancedTempSampler

__all__ = ["BalancedTempSampler", "iter_pair_batches"]


def iter_pair_batches(
    pair_info: PairInfo,
    batch_size: int,
    *,
    shuffle: bool = True,
    seed: int | None = None,
) -> Iterator[np.ndarray]:
    """Yield index batches for a ``PairInfo`` structure.

    This preserves the lightweight batch iteration helper that previously lived
    in the feature-level trainer while delegating full sampler logic to
    :mod:`pmarlo.samplers`.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    rng = np.random.default_rng(seed)
    idx = np.arange(pair_info.idx_t.shape[0], dtype=np.int64)
    if shuffle:
        rng.shuffle(idx)
    step = max(1, int(batch_size))
    for start in range(0, idx.size, step):
        yield idx[start : start + step]
