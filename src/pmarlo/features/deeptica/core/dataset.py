from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np
from mlcolvar.data import DictDataset, DictModule  # type: ignore

__all__ = ["DatasetBundle", "create_dataset", "create_loaders", "split_sequences"]


@dataclass(slots=True)
class DatasetBundle:
    dataset: Any
    train_loader: Optional[Any]
    val_loader: Optional[Any]
    dict_module: Optional[Any]
    lightning_available: bool


def create_dataset(
    Z: np.ndarray,
    idx_t: np.ndarray,
    idx_tau: np.ndarray,
    weights: np.ndarray,
) -> Any:
    """Create a dataset structure backed by :mod:`mlcolvar`."""

    payload = {
        "data": Z[idx_t].astype(np.float32, copy=False),
        "data_lag": Z[idx_tau].astype(np.float32, copy=False),
        "weights": weights.astype(np.float32, copy=False),
        "weights_lag": weights.astype(np.float32, copy=False),
    }
    return DictDataset(payload)


def create_loaders(dataset: Any, cfg: Any) -> DatasetBundle:
    val_frac = max(0.05, float(getattr(cfg, "val_frac", 0.1)))
    dict_module = DictModule(
        dataset,
        batch_size=int(getattr(cfg, "batch_size", 64)),
        shuffle=True,
        split={"train": float(max(0.0, 1.0 - val_frac)), "val": float(val_frac)},
        num_workers=int(max(0, getattr(cfg, "num_workers", 0))),
    )

    return DatasetBundle(
        dataset=dataset,
        train_loader=None,
        val_loader=None,
        dict_module=dict_module,
        lightning_available=True,
    )


def split_sequences(Z: np.ndarray, lengths: Sequence[int]) -> list[np.ndarray]:
    """Slice the normalized feature matrix into per-shard sequences."""

    sequences: list[np.ndarray] = []
    offset = 0
    total = int(Z.shape[0])
    n_features = int(Z.shape[1]) if Z.ndim >= 2 else 0

    for length in lengths:
        n = int(max(0, length))
        if n == 0:
            sequences.append(np.empty((0, n_features), dtype=np.float32))
            continue
        end = min(offset + n, total)
        sequences.append(Z[offset:end])
        offset = end

    if not sequences:
        sequences.append(Z)
    return sequences
