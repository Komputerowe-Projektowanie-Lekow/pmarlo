from __future__ import annotations

from dataclasses import dataclass
from inspect import signature
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
    batch_size = int(getattr(cfg, "batch_size", 64))
    num_workers = int(max(0, getattr(cfg, "num_workers", 0)))
    splits = {"train": float(max(0.0, 1.0 - val_frac)), "val": float(val_frac)}

    dict_module = _instantiate_dict_module(dataset, batch_size, num_workers, splits)

    # Setup the module before accessing dataloaders (required by Lightning)
    if hasattr(dict_module, "setup"):
        dict_module.setup()

    train_loader = dict_module.train_dataloader()
    val_loader = dict_module.val_dataloader()

    return DatasetBundle(
        dataset=dataset,
        train_loader=train_loader,
        val_loader=val_loader,
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


def _instantiate_dict_module(
    dataset: Any, batch_size: int, num_workers: int, splits: dict[str, float]
) -> Any:
    params = signature(DictModule).parameters
    kwargs: dict[str, Any] = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    # Add num_workers only if supported
    if "num_workers" in params:
        kwargs["num_workers"] = num_workers

    if "lengths" in params:
        # Convert splits dict to lengths tuple (train_frac, val_frac)
        train_frac = splits.get("train", 0.9)
        val_frac = splits.get("val", 0.1)
        kwargs["lengths"] = (train_frac, val_frac)
    elif "split" in params:
        kwargs["split"] = splits
    elif "splits" in params:
        kwargs["splits"] = splits
    else:
        raise TypeError(
            "DictModule signature missing 'lengths'/'split'/'splits'; incompatible mlcolvar version"
        )
    return DictModule(**kwargs)
