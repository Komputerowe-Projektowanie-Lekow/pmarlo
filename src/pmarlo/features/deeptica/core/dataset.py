from __future__ import annotations

from dataclasses import dataclass
from inspect import signature
from typing import Any, Iterable, Optional, Sequence

import numpy as np

try:  # pragma: no cover - optional dependency
    from mlcolvar.data import DictDataset, DictModule  # type: ignore
except Exception as exc:  # pragma: no cover - mlcolvar optional
    DictDataset = None  # type: ignore[assignment]
    DictModule = None  # type: ignore[assignment]
    _MLCOLVAR_IMPORT_ERROR: Exception | None = exc
else:  # pragma: no cover - imported for type checking
    _MLCOLVAR_IMPORT_ERROR = None

__all__ = [
    "DatasetBundle",
    "create_dataset",
    "create_loaders",
    "split_sequences",
    "create_torch_pair_loaders",
]


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

    if DictDataset is None:
        raise ImportError(
            "mlcolvar is required to create Lightning datasets"
        ) from _MLCOLVAR_IMPORT_ERROR

    payload = {
        "data": Z[idx_t].astype(np.float32, copy=False),
        "data_lag": Z[idx_tau].astype(np.float32, copy=False),
        "weights": weights.astype(np.float32, copy=False),
        "weights_lag": weights.astype(np.float32, copy=False),
    }
    return DictDataset(payload)


def create_loaders(dataset: Any, cfg: Any) -> DatasetBundle:
    if DictModule is None:
        raise ImportError(
            "mlcolvar is required to construct Lightning dataloaders"
        ) from _MLCOLVAR_IMPORT_ERROR

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


@dataclass(slots=True)
class TorchLoaderBundle:
    """Container for pure PyTorch dataloaders used by the deeptime backend."""

    train_loader: Any
    val_loader: Any | None
    train_size: int
    val_size: int


class _LaggedPairDataset:
    """Lightweight dataset yielding instantaneous and time-lagged samples."""

    def __init__(self, x_t: np.ndarray, x_tau: np.ndarray) -> None:
        import torch

        self._x_t = torch.as_tensor(np.asarray(x_t, dtype=np.float32))
        self._x_tau = torch.as_tensor(np.asarray(x_tau, dtype=np.float32))

    def __len__(self) -> int:  # pragma: no cover - trivial
        return int(self._x_t.shape[0])

    def __getitem__(self, index: int):  # pragma: no cover - exercised indirectly
        return self._x_t[index], self._x_tau[index]


def _weighted_sampler(weights: Iterable[float], *, generator):
    import torch
    from torch.utils.data import WeightedRandomSampler

    weight_tensor = torch.as_tensor(list(weights), dtype=torch.float64)
    if weight_tensor.numel() == 0:
        raise ValueError("Lagged pair sampler received no weights")
    if torch.any(weight_tensor < 0):
        raise ValueError("Lagged pair weights must be non-negative")
    total = float(weight_tensor.sum().item())
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("Lagged pair weights must sum to a positive finite value")
    return WeightedRandomSampler(
        weights=weight_tensor,
        num_samples=int(weight_tensor.numel()),
        replacement=True,
        generator=generator,
    )


def create_torch_pair_loaders(
    Z: np.ndarray,
    idx_t: np.ndarray,
    idx_tau: np.ndarray,
    weights: np.ndarray,
    cfg: Any,
    *,
    seed: int,
) -> TorchLoaderBundle:
    """Build :mod:`torch` dataloaders for the deeptime VAMPNet backend."""

    import torch
    from torch.utils.data import DataLoader, Subset

    batch_size = int(max(1, getattr(cfg, "batch_size", 64)))
    num_workers = int(max(0, getattr(cfg, "num_workers", 0)))
    val_frac = float(getattr(cfg, "val_frac", 0.1))
    val_frac = float(max(0.0, min(0.5, val_frac)))

    x_t = Z[idx_t].astype(np.float32, copy=False)
    x_tau = Z[idx_tau].astype(np.float32, copy=False)
    dataset = _LaggedPairDataset(x_t, x_tau)

    total = len(dataset)
    if total == 0:
        raise ValueError("Lagged pair dataset is empty; cannot train VAMPNet")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    perm = torch.randperm(total, generator=generator)

    val_size = int(round(total * val_frac))
    val_size = min(max(0, val_size), total - 1) if total > 1 else 0
    train_indices = perm[val_size:]
    val_indices = perm[:val_size]

    if train_indices.numel() == 0:
        raise ValueError("Validation split left no samples for training")

    weights_arr = np.asarray(weights, dtype=np.float64).reshape(-1)
    if weights_arr.size != total:
        raise ValueError("Weight vector does not match lagged pair count")

    train_weights = weights_arr[train_indices.cpu().numpy()]
    sampler = _weighted_sampler(train_weights, generator=generator)

    train_dataset = Subset(dataset, train_indices.tolist())
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        drop_last=False,
    )

    if val_indices.numel() == 0:
        val_loader = None
        val_size = 0
    else:
        val_dataset = Subset(dataset, val_indices.tolist())
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
        )
        val_size = len(val_dataset)

    return TorchLoaderBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        train_size=len(train_dataset),
        val_size=val_size,
    )


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
