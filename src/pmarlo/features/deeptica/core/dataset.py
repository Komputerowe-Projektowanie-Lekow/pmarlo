from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch  # type: ignore


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
    """Create a dataset structure compatible with mlcolvar or raw torch."""

    try:
        from mlcolvar.data import DictDataset as _DictDataset  # type: ignore

        payload = {
            "data": Z[idx_t].astype(np.float32, copy=False),
            "data_lag": Z[idx_tau].astype(np.float32, copy=False),
            "weights": weights.astype(np.float32, copy=False),
            "weights_lag": weights.astype(np.float32, copy=False),
        }
        return _DictDataset(payload)
    except Exception:
        return _PairDataset(Z[idx_t], Z[idx_tau], weights)


def create_loaders(dataset: Any, cfg: Any) -> DatasetBundle:
    lightning_available = False
    dict_module = None
    train_loader = None
    val_loader = None

    try:
        from mlcolvar.data import DictModule as _DictModule  # type: ignore

        val_frac = max(0.05, float(getattr(cfg, "val_frac", 0.1)))
        dict_module = _DictModule(
            dataset,
            batch_size=int(getattr(cfg, "batch_size", 64)),
            shuffle=True,
            split={"train": float(max(0.0, 1.0 - val_frac)), "val": float(val_frac)},
            num_workers=int(max(0, getattr(cfg, "num_workers", 0))),
        )
        lightning_available = True
    except Exception:
        train_loader, val_loader = _torch_loaders(dataset, cfg)

    return DatasetBundle(
        dataset=dataset,
        train_loader=train_loader,
        val_loader=val_loader,
        dict_module=dict_module,
        lightning_available=lightning_available,
    )


def _torch_loaders(dataset: Any, cfg: Any) -> tuple[Optional[Any], Optional[Any]]:
    generator = torch.Generator().manual_seed(int(getattr(cfg, "seed", 2024)))
    try:
        length = int(len(dataset))  # type: ignore[arg-type]
    except Exception:
        length = 0

    num_workers = int(max(0, getattr(cfg, "num_workers", 0)))
    batch_size = int(getattr(cfg, "batch_size", 64))
    persistent = bool(num_workers > 0)

    if length >= 2:
        val_frac = max(0.05, float(getattr(cfg, "val_frac", 0.1)))
        n_val = max(1, int(val_frac * length))
        n_train = max(1, length - n_val)
        train_ds, val_ds = torch.utils.data.random_split(  # type: ignore[assignment]
            dataset,
            [n_train, n_val],
            generator=generator,
        )
        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=num_workers,
            persistent_workers=persistent,
            prefetch_factor=2 if persistent else None,
        )
        val_loader = torch.utils.data.DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
            persistent_workers=persistent,
            prefetch_factor=2 if persistent else None,
        )
        return train_loader, val_loader

    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
        persistent_workers=persistent,
        prefetch_factor=2 if persistent else None,
    )
    return train_loader, None


class _PairDataset(torch.utils.data.Dataset):  # type: ignore[type-arg]
    def __init__(self, x_t: np.ndarray, x_tau: np.ndarray, weights: np.ndarray) -> None:
        self.x_t = torch.as_tensor(x_t, dtype=torch.float32)
        self.x_tau = torch.as_tensor(x_tau, dtype=torch.float32)
        self.weights = np.asarray(weights, dtype=np.float32).reshape(-1)

    def __len__(self) -> int:
        return int(self.x_t.shape[0])

    def __getitem__(self, idx: int) -> dict[str, Any]:
        weight = float(self.weights[idx]) if self.weights.size else 1.0
        return {
            "data": self.x_t[idx],
            "data_lag": self.x_tau[idx],
            "weights": np.float32(weight),
            "weights_lag": np.float32(weight),
        }
