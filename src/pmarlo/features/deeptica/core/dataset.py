from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import TYPE_CHECKING, Any, Optional, Sequence

import numpy as np

try:  # pragma: no cover - optional ML stack
    import torch  # type: ignore
except Exception:  # pragma: no cover - torch optional dependency
    torch = None  # type: ignore[assignment]

if TYPE_CHECKING:  # pragma: no cover - type checking only
    import torch as _torch_mod  # noqa: F401

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
        if torch is not None:
            train_loader, val_loader = _torch_loaders(dataset, cfg)
        else:
            train_loader, val_loader = _fallback_loaders(dataset, cfg)

    return DatasetBundle(
        dataset=dataset,
        train_loader=train_loader,
        val_loader=val_loader,
        dict_module=dict_module,
        lightning_available=lightning_available,
    )


def _torch_loaders(dataset: Any, cfg: Any) -> tuple[Optional[Any], Optional[Any]]:
    if torch is None:
        return _fallback_loaders(dataset, cfg)

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


def _fallback_loaders(dataset: Any, cfg: Any) -> tuple[Optional[Any], Optional[Any]]:
    try:
        length = int(len(dataset))  # type: ignore[arg-type]
    except Exception:
        return None, None

    if length <= 0:
        return None, None

    seed = int(getattr(cfg, "seed", 2024))
    batch_size = max(1, int(getattr(cfg, "batch_size", 64)))
    indices = np.arange(length, dtype=int)
    rng = np.random.default_rng(seed)

    if length >= 2:
        rng.shuffle(indices)
        val_frac = max(0.05, float(getattr(cfg, "val_frac", 0.1)))
        n_val = max(1, int(round(val_frac * length)))
        if n_val >= length:
            n_val = length - 1
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]
    else:
        train_indices = indices
        val_indices = np.array([], dtype=int)

    if train_indices.size == 0:
        train_indices = indices
        val_indices = np.array([], dtype=int)

    train_loader = _SimpleDataLoader(
        dataset,
        train_indices.tolist(),
        batch_size,
        shuffle=True,
        seed=seed,
    )
    val_loader = (
        _SimpleDataLoader(
            dataset,
            val_indices.tolist(),
            batch_size,
            shuffle=False,
            seed=seed + 1,
        )
        if val_indices.size
        else None
    )
    return train_loader, val_loader


class _SimpleDataLoader:
    def __init__(
        self,
        dataset: Any,
        indices: list[int],
        batch_size: int,
        *,
        shuffle: bool,
        seed: int,
    ) -> None:
        self._dataset = dataset
        self._indices = [int(i) for i in indices]
        self._batch_size = max(1, int(batch_size))
        self._shuffle = bool(shuffle)
        self._rng = np.random.default_rng(seed)

    def __iter__(self):
        if not self._indices:
            return
        indices = np.array(self._indices, dtype=int)
        if self._shuffle and indices.size > 1:
            self._rng.shuffle(indices)
        for start in range(0, indices.size, self._batch_size):
            batch_idx = indices[start : start + self._batch_size]
            samples = [self._dataset[int(i)] for i in batch_idx]
            yield _collate_samples(samples)

    def __len__(self) -> int:
        if not self._indices:
            return 0
        return ceil(len(self._indices) / self._batch_size)


def _collate_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        return {}
    keys = samples[0].keys()
    collated: dict[str, Any] = {}
    for key in keys:
        values = [sample[key] for sample in samples]
        first = values[0]
        if isinstance(first, np.ndarray):
            collated[key] = np.stack(
                [np.asarray(v, dtype=first.dtype) for v in values], axis=0
            )
        else:
            dtype = np.asarray(first).dtype
            collated[key] = np.asarray(values, dtype=dtype)
    return collated


if torch is not None:
    _DatasetBase = torch.utils.data.Dataset  # type: ignore[attr-defined]
else:

    class _DatasetBase:
        """Minimal base class used when torch is unavailable."""

        pass


class _PairDataset(_DatasetBase):
    def __init__(self, x_t: np.ndarray, x_tau: np.ndarray, weights: np.ndarray) -> None:
        self._torch_mode = torch is not None
        self.weights = np.asarray(weights, dtype=np.float32).reshape(-1)
        if self._torch_mode and torch is not None:
            self.x_t = torch.as_tensor(x_t, dtype=torch.float32)
            self.x_tau = torch.as_tensor(x_tau, dtype=torch.float32)
        else:
            self.x_t = np.asarray(x_t, dtype=np.float32)
            self.x_tau = np.asarray(x_tau, dtype=np.float32)

    def __len__(self) -> int:
        return int(self.x_t.shape[0])

    def __getitem__(self, idx: int) -> dict[str, Any]:
        weight = float(self.weights[idx]) if self.weights.size else 1.0
        if self._torch_mode and torch is not None:
            return {
                "data": self.x_t[idx],
                "data_lag": self.x_tau[idx],
                "weights": np.float32(weight),
                "weights_lag": np.float32(weight),
            }
        data = np.asarray(self.x_t[idx], dtype=np.float32)
        data_lag = np.asarray(self.x_tau[idx], dtype=np.float32)
        return {
            "data": data,
            "data_lag": data_lag,
            "weights": np.float32(weight),
            "weights_lag": np.float32(weight),
        }


def split_sequences(Z: np.ndarray, lengths: Sequence[int]) -> list[np.ndarray]:
    """Slice the normalized feature matrix into per-shard sequences."""

    sequences: list[np.ndarray] = []
    offset = 0
    total = int(Z.shape[0])
    n_features = int(Z.shape[1]) if Z.ndim >= 2 else 0

    for length in lengths:
        n = int(max(0, length))
        if n == 0:
            sequences.append(np.zeros((0, n_features), dtype=np.float32))
            continue
        end = min(offset + n, total)
        sequences.append(Z[offset:end])
        offset = end

    if not sequences:
        sequences.append(Z)
    return sequences
