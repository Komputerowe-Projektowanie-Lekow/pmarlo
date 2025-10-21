from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset, random_split


class LaggedPairs(Dataset):
    def __init__(self, X_t: torch.Tensor, X_tau: torch.Tensor):
        assert X_t.shape == X_tau.shape
        self.X_t = X_t.float()
        self.X_tau = X_tau.float()

    def __len__(self) -> int:
        return int(self.X_t.shape[0])

    def __getitem__(self, idx: int):
        return self.X_t[idx], self.X_tau[idx]


def make_loaders(
    X_t,
    X_tau,
    batch_size: int = 4096,
    val_frac: float = 0.1,
    num_workers: int = 4,
    seed: int = 2024,
):
    ds = LaggedPairs(X_t, X_tau)
    n_val = max(1, int(len(ds) * float(val_frac)))
    n_train = max(1, len(ds) - n_val)
    g = torch.Generator().manual_seed(int(seed))
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=g)
    common = dict(
        batch_size=int(batch_size),
        num_workers=int(num_workers),
        pin_memory=True,
        persistent_workers=bool(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
        drop_last=False,
    )
    if common["prefetch_factor"] is None:
        common.pop("prefetch_factor")
    return (
        DataLoader(train_ds, shuffle=True, **common),
        DataLoader(val_ds, shuffle=False, **common),
    )
