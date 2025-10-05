from __future__ import annotations

import numpy as np

from pmarlo.features.deeptica.core.dataset import create_dataset, create_loaders


class _Cfg:
    batch_size = 4
    val_frac = 0.25
    num_workers = 0
    seed = 2024


def test_create_dataset_returns_pair_dataset_when_mlcolvar_missing():
    Z = np.random.rand(12, 3).astype(np.float32)
    idx_t = np.arange(0, 10, dtype=np.int64)
    idx_tau = idx_t + 1
    weights = np.ones_like(idx_t, dtype=np.float32)

    dataset = create_dataset(Z, idx_t, idx_tau, weights)
    sample = dataset[0]
    assert set(sample.keys()) == {"data", "data_lag", "weights", "weights_lag"}
    assert sample["data"].shape[0] == 3


def test_create_loaders_splits_dataset(tmp_path):
    Z = np.random.rand(20, 3).astype(np.float32)
    idx_t = np.arange(0, 18, dtype=np.int64)
    idx_tau = idx_t + 1
    weights = np.ones_like(idx_t, dtype=np.float32)
    dataset = create_dataset(Z, idx_t, idx_tau, weights)

    bundle = create_loaders(dataset, _Cfg())
    assert bundle.dataset is dataset
    assert bundle.train_loader is not None
    train_batch = next(iter(bundle.train_loader))
    assert "data" in train_batch and "data_lag" in train_batch
