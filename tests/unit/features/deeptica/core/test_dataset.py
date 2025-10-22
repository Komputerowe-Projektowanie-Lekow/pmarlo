from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mdtraj")

from pmarlo.features.deeptica.core import dataset as dataset_mod
from pmarlo.features.deeptica.core.dataset import (
    TorchLoaderBundle,
    create_dataset,
    create_loaders,
    create_torch_pair_loaders,
)


class _Cfg:
    batch_size = 4
    val_frac = 0.25
    num_workers = 0
    seed = 2024


def test_create_dataset_handles_mlcolvar_availability():
    Z = np.random.rand(12, 3).astype(np.float32)
    idx_t = np.arange(0, 10, dtype=np.int64)
    idx_tau = idx_t + 1
    weights = np.ones_like(idx_t, dtype=np.float32)

    if dataset_mod.DictDataset is None:
        with pytest.raises(ImportError):
            create_dataset(Z, idx_t, idx_tau, weights)
        return

    dataset = create_dataset(Z, idx_t, idx_tau, weights)
    sample = dataset[0]
    assert set(sample.keys()) == {"data", "data_lag", "weights", "weights_lag"}
    assert sample["data"].shape[0] == 3


def test_create_loaders_splits_dataset(tmp_path):
    Z = np.random.rand(20, 3).astype(np.float32)
    idx_t = np.arange(0, 18, dtype=np.int64)
    idx_tau = idx_t + 1
    weights = np.ones_like(idx_t, dtype=np.float32)
    if dataset_mod.DictDataset is None:
        pytest.skip("mlcolvar not available")

    dataset = create_dataset(Z, idx_t, idx_tau, weights)
    bundle = create_loaders(dataset, _Cfg())
    assert bundle.dataset is dataset
    assert bundle.train_loader is not None
    train_batch = next(iter(bundle.train_loader))
    assert "data" in train_batch and "data_lag" in train_batch


def test_create_torch_pair_loaders_returns_weighted_loaders():
    Z = np.random.rand(30, 4).astype(np.float32)
    idx_t = np.arange(0, 25, dtype=np.int64)
    idx_tau = idx_t + 1
    weights = np.linspace(0.1, 1.0, idx_t.size, dtype=np.float64)

    bundle = create_torch_pair_loaders(Z, idx_t, idx_tau, weights, _Cfg(), seed=1234)
    assert isinstance(bundle, TorchLoaderBundle)
    train_batch = next(iter(bundle.train_loader))
    assert isinstance(train_batch, (list, tuple))
    assert len(train_batch) == 2
    assert train_batch[0].shape[1] == Z.shape[1]

    if bundle.val_loader is not None:
        val_batch = next(iter(bundle.val_loader))
        assert len(val_batch) == 2


def test_create_torch_pair_loaders_validates_weight_shape():
    Z = np.random.rand(10, 2).astype(np.float32)
    idx_t = np.arange(0, 8, dtype=np.int64)
    idx_tau = idx_t + 1
    weights = np.ones((idx_t.size - 1,), dtype=np.float32)

    with pytest.raises(ValueError, match="Weight vector"):
        create_torch_pair_loaders(Z, idx_t, idx_tau, weights, _Cfg(), seed=0)
