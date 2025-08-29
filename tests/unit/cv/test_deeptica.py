from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

mlc = pytest.importorskip("mlcolvar")
torch = pytest.importorskip("torch")


def test_deeptica_train_transform_export_and_snippet(tmp_path: Path):
    from pmarlo.cv.deeptica import DeepTICAConfig, train_deeptica
    from pmarlo.cv.pairs import scaled_time_pairs

    rng = np.random.default_rng(0)
    # Tiny synthetic dataset: two shards emulated by two arrays
    X1 = rng.normal(size=(64, 3))
    X2 = rng.normal(size=(48, 3))
    X_list = [X1, X2]
    # Uniform-time pairs
    i1, j1 = scaled_time_pairs(len(X1), None, tau_scaled=3.0)
    i2, j2 = scaled_time_pairs(len(X2), None, tau_scaled=3.0)
    idx_t = np.concatenate([i1, len(X1) + i2])
    idx_tlag = np.concatenate([j1, len(X1) + j2])

    cfg = DeepTICAConfig(
        lag=3,
        n_out=2,
        hidden=(16, 16),
        max_epochs=3,
        early_stopping=1,
        batch_size=128,
        seed=1,
    )
    model = train_deeptica(X_list, (idx_t, idx_tlag), cfg, weights=None)

    # Transform full concatenated features
    X_concat = np.concatenate(X_list, axis=0)
    Z = model.transform(X_concat)
    assert Z.shape == (X_concat.shape[0], cfg.n_out)

    # Export TorchScript and PLUMED snippet
    base = tmp_path / "deeptica"
    ts_path = model.to_torchscript(base)
    assert ts_path.exists()
    model.save(base)
    assert base.with_suffix(".json").exists()
    assert base.with_suffix(".pt").exists()
    assert base.with_suffix(".scaler.pt").exists()

    snippet = model.plumed_snippet(base)
    assert "PYTORCH_MODEL" in snippet
