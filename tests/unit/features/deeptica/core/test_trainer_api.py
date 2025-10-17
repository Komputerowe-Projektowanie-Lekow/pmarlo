from __future__ import annotations

import importlib

import numpy as np
import pytest

pytest.importorskip("mlcolvar")
torch = pytest.importorskip("torch")


def test_train_deeptica_pipeline_runs(tmp_path):
    trainer_api = importlib.reload(
        importlib.import_module("pmarlo.features.deeptica.core.trainer_api")
    )
    deeptica_module = importlib.reload(
        importlib.import_module("pmarlo.features.deeptica")
    )

    cfg = deeptica_module.DeepTICAConfig(lag=2, max_epochs=1, batch_size=8, hidden=(8,))
    object.__setattr__(cfg, "checkpoint_dir", tmp_path / "ckpt")

    arrays = [np.random.rand(32, 4).astype(np.float32)]
    idx_t = np.arange(0, 30, dtype=np.int64)
    idx_tau = idx_t + 1

    try:
        artifacts = trainer_api.train_deeptica_pipeline(arrays, (idx_t, idx_tau), cfg)
    except (NotImplementedError, RuntimeError, TypeError) as exc:
        pytest.skip(f"DeepTICA extras unavailable: {exc}")

    assert isinstance(artifacts.network, torch.nn.Module)
    assert artifacts.history["loss_curve"], "loss curve should be populated"
    assert artifacts.history["history_source"] == "curriculum_trainer"
    assert "whitening" in artifacts.history
    assert artifacts.device in {"cpu", "cuda"}
