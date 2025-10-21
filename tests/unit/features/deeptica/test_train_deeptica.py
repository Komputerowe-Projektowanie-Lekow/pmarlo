from __future__ import annotations

import numpy as np
import pytest
import torch.nn as nn

from pmarlo.features.deeptica import _full as deeptica_full
from pmarlo.features.deeptica.core.trainer_api import TrainingArtifacts


class _ScalerStub:
    def __init__(self, n_features: int = 1) -> None:
        self.mean_ = np.zeros(n_features, dtype=np.float64)
        self.scale_ = np.ones(n_features, dtype=np.float64)


def _make_artifacts() -> TrainingArtifacts:
    scaler = _ScalerStub()
    network = nn.Linear(1, 1)
    history: dict[str, object] = {}
    return TrainingArtifacts(
        scaler=scaler, network=network, history=history, device="cpu"
    )


def _dummy_pairs() -> tuple[np.ndarray, np.ndarray]:
    idx_t = np.array([0, 1], dtype=np.int64)
    idx_tau = np.array([1, 2], dtype=np.int64)
    return idx_t, idx_tau


def _dummy_data() -> list[np.ndarray]:
    return [np.zeros((3, 1), dtype=np.float32)]


def test_train_deeptica_uses_lightning_backend_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called: dict[str, bool] = {}

    def fake_pipeline(
        X_list, pairs, cfg, weights=None
    ):  # pragma: no cover - simple stub
        called["lightning"] = True
        return _make_artifacts()

    monkeypatch.setitem(deeptica_full._TRAINING_BACKENDS, "lightning", fake_pipeline)
    monkeypatch.setitem(deeptica_full._TRAINING_BACKENDS, "curriculum", fake_pipeline)

    cfg = deeptica_full.DeepTICAConfig(lag=2)
    model = deeptica_full.train_deeptica(_dummy_data(), _dummy_pairs(), cfg)

    assert called.get("lightning") is True
    assert isinstance(model, deeptica_full.DeepTICAModel)


def test_train_deeptica_routes_to_mlcolvar_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called: dict[str, bool] = {}

    def fake_mlcolvar(
        X_list, pairs, cfg, weights=None
    ):  # pragma: no cover - simple stub
        called["mlcolvar"] = True
        return _make_artifacts()

    monkeypatch.setitem(deeptica_full._TRAINING_BACKENDS, "mlcolvar", fake_mlcolvar)

    cfg = deeptica_full.DeepTICAConfig(lag=2, trainer_backend="mlcolvar")
    model = deeptica_full.train_deeptica(_dummy_data(), _dummy_pairs(), cfg)

    assert called.get("mlcolvar") is True
    assert isinstance(model, deeptica_full.DeepTICAModel)


def test_deeptica_config_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError, match="trainer_backend"):
        deeptica_full.DeepTICAConfig(lag=2, trainer_backend="legacy")
