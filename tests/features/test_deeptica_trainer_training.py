from __future__ import annotations

import sys
import types

if "mlcolvar" not in sys.modules:
    mlcolvar = types.ModuleType("mlcolvar")
    sys.modules["mlcolvar"] = mlcolvar
    cvs = types.ModuleType("mlcolvar.cvs")
    cvs.DeepTICA = object  # type: ignore[attr-defined]
    sys.modules["mlcolvar.cvs"] = cvs
    utils = types.ModuleType("mlcolvar.utils.timelagged")
    utils.create_timelagged_dataset = lambda *a, **k: None
    sys.modules["mlcolvar.utils.timelagged"] = utils

import numpy as np
import torch

from pmarlo.features.deeptica_trainer import DeepTICATrainer, TrainerConfig


class _DummyModel:
    def __init__(self, in_dim: int = 2, out_dim: int = 2) -> None:
        self.net = torch.nn.Linear(in_dim, out_dim)
        self.cfg = type("Cfg", (), {"n_out": out_dim})()
        self.training_history: dict = {}


def test_trainer_step_updates_history():
    model = _DummyModel()
    trainer = DeepTICATrainer(model, TrainerConfig(tau_steps=1, log_every=1))

    x_t = np.random.randn(32, 2).astype(np.float32)
    x_tau = np.random.randn(32, 2).astype(np.float32)

    metrics = trainer.step([(x_t, x_tau, None)])
    assert "loss" in metrics
    assert trainer.history
    assert model.training_history["steps"], "training history should record steps"
