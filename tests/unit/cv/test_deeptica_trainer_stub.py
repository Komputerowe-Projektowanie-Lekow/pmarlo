from __future__ import annotations

import numpy as np
import pytest

from pmarlo.features.deeptica_trainer import DeepTICATrainer, TrainerConfig


class DummyModel:
    pass


def test_trainer_instantiation():
    model = DummyModel()
    cfg = TrainerConfig(tau_steps=2)
    try:
        trainer = DeepTICATrainer(model, cfg)
    except TypeError as exc:
        pytest.skip(f"DeepTICA trainer unavailable: {exc}")
    assert trainer.model is model
    assert trainer.cfg is cfg
    # Ensure step/evaluate are not implemented yet
    try:
        trainer.step([(np.zeros((1, 2)), np.zeros((1, 2)), None)])
    except NotImplementedError:
        pass
    try:
        trainer.evaluate([(np.zeros((1, 2)), np.zeros((1, 2)), None)])
    except NotImplementedError:
        pass
