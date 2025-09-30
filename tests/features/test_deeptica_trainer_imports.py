from __future__ import annotations

import sys
import types

# Provide lightweight stubs for optional mlcolvar dependency
if "mlcolvar" not in sys.modules:
    mlcolvar = types.ModuleType("mlcolvar")
    sys.modules["mlcolvar"] = mlcolvar
    cvs = types.ModuleType("mlcolvar.cvs")
    cvs.DeepTICA = object  # type: ignore[attr-defined]
    sys.modules["mlcolvar.cvs"] = cvs
    utils = types.ModuleType("mlcolvar.utils.timelagged")
    utils.create_timelagged_dataset = lambda *a, **k: None
    sys.modules["mlcolvar.utils.timelagged"] = utils


from pmarlo.features.deeptica_trainer import DeepTICATrainer, TrainerConfig


class DummyModel:
    pass


def test_trainer_import_and_instantiation():
    trainer = DeepTICATrainer(DummyModel(), TrainerConfig(tau_steps=3))
    assert trainer.cfg.tau_steps == 3
    assert trainer.model.__class__ is DummyModel
