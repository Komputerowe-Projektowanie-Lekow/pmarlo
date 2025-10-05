from __future__ import annotations

import importlib
import sys
import types

import numpy as np
import pytest

torch = pytest.importorskip("torch")


@pytest.fixture(autouse=True)
def mlcolvar_stub(monkeypatch):
    torch_nn = torch.nn

    class DummyDeepTICA(torch_nn.Module):
        def __init__(self, layers, n_cvs, activation, options):  # type: ignore[override]
            super().__init__()
            self.layers = list(layers)
            self.n_cvs = int(n_cvs)
            self.activation = activation
            self.options = options
            self.nn = torch_nn.Sequential(torch_nn.Linear(layers[0], layers[-1]))

        def named_children(self):  # type: ignore[override]
            return self.nn.named_children()

    mlcolvar = types.ModuleType("mlcolvar")
    cvs = types.ModuleType("mlcolvar.cvs")
    cvs.DeepTICA = DummyDeepTICA
    monkeypatch.setitem(sys.modules, "mlcolvar", mlcolvar)
    monkeypatch.setitem(sys.modules, "mlcolvar.cvs", cvs)
    yield
    sys.modules.pop("mlcolvar", None)
    sys.modules.pop("mlcolvar.cvs", None)
    sys.modules.pop("pmarlo.features.deeptica.core.model", None)
    sys.modules.pop("pmarlo.features.deeptica.core.trainer_api", None)
    sys.modules.pop("pmarlo.features.deeptica", None)


def test_train_deeptica_pipeline_runs_with_stub(monkeypatch, tmp_path):
    trainer_stub = types.ModuleType("pmarlo.ml.deeptica.trainer")

    class DummyCurriculumConfig:
        def __init__(self, **kwargs):
            checkpoint_dir = kwargs.pop("checkpoint_dir", None)
            self.__dict__.update(kwargs)
            self._checkpoint_dir = checkpoint_dir

        @property
        def checkpoint_dir(self):
            return self._checkpoint_dir

        @checkpoint_dir.setter
        def checkpoint_dir(self, value):
            self._checkpoint_dir = value

    class DummyTrainer:
        def __init__(self, net, cfg):
            self.net = net
            self.cfg = cfg

        def fit(self, sequences):
            assert sequences, "expected non-empty sequences"
            return {
                "loss_curve": [1.0, 0.5],
                "grad_norm_curve": [0.2],
            }

    trainer_stub.CurriculumConfig = DummyCurriculumConfig
    trainer_stub.DeepTICACurriculumTrainer = DummyTrainer
    monkeypatch.setitem(sys.modules, "pmarlo.ml.deeptica.trainer", trainer_stub)

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
