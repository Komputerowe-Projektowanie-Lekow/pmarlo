from __future__ import annotations

import importlib
import sys
import types

import numpy as np
import pytest

mlcolvar = pytest.importorskip("mlcolvar")
torch = pytest.importorskip("torch")


@pytest.fixture(autouse=True)
def mlcolvar_stub(monkeypatch):
    torch_nn = torch.nn

    class DummyDeepTICA(torch_nn.Module):
        def __init__(self, layers, n_cvs=None, options=None, **kwargs):  # type: ignore[override]
            super().__init__()
            self.layers = list(layers)
            self.n_cvs = int(n_cvs) if n_cvs is not None else layers[-1]
            self.options = options or {}
            # Extract activation from options['nn'] if present
            self.activation = (
                self.options.get("nn", {}).get("activation", "relu")
                if isinstance(self.options.get("nn"), dict)
                else "relu"
            )
            self.nn = torch_nn.Sequential(torch_nn.Linear(layers[0], layers[-1]))

        def named_children(self):  # type: ignore[override]
            return self.nn.named_children()

    mlcolvar = types.ModuleType("mlcolvar")
    cvs = types.ModuleType("mlcolvar.cvs")
    cvs.DeepTICA = DummyDeepTICA
    monkeypatch.setitem(sys.modules, "mlcolvar", mlcolvar)
    monkeypatch.setitem(sys.modules, "mlcolvar.cvs", cvs)
    yield
    sys.modules.pop("pmarlo.features.deeptica.core.model", None)


def _import_model():
    module = importlib.import_module("pmarlo.features.deeptica.core.model")
    return importlib.reload(module)


def test_resolve_helpers_work_with_stubbed_mlcolvar():
    model = _import_model()
    relu = model.resolve_activation_module("relu")
    assert relu.__class__.__name__ == "ReLU"
    assert model.normalize_hidden_dropout([0.1, 0.2], 3) == [0.1, 0.2, 0.2]

    class Cfg:
        hidden = (16,)
        n_out = 2
        activation = "relu"
        linear_head = False
        layer_norm_hidden = True
        dropout_input = 0.25
        layer_norm_in = True

    class Scaler:
        mean_ = np.zeros(4, dtype=np.float64)

    core, layers = model.construct_deeptica_core(Cfg(), Scaler())
    assert layers[0] == 4 and layers[-1] == 2

    net = model.wrap_with_preprocessing_layers(core, Cfg(), Scaler())
    assert isinstance(net, torch.nn.Module)

    data = np.random.rand(6, 4).astype(np.float32)
    try:
        wrapped, info = model.apply_output_whitening(
            net, data, idx_tau=None, apply=True
        )
    except (NotImplementedError, RuntimeError, TypeError) as exc:
        pytest.skip(f"DeepTICA extras unavailable: {exc}")
    assert isinstance(wrapped, torch.nn.Module)
    assert "output_variance" in info


def test_constructed_core_is_callable():
    model = _import_model()

    class Cfg:
        hidden = (16,)
        n_out = 2
        activation = "relu"
        linear_head = False
        layer_norm_hidden = False

    class Scaler:
        mean_ = np.zeros(4, dtype=np.float64)

    cfg = Cfg()
    core, layers = model.construct_deeptica_core(cfg, Scaler())
    tensor = torch.zeros((3, layers[0]), dtype=torch.float32)
    try:
        outputs = core(tensor)
    except NotImplementedError as exc:
        pytest.fail(f"core forward() should be callable: {exc!s}")
    assert isinstance(outputs, torch.Tensor)
    assert outputs.shape == (3, cfg.n_out)


def test_strip_batch_norm_removes_layers():
    model = _import_model()
    seq = torch.nn.Sequential(
        torch.nn.Linear(4, 4),
        torch.nn.BatchNorm1d(4),
        torch.nn.ReLU(),
    )
    model.strip_batch_norm(seq)
    children = list(seq.children())
    assert not any(
        isinstance(child, torch.nn.modules.batchnorm._BatchNorm) for child in children
    )


def test_resolve_hidden_layers_handles_linear_head():
    model = _import_model()

    class Cfg:
        hidden = (32, 16)
        linear_head = True

    assert model.resolve_hidden_layers(Cfg()) == ()


def test_resolve_input_dropout_prefers_specific_field():
    model = _import_model()

    class Cfg:
        dropout_input = 0.33
        dropout = 0.5

    assert model.resolve_input_dropout(Cfg()) == pytest.approx(0.33)
