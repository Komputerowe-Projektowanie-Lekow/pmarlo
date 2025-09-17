from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_torch = pytest.importorskip("torch")

_spec = importlib.util.spec_from_file_location(
    "_pmarlo_deeptica_losses",
    Path(__file__).resolve().parents[3]
    / "src"
    / "pmarlo"
    / "features"
    / "deeptica"
    / "losses.py",
)
losses_module = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(losses_module)
VAMP2Loss = losses_module.VAMP2Loss

def test_vamp2_loss_float64_smoke() -> None:
    loss_fn = VAMP2Loss()
    z0 = _torch.randn(64, 3, dtype=_torch.float32, requires_grad=True)
    zt = _torch.randn(64, 3, dtype=_torch.float32)
    loss, score = loss_fn(z0, zt)
    assert loss.dtype == _torch.float64
    assert score.dtype == _torch.float64
    assert _torch.isfinite(loss)
    assert _torch.isfinite(score)
    loss.backward()
    assert z0.grad is not None
    assert _torch.isfinite(z0.grad).all()
