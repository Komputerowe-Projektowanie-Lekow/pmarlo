from __future__ import annotations

import sys
import types

if "mlcolvar" not in sys.modules:
    _mlc = types.ModuleType("mlcolvar")
    _cvs = types.ModuleType("mlcolvar.cvs")
    _utils = types.ModuleType("mlcolvar.utils")
    _timelagged = types.ModuleType("mlcolvar.utils.timelagged")
    _mlc.cvs = _cvs
    _mlc.utils = _utils
    _utils.timelagged = _timelagged
    _timelagged.create_timelagged_dataset = lambda *args, **kwargs: None
    _cvs.DeepTICA = object
    sys.modules["mlcolvar"] = _mlc
    sys.modules["mlcolvar.cvs"] = _cvs
    sys.modules["mlcolvar.utils"] = _utils
    sys.modules["mlcolvar.utils.timelagged"] = _timelagged

import math
from typing import Tuple

import numpy as np
import torch

from pmarlo.features.deeptica.losses import VAMP2Loss


def generate_ar1_pairs(
    n_steps: int = 256,
    *,
    lag: int = 1,
    rho: float = 0.85,
    dim: int = 2,
    seed: int = 7,
) -> Tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(size=(n_steps + lag, dim))
    series = np.zeros_like(noise, dtype=np.float64)
    for t in range(1, series.shape[0]):
        series[t] = rho * series[t - 1] + noise[t]
    z0 = series[:-lag]
    zt = series[lag:]
    return (
        torch.as_tensor(z0, dtype=torch.float32),
        torch.as_tensor(zt, dtype=torch.float32),
    )


def test_score_positive_and_scale_invariant():
    loss_fn = VAMP2Loss()
    z0, zt = generate_ar1_pairs()
    _, score = loss_fn(z0, zt)
    assert torch.isfinite(score)
    assert score.item() > 0

    scale0, scale1 = 2.3, 0.7
    _, score_scaled = loss_fn(z0 * scale0, zt * scale1)
    rel_diff = float(abs(score_scaled - score) / score)
    assert rel_diff < 0.05


def test_gradient_nonzero_for_linear_map():
    torch.manual_seed(123)
    loss_fn = VAMP2Loss()
    linear = torch.nn.Linear(2, 3, bias=False)
    z0, zt = generate_ar1_pairs(dim=2)
    z0.requires_grad_(True)
    y0 = linear(z0)
    y1 = linear(zt)
    loss, _ = loss_fn(y0, y1)
    loss.backward()
    grad_norm = float(torch.linalg.vector_norm(linear.weight.grad))
    assert grad_norm > 0, "Expected gradients to flow through the linear map"
