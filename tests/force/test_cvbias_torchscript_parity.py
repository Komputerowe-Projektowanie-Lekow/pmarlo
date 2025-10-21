from __future__ import annotations

import torch

from .utils import (
    build_eager_module,
    default_box_tensor,
    reference_positions,
)


def test_cvbias_torchscript_energy_parity() -> None:
    module, _ = build_eager_module()
    scripted = torch.jit.script(module)

    base = torch.tensor(reference_positions(), dtype=torch.float32)
    box = default_box_tensor()

    torch.manual_seed(123)
    for _ in range(5):
        perturb = torch.randn_like(base) * 0.05
        positions = base + perturb
        eager_energy = module(positions, box)
        scripted_energy = scripted(positions, box)
        assert torch.allclose(eager_energy, scripted_energy, atol=1.0e-6)
