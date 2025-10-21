from __future__ import annotations

import torch

from .utils import (
    BOX_LENGTHS,
    build_eager_module,
    default_box_tensor,
    reference_positions,
)


def test_pbc_wrap_invariance() -> None:
    module, _ = build_eager_module()
    base = torch.tensor(reference_positions(), dtype=torch.float32)
    box = default_box_tensor()

    translations = []
    zero_shift = torch.zeros_like(base)
    translations.append(zero_shift)

    for axis in range(3):
        shift_vec = torch.zeros(3, dtype=torch.float32)
        shift_vec[axis] = BOX_LENGTHS[axis]
        translations.append(shift_vec.unsqueeze(0).repeat(base.size(0), 1))
        neg_shift = shift_vec.clone()
        neg_shift[axis] = -neg_shift[axis]
        translations.append(neg_shift.unsqueeze(0).repeat(base.size(0), 1))

    combined_shift = BOX_LENGTHS.unsqueeze(0).repeat(base.size(0), 1)
    translations.append(combined_shift)

    energy_tol = 1.0e-5
    force_tol = 5.0e-3
    baseline_energy = None
    baseline_forces = None
    for offset in translations:
        pos = (base + offset).clone().detach().requires_grad_(True)
        energy = module(pos, box)
        energy.backward()
        forces = -pos.grad.detach()

        if baseline_energy is None:
            baseline_energy = energy.detach()
            baseline_forces = forces
        else:
            diff_energy = abs(
                float(energy.detach().item()) - float(baseline_energy.item())
            )
            assert diff_energy < energy_tol
            max_force_diff = torch.max(torch.abs(forces - baseline_forces)).item()
            assert max_force_diff < force_tol
