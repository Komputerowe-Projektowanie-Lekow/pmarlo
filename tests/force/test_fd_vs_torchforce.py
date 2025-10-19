from __future__ import annotations

import numpy as np
import torch
from tempfile import TemporaryDirectory
from pathlib import Path

import openmm
from openmm import unit
import pytest

openmmtorch = pytest.importorskip(
    "openmmtorch", reason="openmm-torch required for force tests"
)
from openmmtorch import TorchForce  # type: ignore

from .utils import (
    BOX_LENGTHS,
    default_box_tensor,
    export_bias_module,
    reference_positions,
)


def test_torchforce_matches_finite_difference() -> None:

    base_positions = reference_positions()
    n_atoms = base_positions.shape[0]
    indices = np.arange(min(10, n_atoms))

    with TemporaryDirectory() as tmpdir:
        module, bundle = export_bias_module(Path(tmpdir))
        scripted = torch.jit.load(str(bundle.model_path))

        def energy_fn(positions_nm: np.ndarray) -> float:
            pos_t = torch.tensor(positions_nm, dtype=torch.float32)
            box_t = default_box_tensor()
            with torch.no_grad():
                return float(scripted(pos_t, box_t).item())

        system = openmm.System()
        for _ in range(n_atoms):
            system.addParticle(12.0 * unit.amu)

        torch_force = TorchForce(str(bundle.model_path))
        torch_force.setUsesPeriodicBoundaryConditions(True)
        try:
            torch_force.setProperty("precision", "single")
        except AttributeError:
            pass
        system.addForce(torch_force)

        integrator = openmm.VerletIntegrator(1.0 * unit.femtoseconds)
        platform = openmm.Platform.getPlatformByName("Reference")
        context = openmm.Context(system, integrator, platform)

        context.setPeriodicBoxVectors(
            openmm.Vec3(BOX_LENGTHS[0].item(), 0.0, 0.0),
            openmm.Vec3(0.0, BOX_LENGTHS[1].item(), 0.0),
            openmm.Vec3(0.0, 0.0, BOX_LENGTHS[2].item()),
        )
        context.setPositions([openmm.Vec3(*pos) for pos in base_positions])

        state = context.getState(getForces=True, getEnergy=True)
        torch_forces = np.array(
            state.getForces().value_in_unit(unit.kilojoule_per_mole / unit.nanometer)
        )

        epsilon = 1.0e-6
        fd_forces = np.zeros_like(torch_forces)
        for idx in indices:
            for comp in range(3):
                pos_plus = base_positions.copy()
                pos_minus = base_positions.copy()
                pos_plus[idx, comp] += epsilon
                pos_minus[idx, comp] -= epsilon
                e_plus = energy_fn(pos_plus)
                e_minus = energy_fn(pos_minus)
                fd = -(e_plus - e_minus) / (2.0 * epsilon)
                fd_forces[idx, comp] = fd

        diff = torch_forces[indices] - fd_forces[indices]
        rms_error = np.sqrt(np.mean(diff * diff))
        assert rms_error < 5.0e-3

        del context, integrator
