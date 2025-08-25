from __future__ import annotations

from typing import Optional

import openmm
import openmm.unit as unit


def create_langevin_integrator(
    temperature: float, random_seed: Optional[int] = None
) -> openmm.Integrator:
    """Create a Langevin integrator with common defaults.

    Parameters
    ----------
    temperature:
        Temperature in Kelvin for the integrator.
    random_seed:
        Optional seed for deterministic behaviour.
    """
    integrator = openmm.LangevinIntegrator(
        temperature * unit.kelvin,
        1.0 / unit.picosecond,
        2.0 * unit.femtoseconds,
    )
    if random_seed is not None:
        try:
            integrator.setRandomNumberSeed(int(random_seed))
        except Exception:
            pass
    return integrator
