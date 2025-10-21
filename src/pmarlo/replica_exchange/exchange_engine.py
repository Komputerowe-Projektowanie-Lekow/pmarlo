from __future__ import annotations

from typing import List

import numpy as np
import openmm
from openmm import unit


def _quantity_to_float(
    quantity: openmm.unit.quantity.Quantity,
    expected_unit: openmm.unit.Unit | None,
    *,
    context: str,
) -> float:
    """Coerce an OpenMM quantity into a float while validating its unit."""

    if not hasattr(quantity, "value_in_unit"):
        raise TypeError(f"{context} must provide a value_in_unit method")

    if expected_unit is None:
        expected_unit = unit.dimensionless

    try:
        return float(quantity.value_in_unit(expected_unit))
    except TypeError as exc:  # pragma: no cover - defensive path guarded by tests
        raise TypeError(f"{context} is not convertible to the expected unit") from exc


class ExchangeEngine:
    def __init__(self, temperatures: List[float], rng: np.random.Generator):
        self.temperatures = temperatures
        self.rng = rng

    def calculate_probability(
        self,
        replica_states: List[int],
        energies: List[openmm.unit.quantity.Quantity],
        i: int,
        j: int,
    ) -> float:
        temp_i = self.temperatures[replica_states[i]]
        temp_j = self.temperatures[replica_states[j]]

        gas_constant_unit = getattr(unit, "kilojoule_per_mole", None)
        if gas_constant_unit is not None:
            gas_constant_unit = gas_constant_unit / unit.kelvin

        gas_constant = (
            float(unit.MOLAR_GAS_CONSTANT_R)
            if not hasattr(unit.MOLAR_GAS_CONSTANT_R, "value_in_unit")
            else _quantity_to_float(
                unit.MOLAR_GAS_CONSTANT_R,
                gas_constant_unit,
                context="Gas constant",
            )
        )

        beta_i = 1.0 / (gas_constant * temp_i)
        beta_j = 1.0 / (gas_constant * temp_j)

        energy_unit = getattr(unit, "kilojoule_per_mole", None)
        energy_i = _quantity_to_float(
            energies[i],
            energy_unit,
            context="Energy term",
        )
        energy_j = _quantity_to_float(
            energies[j],
            energy_unit,
            context="Energy term",
        )

        delta = (beta_i - beta_j) * (energy_j - energy_i)
        return float(min(1.0, np.exp(delta)))

    def accept(self, prob: float) -> bool:
        return bool(self.rng.random() < prob)
