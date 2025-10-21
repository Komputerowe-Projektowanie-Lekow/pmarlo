from __future__ import annotations

from typing import List

import numpy as np
import openmm
from openmm import unit


def _quantity_to_float(
    quantity: openmm.unit.quantity.Quantity | float | int,
    expected_unit: openmm.unit.Unit | None,
    *,
    context: str,
) -> float:
    """Coerce an OpenMM quantity into a float while validating its unit."""

    if not hasattr(quantity, "value_in_unit"):
        try:
            return float(quantity)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{context} is not convertible to a float") from exc

    try:
        return float(quantity.value_in_unit(expected_unit))
    except TypeError as exc:
        raise TypeError(f"{context} is not convertible to the expected unit") from exc


class ExchangeEngine:
    def __init__(self, temperatures: List[float], rng: np.random.Generator):
        self.temperatures = temperatures
        self.rng = rng

    def _compute_delta(
        self,
        temp_i: float,
        temp_j: float,
        energy_i: openmm.unit.quantity.Quantity | float | int,
        energy_j: openmm.unit.quantity.Quantity | float | int,
    ) -> float:
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

        energy_unit = getattr(unit, "kilojoule_per_mole", None)
        energy_i_value = _quantity_to_float(
            energy_i, energy_unit, context="Energy term"
        )
        energy_j_value = _quantity_to_float(
            energy_j, energy_unit, context="Energy term"
        )

        beta_i = 1.0 / (gas_constant * temp_i)
        beta_j = 1.0 / (gas_constant * temp_j)

        delta = (beta_i - beta_j) * (energy_j_value - energy_i_value)
        return float(delta)

    @staticmethod
    def probability_from_delta(delta: float) -> float:
        return float(min(1.0, np.exp(float(delta))))

    def delta_from_values(
        self,
        temp_i: float,
        temp_j: float,
        energy_i: openmm.unit.quantity.Quantity | float | int,
        energy_j: openmm.unit.quantity.Quantity | float | int,
    ) -> float:
        return self._compute_delta(temp_i, temp_j, energy_i, energy_j)

    def probability_from_values(
        self,
        temp_i: float,
        temp_j: float,
        energy_i: openmm.unit.quantity.Quantity | float | int,
        energy_j: openmm.unit.quantity.Quantity | float | int,
    ) -> float:
        delta = self._compute_delta(temp_i, temp_j, energy_i, energy_j)
        return self.probability_from_delta(delta)

    def calculate_probability(
        self,
        replica_states: List[int],
        energies: List[openmm.unit.quantity.Quantity | float | int],
        i: int,
        j: int,
    ) -> float:
        temp_i = self.temperatures[replica_states[i]]
        temp_j = self.temperatures[replica_states[j]]

        delta = self._compute_delta(temp_i, temp_j, energies[i], energies[j])
        return self.probability_from_delta(delta)

    def accept(self, prob: float) -> bool:
        return bool(self.rng.random() < prob)
