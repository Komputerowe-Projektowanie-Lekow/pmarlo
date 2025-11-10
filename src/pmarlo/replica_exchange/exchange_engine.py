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
    """Coerce an OpenMM quantity into a float while validating its unit.

    Optimized to handle the common case of already-converted floats first,
    avoiding expensive hasattr checks and unit conversions.
    """
    # Fast path: already a float or int (most common after preconversion)
    if isinstance(quantity, (float, int)):
        return float(quantity)

    # Slow path: OpenMM Quantity object that needs conversion
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
        energies: List[float],  # Now expects pre-converted floats
        i: int,
        j: int,
    ) -> float:
        temp_i = self.temperatures[replica_states[i]]
        temp_j = self.temperatures[replica_states[j]]

        delta = self._compute_delta(temp_i, temp_j, energies[i], energies[j])
        return self.probability_from_delta(delta)

    def accept(self, prob: float) -> bool:
        return bool(self.rng.random() < prob)

    def attempt_exchanges_vectorized(
        self,
        replica_states: List[int],
        energies: List[float],
        i_indices: List[int],
        j_indices: List[int],
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Vectorized exchange attempt for multiple pairs.

        Args:
            replica_states: Current state assignment for each replica
            energies: Pre-converted energy values (floats in kJ/mol)
            i_indices: First replica indices for each pair
            j_indices: Second replica indices for each pair

        Returns:
            Tuple of (deltas, accepted) where:
                - deltas: np.ndarray of delta values for each pair
                - accepted: np.ndarray of booleans indicating acceptance
        """
        # Convert to numpy arrays for vectorized operations
        i_idx = np.array(i_indices, dtype=np.int32)
        j_idx = np.array(j_indices, dtype=np.int32)

        # Get states and energies for all pairs
        states_i = np.array([replica_states[i] for i in i_idx], dtype=np.int32)
        states_j = np.array([replica_states[j] for j in j_idx], dtype=np.int32)

        # Get temperatures for all pairs
        temps_i = np.array([self.temperatures[s] for s in states_i], dtype=np.float64)
        temps_j = np.array([self.temperatures[s] for s in states_j], dtype=np.float64)

        # Get energies for all pairs
        energies_i = np.array([energies[i] for i in i_idx], dtype=np.float64)
        energies_j = np.array([energies[j] for j in j_idx], dtype=np.float64)

        # Compute gas constant (do once, not per pair)
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

        # Vectorized computation of beta and delta
        beta_i = 1.0 / (gas_constant * temps_i)
        beta_j = 1.0 / (gas_constant * temps_j)

        # Vectorize: dBeta = beta[j] - beta[i], dU = U[j] - U[i]
        dBeta = beta_j - beta_i
        dU = energies_j - energies_i

        # Vectorize: delta = -dBeta * dU (note: sign convention)
        deltas = -dBeta * dU

        # Vectorized acceptance: log(rand) < -(-delta) = delta
        # Equivalent to: rand < exp(delta)
        random_values = self.rng.random(len(deltas))
        accepted = np.log(random_values) < deltas

        return deltas, accepted
