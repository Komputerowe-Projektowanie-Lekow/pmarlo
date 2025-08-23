from __future__ import annotations

from typing import List

import numpy as np
import openmm
from openmm import unit


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
        beta_i = 1.0 / (unit.MOLAR_GAS_CONSTANT_R * temp_i * unit.kelvin)
        beta_j = 1.0 / (unit.MOLAR_GAS_CONSTANT_R * temp_j * unit.kelvin)
        e_i = energies[i]
        e_j = energies[j]
        if not hasattr(e_i, "value_in_unit"):
            e_i = float(e_i) * unit.kilojoules_per_mole
        if not hasattr(e_j, "value_in_unit"):
            e_j = float(e_j) * unit.kilojoules_per_mole
        # Correct Metropolis acceptance for exchanging temperatures of two states
        # Δ = (β_i - β_j) * (U_j - U_i)
        delta_q = (beta_i - beta_j) * (e_j - e_i)
        try:
            delta = delta_q.value_in_unit(unit.dimensionless)
        except Exception:
            delta = float(delta_q)
        return float(min(1.0, np.exp(delta)))

    def accept(self, prob: float) -> bool:
        return bool(self.rng.random() < prob)
