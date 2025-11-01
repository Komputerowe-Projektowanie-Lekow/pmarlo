# Copyright (c) 2025 PMARLO Development Team
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Exchange probability calculation and execution for replica exchange simulations.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import openmm
from openmm import unit

from pmarlo import constants as const
from pmarlo.utils.naming import base_shape_str, permutation_name

from .exchange_engine import ExchangeEngine

logger = logging.getLogger("pmarlo")


class ExchangeOperations:
    """Handles exchange probability calculations and execution."""

    @staticmethod
    def calculate_exchange_probability(
        context_i: openmm.Context,
        context_j: openmm.Context,
        temp_i: float,
        temp_j: float,
        exchange_engine: ExchangeEngine,
    ) -> float:
        """
        Calculate the probability of exchanging two replicas.

        Args:
            context_i: Context of first replica
            context_j: Context of second replica
            temp_i: Temperature of first replica
            temp_j: Temperature of second replica
            exchange_engine: Exchange engine for probability calculations

        Returns:
            Exchange probability
        """
        # Get current energies
        state_i = context_i.getState(getEnergy=True)
        state_j = context_j.getState(getEnergy=True)

        energy_i = state_i.getPotentialEnergy()
        energy_j = state_j.getPotentialEnergy()

        delta = exchange_engine.delta_from_values(temp_i, temp_j, energy_i, energy_j)
        prob = exchange_engine.probability_from_delta(delta)

        # Debug logging for troubleshooting low acceptance rates
        logger.debug(
            (
                f"Exchange calculation: E_i={energy_i}, E_j={energy_j}, "
                f"T_i={temp_i:.1f}K, T_j={temp_j:.1f}K, "
                f"delta={delta:.3f}, prob={prob:.6f}"
            )
        )

        return float(prob)

    @staticmethod
    def calculate_probability_from_cached(
        replica_states: List[int],
        energies: List[Any],
        replica_i: int,
        replica_j: int,
        exchange_engine: ExchangeEngine,
    ) -> float:
        """Calculate exchange probability from cached energies."""
        return exchange_engine.calculate_probability(
            replica_states,
            energies,
            replica_i,
            replica_j,
        )

    @staticmethod
    def perform_exchange(
        replica_i: int,
        replica_j: int,
        replica_states: List[int],
        state_replicas: List[int],
        temperatures: List[float],
        integrators: List[openmm.Integrator],
        contexts: List[openmm.Context],
    ) -> Tuple[List[int], List[int]]:
        """
        Perform the exchange between two replicas.

        Args:
            replica_i: Index of first replica
            replica_j: Index of second replica
            replica_states: Current replica-to-state mapping
            state_replicas: Current state-to-replica mapping
            temperatures: Temperature ladder
            integrators: List of integrators
            contexts: List of contexts

        Returns:
            Updated (replica_states, state_replicas) tuple
        """
        # Bounds checking
        if replica_i >= len(replica_states) or replica_j >= len(replica_states):
            raise RuntimeError(f"replica_states array too small: {len(replica_states)}")

        old_state_i = replica_states[replica_i]
        old_state_j = replica_states[replica_j]

        if old_state_i >= len(state_replicas) or old_state_j >= len(state_replicas):
            raise RuntimeError(f"Invalid state indices: {old_state_i}, {old_state_j}")

        # Swap states
        replica_states[replica_i] = old_state_j
        replica_states[replica_j] = old_state_i

        # Cache a deterministic name for the new permutation of replicas
        shape_name = base_shape_str((len(replica_states),))
        perm_name = permutation_name(tuple(replica_states))
        logger.debug(
            "Replica state permutation %s applied (shape %s)", perm_name, shape_name
        )

        state_replicas[old_state_i] = replica_j
        state_replicas[old_state_j] = replica_i

        # Update integrator temperatures
        integrators[replica_i].setTemperature(temperatures[old_state_j] * unit.kelvin)
        integrators[replica_j].setTemperature(temperatures[old_state_i] * unit.kelvin)

        # Rescale velocities deterministically instead of redrawing
        Ti = temperatures[old_state_i]
        Tj = temperatures[old_state_j]
        scale_ij = float(
            np.sqrt(
                max(
                    const.NUMERIC_MIN_POSITIVE,
                    Tj / max(const.NUMERIC_MIN_POSITIVE, Ti),
                )
            )
        )
        vi = contexts[replica_i].getState(getVelocities=True).getVelocities()
        vj = contexts[replica_j].getState(getVelocities=True).getVelocities()
        contexts[replica_i].setVelocities(vi * scale_ij)
        contexts[replica_j].setVelocities(vj / scale_ij)

        return replica_states, state_replicas

    @staticmethod
    def validate_replica_indices(
        replica_i: int, replica_j: int, n_replicas: int, contexts: List[openmm.Context]
    ) -> None:
        """Validate that replica indices are within bounds."""
        if replica_i < 0 or replica_i >= n_replicas:
            raise ValueError(
                f"replica_i={replica_i} is out of bounds [0, {n_replicas})"
            )
        if replica_j < 0 or replica_j >= n_replicas:
            raise ValueError(
                f"replica_j={replica_j} is out of bounds [0, {n_replicas})"
            )
        if replica_i >= len(contexts):
            raise RuntimeError(
                f"replica_i={replica_i} >= len(contexts)={len(contexts)}"
            )
        if replica_j >= len(contexts):
            raise RuntimeError(
                f"replica_j={replica_j} >= len(contexts)={len(contexts)}"
            )

    @staticmethod
    def attempt_all_exchanges(
        n_replicas: int,
        contexts: List[openmm.Context],
        replica_states: List[int],
        state_replicas: List[int],
        temperatures: List[float],
        integrators: List[openmm.Integrator],
        energies: List[Any],
        exchange_engine: ExchangeEngine,
        pair_attempt_counts: Dict[Tuple[int, int], int],
        pair_accept_counts: Dict[Tuple[int, int], int],
        acceptance_matrix: Optional[np.ndarray],
    ) -> Tuple[int, int, np.ndarray]:
        """
        Attempt exchanges for all neighboring replica pairs.

        Returns:
            Tuple of (total_attempts, total_accepted, updated_acceptance_matrix)
        """
        total_attempts = 0
        total_accepted = 0

        # Even pairs
        for i in range(0, n_replicas - 1, 2):
            try:
                prob = ExchangeOperations.calculate_probability_from_cached(
                    replica_states, energies, i, i + 1, exchange_engine
                )

                state_i_val = replica_states[i]
                state_j_val = replica_states[i + 1]
                pair = (min(state_i_val, state_j_val), max(state_i_val, state_j_val))
                pair_attempt_counts[pair] = pair_attempt_counts.get(pair, 0) + 1
                total_attempts += 1

                if exchange_engine.accept(prob):
                    ExchangeOperations.perform_exchange(
                        i,
                        i + 1,
                        replica_states,
                        state_replicas,
                        temperatures,
                        integrators,
                        contexts,
                    )
                    total_accepted += 1
                    pair_accept_counts[pair] = pair_accept_counts.get(pair, 0) + 1
                    logger.debug(
                        f"Exchange accepted: replica {i} <-> {i + 1} (prob={prob:.3f})"
                    )

                    # Update acceptance matrix
                    if acceptance_matrix is not None:
                        row = i
                        acceptance_matrix[row, 0] += 1  # attempts
                        acceptance_matrix[row, 1] += 1  # accepts
                else:
                    if acceptance_matrix is not None:
                        acceptance_matrix[i, 0] += 1
                    logger.debug(
                        f"Exchange rejected: replica {i} <-> {i + 1} (prob={prob:.3f})"
                    )

            except Exception as exc:
                logger.warning(
                    f"Exchange attempt failed between replicas {i} and {i + 1}: {exc}"
                )

        # Odd pairs
        for i in range(1, n_replicas - 1, 2):
            try:
                prob = ExchangeOperations.calculate_probability_from_cached(
                    replica_states, energies, i, i + 1, exchange_engine
                )

                state_i_val = replica_states[i]
                state_j_val = replica_states[i + 1]
                pair = (min(state_i_val, state_j_val), max(state_i_val, state_j_val))
                pair_attempt_counts[pair] = pair_attempt_counts.get(pair, 0) + 1
                total_attempts += 1

                if exchange_engine.accept(prob):
                    ExchangeOperations.perform_exchange(
                        i,
                        i + 1,
                        replica_states,
                        state_replicas,
                        temperatures,
                        integrators,
                        contexts,
                    )
                    total_accepted += 1
                    pair_accept_counts[pair] = pair_accept_counts.get(pair, 0) + 1
                    logger.debug(
                        f"Exchange accepted: replica {i} <-> {i + 1} (prob={prob:.3f})"
                    )

                    # Update acceptance matrix
                    if acceptance_matrix is not None:
                        row = i
                        acceptance_matrix[row, 0] += 1
                        acceptance_matrix[row, 1] += 1
                else:
                    if acceptance_matrix is not None:
                        acceptance_matrix[i, 0] += 1
                    logger.debug(
                        f"Exchange rejected: replica {i} <-> {i + 1} (prob={prob:.3f})"
                    )

            except Exception as exc:
                logger.warning(
                    f"Exchange attempt failed between replicas {i} and {i + 1}: {exc}"
                )

        return total_attempts, total_accepted, acceptance_matrix
