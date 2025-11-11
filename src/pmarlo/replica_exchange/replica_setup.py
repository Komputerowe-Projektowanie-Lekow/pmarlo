"""
Thread-safe, immutable minimized state cache for replica exchange initialization.

This module implements the "Compute-once, Broadcast Pattern" optimization:
- First replica triggers energy minimization
- Subsequent replicas reuse cached minimized state via reference sharing
- Trade-off: Adds memory pressure but eliminates redundant compute
- Thread-safe implementation using lock-based caching
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from openmm import unit

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MinimizedState:
    """
    Immutable container for minimized system state.

    This frozen dataclass ensures the cached state cannot be accidentally
    modified, preventing subtle bugs when sharing state across replicas.

    Attributes:
        positions: Atomic positions in nanometers (immutable numpy array)
        potential_energy: System potential energy in kJ/mol
        replica_index: Index of the replica that performed minimization
        minimization_iterations: Total iterations used during minimization
        metadata: Additional diagnostic information (frozen dict)
    """

    positions: np.ndarray  # Shape: (n_atoms, 3), unit: nm
    potential_energy: float  # kJ/mol
    replica_index: int
    minimization_iterations: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Ensure positions array is read-only and validate dimensions."""
        # Convert positions to read-only array for immutability
        if not isinstance(self.positions, np.ndarray):
            raise TypeError("positions must be a numpy array")

        if self.positions.ndim != 2 or self.positions.shape[1] != 3:
            raise ValueError(
                f"positions must have shape (n_atoms, 3), got {self.positions.shape}"
            )

        # Make positions truly immutable
        object.__setattr__(self, "positions", self.positions.copy())
        self.positions.flags.writeable = False

        # Validate energy
        if not np.isfinite(self.potential_energy):
            raise ValueError(
                f"potential_energy must be finite, got {self.potential_energy}"
            )

    def to_openmm_positions(self):
        """Convert positions to OpenMM Quantity with units."""
        return self.positions * unit.nanometer

    def memory_footprint_mb(self) -> float:
        """Estimate memory footprint in megabytes."""
        # positions: n_atoms * 3 * 8 bytes (float64)
        pos_bytes = self.positions.nbytes
        # Small overhead for other fields
        total_bytes = pos_bytes + 1024  # ~1KB overhead
        return total_bytes / (1024 * 1024)


class MinimizedStateCache:
    """
    Thread-safe cache for minimized replica states.

    Implements lazy initialization pattern:
    - First replica to call get_or_compute() performs minimization
    - Subsequent replicas receive cached result immediately
    - Thread-safe using lock to prevent race conditions

    Design considerations:
    - Single cache instance shared across all replica setup operations
    - Lock contention minimal: only during first cache miss
    - Memory pressure: One MinimizedState (~few MB) per cache instance
    """

    def __init__(self):
        """Initialize empty cache with thread-safety lock."""
        self._cache: Optional[MinimizedState] = None
        self._lock = threading.Lock()
        self._cache_hits = 0
        self._cache_misses = 0

    def get_or_compute(
        self, replica_index: int, compute_fn: callable, *compute_args, **compute_kwargs
    ) -> MinimizedState:
        """
        Get cached state or compute if not available.

        Thread-safe implementation ensures only one replica performs
        the expensive minimization operation.

        Args:
            replica_index: Index of requesting replica
            compute_fn: Function to compute MinimizedState if cache miss
            *compute_args: Positional arguments for compute_fn
            **compute_kwargs: Keyword arguments for compute_fn

        Returns:
            MinimizedState: Either cached or freshly computed state
        """
        with self._lock:
            if self._cache is not None:
                self._cache_hits += 1
                logger.debug(
                    f"Replica {replica_index}: Cache HIT (reusing minimized state "
                    f"from replica {self._cache.replica_index})"
                )
                return self._cache

            # Cache miss - compute new state
            self._cache_misses += 1
            logger.info(
                f"Replica {replica_index}: Cache MISS - computing minimized state "
                "(this may take a moment...)"
            )

            state = compute_fn(*compute_args, **compute_kwargs)

            if not isinstance(state, MinimizedState):
                raise TypeError(
                    f"compute_fn must return MinimizedState, got {type(state)}"
                )

            self._cache = state

            logger.info(
                f"Replica {replica_index}: Minimized state cached "
                f"(energy: {state.potential_energy:.2f} kJ/mol, "
                f"iterations: {state.minimization_iterations}, "
                f"memory: {state.memory_footprint_mb():.2f} MB)"
            )

            return state

    def invalidate(self):
        """Clear cached state (useful for testing or restart scenarios)."""
        with self._lock:
            self._cache = None
            logger.debug("MinimizedStateCache invalidated")

    def get_statistics(self) -> dict[str, int]:
        """Return cache performance statistics."""
        with self._lock:
            return {
                "hits": self._cache_hits,
                "misses": self._cache_misses,
                "is_populated": self._cache is not None,
            }

    @property
    def is_populated(self) -> bool:
        """Check if cache contains a state (thread-safe)."""
        with self._lock:
            return self._cache is not None


def create_minimized_state_from_simulation(
    simulation: Any,  # openmm.app.Simulation
    replica_index: int,
    minimization_iterations: int,
    additional_metadata: Optional[dict[str, Any]] = None,
) -> MinimizedState:
    """
    Extract minimized state from OpenMM simulation context.

    Args:
        simulation: OpenMM Simulation object with minimized context
        replica_index: Index of replica that performed minimization
        minimization_iterations: Total iterations used in minimization
        additional_metadata: Optional extra diagnostic information

    Returns:
        MinimizedState: Immutable state object ready for caching
    """
    # Extract state from simulation context
    state = simulation.context.getState(getPositions=True, getEnergy=True)

    # Convert positions to numpy array (immutable)
    positions_quantity = state.getPositions()
    positions_nm = np.array(
        [
            [pos.x, pos.y, pos.z]
            for pos in positions_quantity.value_in_unit(unit.nanometer)
        ],
        dtype=np.float64,
    )

    # Extract energy
    energy = state.getPotentialEnergy()
    energy_kj_mol = float(energy.value_in_unit(unit.kilojoules_per_mole))

    # Build metadata
    metadata = {
        "timestamp": None,  # Could add time.time() if needed
        "platform": simulation.context.getPlatform().getName(),
    }
    if additional_metadata:
        metadata.update(additional_metadata)

    return MinimizedState(
        positions=positions_nm,
        potential_energy=energy_kj_mol,
        replica_index=replica_index,
        minimization_iterations=minimization_iterations,
        metadata=metadata,
    )
