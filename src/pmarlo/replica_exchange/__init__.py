"""Replica-exchange conveniences requiring the full dependency stack."""

from __future__ import annotations

from .replica_exchange import ReplicaExchange
from .replica_setup import (
    MinimizedState,
    MinimizedStateCache,
    create_minimized_state_from_simulation,
)
from .simulation import (
    Simulation,
    build_transition_model,
    feature_extraction,
    plot_DG,
    prepare_system,
    production_run,
    relative_energies,
)

__all__ = [
    "ReplicaExchange",
    "MinimizedState",
    "MinimizedStateCache",
    "create_minimized_state_from_simulation",
    "Simulation",
    "prepare_system",
    "production_run",
    "feature_extraction",
    "build_transition_model",
    "relative_energies",
    "plot_DG",
]


def __dir__() -> list[str]:
    return sorted(__all__)
