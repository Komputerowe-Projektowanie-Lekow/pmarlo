"""Replica-exchange conveniences requiring the full dependency stack."""

from __future__ import annotations

from .demux_compat import ExchangeRecord, parse_exchange_log, parse_temperature_ladder
from .replica_exchange import ReplicaExchange
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
    "ExchangeRecord",
    "parse_temperature_ladder",
    "parse_exchange_log",
    "ReplicaExchange",
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
