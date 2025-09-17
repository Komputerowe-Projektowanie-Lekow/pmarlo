# Copyright (c) 2025 PMARLO Development Team
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Replica Exchange module for PMARLO.

Provides enhanced sampling through replica exchange molecular dynamics.
"""

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
    "ReplicaExchange",
    "ExchangeRecord",
    "parse_temperature_ladder",
    "parse_exchange_log",
    "Simulation",
    "prepare_system",
    "production_run",
    "feature_extraction",
    "build_transition_model",
    "relative_energies",
    "plot_DG",
]
