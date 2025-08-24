# Copyright (c) 2025 PMARLO Development Team
# SPDX-License-Identifier: GPL-3.0-or-later

"""
PMARLO: Protein Markov State Model Analysis with Replica Exchange

A Python package for protein simulation and Markov state model chain generation,
providing an OpenMM-like interface for molecular dynamics simulations.
"""

from .protein.protein import Protein
from .replica_exchange.config import RemdConfig
from .replica_exchange.replica_exchange import ReplicaExchange
from .simulation.simulation import Simulation
from .utils.msm_utils import candidate_lag_ladder
from .utils.replica_utils import power_of_two_temperature_ladder
from .utils.seed import quiet_external_loggers

try:  # Lazy imports: these modules may require heavy dependencies
    from .pipeline import LegacyPipeline, Pipeline
except Exception:  # pragma: no cover
    LegacyPipeline = Pipeline = None  # type: ignore[assignment]

try:  # Markov state model may be unavailable in minimal installs
    from .markov_state_model.markov_state_model import (
        EnhancedMSM as MarkovStateModel,
    )
except Exception:  # pragma: no cover - defensive against optional deps
    MarkovStateModel = None  # type: ignore[assignment]

__version__ = "0.1.0"
__author__ = "PMARLO Development Team"

# Main classes for the clean API
__all__ = [
    "Protein",
    "ReplicaExchange",
    "RemdConfig",
    "Simulation",
    "power_of_two_temperature_ladder",
    "candidate_lag_ladder",
]

if MarkovStateModel is not None:
    __all__.insert(3, "MarkovStateModel")

if Pipeline is not None and LegacyPipeline is not None:
    __all__.extend(["Pipeline", "LegacyPipeline"])

# Reduce noise from third-party libraries upon import
quiet_external_loggers()
