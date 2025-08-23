# Copyright (c) 2025 PMARLO Development Team
# SPDX-License-Identifier: GPL-3.0-or-later

"""
PMARLO: Protein Markov State Model Analysis with Replica Exchange

A Python package for protein simulation and Markov state model chain generation,
providing an OpenMM-like interface for molecular dynamics simulations.
"""

from .markov_state_model.markov_state_model import EnhancedMSM as MarkovStateModel
from .pipeline import LegacyPipeline, Pipeline
from .protein.protein import Protein
from .replica_exchange.config import RemdConfig
from .replica_exchange.replica_exchange import ReplicaExchange
from .simulation.simulation import Simulation
from .utils.msm_utils import candidate_lag_ladder
from .utils.replica_utils import power_of_two_temperature_ladder
from .utils.seed import set_global_seed

__version__ = "0.1.0"
__author__ = "PMARLO Development Team"

# Main classes for the clean API
__all__ = [
    "Protein",
    "ReplicaExchange",
    "RemdConfig",
    "MarkovStateModel",
    "Simulation",
    "Pipeline",
    "LegacyPipeline",
    "power_of_two_temperature_ladder",
    "candidate_lag_ladder",
    "set_global_seed",
]
