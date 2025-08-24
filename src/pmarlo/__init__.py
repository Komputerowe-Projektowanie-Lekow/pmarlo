# Copyright (c) 2025 PMARLO Development Team
# SPDX-License-Identifier: GPL-3.0-or-later

"""
PMARLO: Protein Markov State Model Analysis with Replica Exchange

A Python package for protein simulation and Markov state model chain generation,
providing an OpenMM-like interface for molecular dynamics simulations.
"""

from typing import TYPE_CHECKING, Optional, Type

from .protein.protein import Protein
from .replica_exchange.config import RemdConfig
from .replica_exchange.replica_exchange import ReplicaExchange
from .simulation.simulation import Simulation
from .utils.msm_utils import candidate_lag_ladder
from .utils.replica_utils import power_of_two_temperature_ladder
from .utils.seed import quiet_external_loggers

if TYPE_CHECKING:  # Only for type annotations; avoids importing heavy deps at runtime
    from .pipeline import LegacyPipeline as LegacyPipelineType
    from .pipeline import Pipeline as PipelineType

# Public API names with precise optional type annotations
LegacyPipeline: Optional[Type["LegacyPipelineType"]] = None
Pipeline: Optional[Type["PipelineType"]] = None

try:  # Lazy imports: these modules may require heavy dependencies
    from .pipeline import LegacyPipeline as _LegacyPipelineRuntime
    from .pipeline import Pipeline as _PipelineRuntime

    LegacyPipeline = _LegacyPipelineRuntime
    Pipeline = _PipelineRuntime
except Exception:  # pragma: no cover
    pass

if TYPE_CHECKING:
    from .markov_state_model.enhanced_msm import EnhancedMSM as MarkovStateModelType

MarkovStateModel: Optional[Type["MarkovStateModelType"]] = None

try:  # Markov state model may be unavailable in minimal installs
    from .markov_state_model.enhanced_msm import EnhancedMSM as _EnhancedMSMRuntime

    MarkovStateModel = _EnhancedMSMRuntime
except Exception:  # pragma: no cover - defensive against optional deps
    pass

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
