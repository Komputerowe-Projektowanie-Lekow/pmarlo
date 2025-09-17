# Copyright (c) 2025 PMARLO Development Team
# SPDX-License-Identifier: GPL-3.0-or-later

"""
PMARLO: Protein Markov State Model Analysis with Replica Exchange

A Python package for protein simulation and Markov state model chain generation,
providing an OpenMM-like interface for molecular dynamics simulations.
"""

from typing import TYPE_CHECKING, Optional, Type

from .data.aggregate import aggregate_and_build
from .data.demux_dataset import DemuxDataset, build_demux_dataset
from .data.emit import emit_shards_from_trajectories
from .data.shard import ShardMeta, read_shard, write_shard
from .markov_state_model._msm_utils import candidate_lag_ladder
from .protein.protein import Protein
from .replica_exchange.config import RemdConfig
from .replica_exchange.replica_exchange import ReplicaExchange
from .replica_exchange.simulation import Simulation
from .transform import pm_apply_plan, pm_get_plan
from .transform.build import AppliedOpts, BuildOpts, build_result
from .utils.replica_utils import power_of_two_temperature_ladder
from .utils.seed import quiet_external_loggers

# Free energy surface functionality (stable API surface)
try:
    from .markov_state_model.free_energy import (
        FESResult,
        PMFResult,
        generate_1d_pmf,
        generate_2d_fes,
    )
except Exception:  # pragma: no cover - defensive against optional deps
    FESResult = None
    PMFResult = None
    generate_1d_pmf = None
    generate_2d_fes = None

if TYPE_CHECKING:  # Only for type annotations; avoids importing heavy deps at runtime
    from .transform.pipeline import Pipeline as PipelineType

# Public API names with precise optional type annotations
Pipeline: Optional[Type["PipelineType"]] = None

try:  # Lazy imports: these modules may require heavy dependencies
    from .transform.pipeline import Pipeline as _PipelineRuntime

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
    "BuildOpts",
    "AppliedOpts",
    "build_result",
    "ShardMeta",
    "write_shard",
    "read_shard",
    "emit_shards_from_trajectories",
    "aggregate_and_build",
    "DemuxDataset",
    "build_demux_dataset",
    "power_of_two_temperature_ladder",
    "candidate_lag_ladder",
    "pm_get_plan",
    "pm_apply_plan",
]

# Add free energy exports if available
if FESResult is not None:
    __all__.extend(["FESResult", "PMFResult", "generate_1d_pmf", "generate_2d_fes"])

if MarkovStateModel is not None:
    __all__.insert(3, "MarkovStateModel")

if Pipeline is not None:
    __all__.append("Pipeline")

# Reduce noise from third-party libraries upon import
quiet_external_loggers()
