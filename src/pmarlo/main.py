# Copyright (c) 2025 PMARLO Development Team
# SPDX-License-Identifier: GPL-3.0-or-later

"""
PMARLO: Protein Markov State Model Analysis with Replica Exchange

Main entry point for the PMARLO package. This module provides the core API
imports and essential functionality for protein simulation and MSM analysis.
"""

# Core API imports - the main interface for users
from .transform.pipeline import Pipeline, run_pmarlo
from .protein.protein import Protein
from .markov_state_model import MarkovStateModel
from .replica_exchange import ReplicaExchange, Simulation
from .transform.plan import TransformPlan, TransformStep
from .transform.runner import apply_plan

# Essential result classes
from .markov_state_model.results import (
    BaseResult,
    REMDResult,
    DemuxResult,
    ClusteringResult,
    MSMResult,
    CKResult,
    ITSResult,
)
from .markov_state_model.free_energy import FESResult, PMFResult

# Progress reporting
from .transform.progress import (
    ProgressReporter,
    ProgressPrinter,
    console_progress_cb,
)

# Error classes
from .utils.errors import (
    TemperatureConsistencyError,
    DemuxError,
    DemuxIntegrityError,
    DemuxIOError,
    DemuxPlanError,
    DemuxWriterError,
)

# Essential utilities
from .utils.seed import set_global_seed


__all__ = [
    # Main API
    "Pipeline",
    "run_pmarlo",
    "Protein",
    "MarkovStateModel",
    "ReplicaExchange",
    "Simulation",
    # Transform system
    "TransformPlan",
    "TransformStep",
    "apply_plan",
    # Result classes
    "BaseResult",
    "REMDResult",
    "DemuxResult",
    "ClusteringResult",
    "MSMResult",
    "CKResult",
    "ITSResult",
    "FESResult",
    "PMFResult",
    # Progress
    "ProgressReporter",
    "ProgressPrinter",
    "console_progress_cb",
    # Errors
    "TemperatureConsistencyError",
    "DemuxError",
    "DemuxIntegrityError",
    "DemuxIOError",
    "DemuxPlanError",
    "DemuxWriterError",
    # Utils
    "set_global_seed",
]


def get_version() -> str:
    """Get the PMARLO version string."""
    try:
        from ._version import __version__
        return __version__
    except ImportError:
        return "unknown"


def get_info() -> dict:
    """Get information about the PMARLO installation."""
    return {
        "version": get_version(),
        "package": "pmarlo",
        "description": "Protein Markov State Model Analysis with Replica Exchange",
    }


# This module serves as the main API entry point.
# For examples and demos, see the example_programs/ directory.
