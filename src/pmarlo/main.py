# Copyright (c) 2025 PMARLO Development Team
# SPDX-License-Identifier: GPL-3.0-or-later

"""Main entry point for single-trajectory PMARLO analysis."""

from ._version import __version__ as _PACKAGE_VERSION
from .markov_state_model import MarkovStateModel
from .markov_state_model.free_energy import FESResult, PMFResult

# Essential result classes
from .markov_state_model.results import (
    BaseResult,
    CKResult,
    ClusteringResult,
    ITSResult,
    MSMResult,
)
from .protein.protein import Protein

# Error classes
from .utils.errors import (
    TemperatureConsistencyError,
)

# Essential utilities
from .utils.seed import set_global_seed

__all__ = [
    # Main API
    "Protein",
    "MarkovStateModel",
    # Result classes
    "BaseResult",
    "ClusteringResult",
    "MSMResult",
    "CKResult",
    "ITSResult",
    "FESResult",
    "PMFResult",
    # Errors
    "TemperatureConsistencyError",
    # Utils
    "set_global_seed",
]

def get_version() -> str:
    """Get the PMARLO version string."""
    return _PACKAGE_VERSION


def get_info() -> dict:
    """Get information about the PMARLO installation."""
    return {
        "version": get_version(),
        "package": "pmarlo",
        "description": "Protein Markov State Model analysis for one trajectory",
    }

