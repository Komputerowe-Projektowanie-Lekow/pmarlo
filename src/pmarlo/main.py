# Copyright (c) 2025 PMARLO Development Team
# SPDX-License-Identifier: GPL-3.0-or-later

"""Public API for single-trajectory PMARLO analysis."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ._version import __version__

if TYPE_CHECKING:
    from .markov_state_model import MarkovStateModel
    from .markov_state_model.free_energy import FESResult, PMFResult
    from .markov_state_model.results import (
        BaseResult,
        CKResult,
        ClusteringResult,
        ITSResult,
        MSMResult,
    )
    from .protein.protein import Protein
    from .utils.errors import TemperatureConsistencyError
    from .utils.seed import set_global_seed

__all__ = [
    "Protein",
    "MarkovStateModel",
    "BaseResult",
    "ClusteringResult",
    "MSMResult",
    "CKResult",
    "ITSResult",
    "FESResult",
    "PMFResult",
    "TemperatureConsistencyError",
    "set_global_seed",
    "get_version",
    "get_info",
    "__version__",
]


def __getattr__(name: str) -> Any:
    """Lazily import public objects from PMARLO submodules."""
    if name == "Protein":
        from .protein.protein import Protein

        return Protein

    if name == "MarkovStateModel":
        from .markov_state_model import MarkovStateModel

        return MarkovStateModel

    if name in {"FESResult", "PMFResult"}:
        from .markov_state_model.free_energy import FESResult, PMFResult

        return {
            "FESResult": FESResult,
            "PMFResult": PMFResult,
        }[name]

    if name in {
        "BaseResult",
        "ClusteringResult",
        "MSMResult",
        "CKResult",
        "ITSResult",
    }:
        from .markov_state_model.results import (
            BaseResult,
            CKResult,
            ClusteringResult,
            ITSResult,
            MSMResult,
        )

        return {
            "BaseResult": BaseResult,
            "ClusteringResult": ClusteringResult,
            "MSMResult": MSMResult,
            "CKResult": CKResult,
            "ITSResult": ITSResult,
        }[name]

    if name == "TemperatureConsistencyError":
        from .utils.errors import TemperatureConsistencyError

        return TemperatureConsistencyError

    if name == "set_global_seed":
        from .utils.seed import set_global_seed

        return set_global_seed

    raise AttributeError(f"module 'pmarlo' has no attribute {name!r}")


def get_version() -> str:
    """Return the PMARLO version string."""
    return __version__


def get_info() -> dict[str, str]:
    """Return basic information about the PMARLO installation."""
    return {
        "version": __version__,
        "package": "pmarlo",
        "description": "Protein Markov State Model analysis for one trajectory",
    }
