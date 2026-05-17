"""Conformations finder module using Transition Path Theory (TPT).

This module provides comprehensive tools for identifying and analyzing protein
conformations from Markov State Model (MSM) data using Transition Path Theory.

Key Features:
- Automatic detection of source/sink states from FES and MSM data
- TPT analysis: committors, reactive flux, pathway extraction
- Kinetic Importance Score (KIS) with stability validation
- Uncertainty quantification via bootstrap and hyperparameter ensembles
- Representative structure extraction for multiple conformation types

Usage:
    >>> from pmarlo.api import find_conformations_from_msm
    >>> results = find_conformations_from_msm(
    ...     msm_data={'T': T, 'pi': pi, 'dtrajs': dtrajs, 'features': features},
    ...     trajectories=trajectories,
    ...     auto_detect=True,
    ...     compute_kis=True,
    ...     uncertainty_analysis=True
    ... )
"""

from __future__ import annotations

from .finder import find_conformations
from .kinetic_importance import KineticImportanceScore
from .representative_picker import RepresentativePicker
from .results import ConformationSet, KISResult, TPTResult, UncertaintyResult
from .state_detection import StateDetector
from .tpt_analysis import TPTAnalysis
from .uncertainty import UncertaintyQuantifier
from .visualizations import (
    plot_committors,
    plot_flux_network,
    plot_pathways,
    plot_pcca_states_on_fes,
    plot_tpt_summary,
)

__all__ = [
    "find_conformations",
    "TPTAnalysis",
    "KineticImportanceScore",
    "ConformationSet",
    "StateDetector",
    "RepresentativePicker",
    "UncertaintyQuantifier",
    "TPTResult",
    "KISResult",
    "UncertaintyResult",
    "plot_tpt_summary",
    "plot_committors",
    "plot_flux_network",
    "plot_pathways",
    "plot_pcca_states_on_fes",
]


def __dir__() -> list[str]:
    """List all public exports."""
    return sorted(__all__)
