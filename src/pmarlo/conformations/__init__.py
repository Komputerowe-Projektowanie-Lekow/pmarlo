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
    >>> from pmarlo.conformations import find_conformations
    >>> results = find_conformations(
    ...     msm_data={'T': T, 'pi': pi, 'dtrajs': dtrajs, 'features': features},
    ...     trajectories=trajectories,
    ...     auto_detect=True,
    ...     compute_kis=True,
    ...     uncertainty_analysis=True
    ... )
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

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
]

_EXPORTS: dict[str, tuple[str, str]] = {
    "find_conformations": (
        "pmarlo.conformations.finder",
        "find_conformations",
    ),
    "TPTAnalysis": (
        "pmarlo.conformations.tpt_analysis",
        "TPTAnalysis",
    ),
    "KineticImportanceScore": (
        "pmarlo.conformations.kinetic_importance",
        "KineticImportanceScore",
    ),
    "ConformationSet": (
        "pmarlo.conformations.results",
        "ConformationSet",
    ),
    "StateDetector": (
        "pmarlo.conformations.state_detection",
        "StateDetector",
    ),
    "RepresentativePicker": (
        "pmarlo.conformations.representative_picker",
        "RepresentativePicker",
    ),
    "UncertaintyQuantifier": (
        "pmarlo.conformations.uncertainty",
        "UncertaintyQuantifier",
    ),
    "TPTResult": (
        "pmarlo.conformations.results",
        "TPTResult",
    ),
    "KISResult": (
        "pmarlo.conformations.results",
        "KISResult",
    ),
    "UncertaintyResult": (
        "pmarlo.conformations.results",
        "UncertaintyResult",
    ),
    "plot_tpt_summary": (
        "pmarlo.conformations.visualizations",
        "plot_tpt_summary",
    ),
    "plot_committors": (
        "pmarlo.conformations.visualizations",
        "plot_committors",
    ),
    "plot_flux_network": (
        "pmarlo.conformations.visualizations",
        "plot_flux_network",
    ),
    "plot_pathways": (
        "pmarlo.conformations.visualizations",
        "plot_pathways",
    ),
}


def __getattr__(name: str) -> Any:
    """Lazy import to avoid loading heavy dependencies at package import."""
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from None

    try:
        module = import_module(module_name)
    except Exception as exc:
        raise ImportError(
            f"Failed to import {module_name!r} required for {name!r}. "
            f"Ensure deeptime and scikit-learn are installed: {exc}"
        ) from exc

    try:
        value = getattr(module, attr_name)
    except AttributeError as exc:
        raise ImportError(
            f"Module {module_name!r} does not expose attribute {attr_name!r}"
        ) from exc

    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """List all public exports."""
    return sorted(__all__)

