# Copyright (c) 2025 PMARLO Development Team
# SPDX-License-Identifier: GPL-3.0-or-later

"""PMARLO: Protein Markov State Model analysis for single-trajectory workflows."""

from __future__ import annotations

from importlib import import_module
from typing import Any

try:
    from ._version import __version__
except ImportError:  # pragma: no cover
    __version__ = "0.0.0.dev0"

__author__ = "PMARLO Development Team"

_MANDATORY_EXPORTS: dict[str, tuple[str, str]] = {
    "Protein": ("pmarlo.protein.protein", "Protein"),
    "candidate_lag_ladder": (
        "pmarlo.markov_state_model._msm_utils",
        "candidate_lag_ladder",
    ),
}

_MODULE_EXPORTS: dict[str, str] = {
    "api": "pmarlo.api",
    "visualization": "pmarlo.visualization",
}

_OPTIONAL_EXPORTS: dict[str, tuple[str, str]] = {
    "MarkovStateModel": ("pmarlo.markov_state_model.enhanced_msm", "EnhancedMSM"),
    "FESResult": ("pmarlo.markov_state_model.free_energy", "FESResult"),
    "PMFResult": ("pmarlo.markov_state_model.free_energy", "PMFResult"),
    "generate_1d_pmf": ("pmarlo.markov_state_model.free_energy", "generate_1d_pmf"),
    "generate_2d_fes": ("pmarlo.markov_state_model.free_energy", "generate_2d_fes"),
}

_ANALYSIS_HINT = "Install with `pip install 'pmarlo[analysis]'`."

_OPTIONAL_HINTS: dict[str, str] = {
    "api": _ANALYSIS_HINT,
    "visualization": _ANALYSIS_HINT,
    "MarkovStateModel": _ANALYSIS_HINT,
    "FESResult": _ANALYSIS_HINT,
    "PMFResult": _ANALYSIS_HINT,
    "generate_1d_pmf": _ANALYSIS_HINT,
    "generate_2d_fes": _ANALYSIS_HINT,
}

__all__ = [
    "__version__",
    "__author__",
    # Main API
    "Protein",
    "MarkovStateModel",
    # Utilities
    "candidate_lag_ladder",
    # Submodules
    "api",
    "visualization",
    # Free energy
    "FESResult",
    "PMFResult",
    "generate_1d_pmf",
    "generate_2d_fes",
]


def _bind_export(name: str, value: Any) -> Any:
    """Bind a lazily imported object into the module namespace."""
    globals()[name] = value
    return value


def _load_module(module_name: str, *, feature: str, optional: bool) -> Any:
    """Import a module and add a clear hint for optional PMARLO features."""
    try:
        return import_module(module_name)
    except ImportError as exc:
        if optional:
            hint = _OPTIONAL_HINTS.get(feature, "")
            message = (
                f"Cannot import {feature!r}. This PMARLO feature requires "
                f"optional analysis dependencies."
            )

            if hint:
                message = f"{message} {hint}"

            raise ImportError(message) from exc

        raise ImportError(
            f"Cannot import required PMARLO component {feature!r} "
            f"from module {module_name!r}."
        ) from exc


def _resolve_export(name: str) -> Any:
    """Resolve one public PMARLO export."""
    if name in _MODULE_EXPORTS:
        module_name = _MODULE_EXPORTS[name]
        module = _load_module(
            module_name,
            feature=name,
            optional=name in _OPTIONAL_HINTS,
        )
        return _bind_export(name, module)

    if name in _MANDATORY_EXPORTS:
        module_name, attr_name = _MANDATORY_EXPORTS[name]
        optional = False
    elif name in _OPTIONAL_EXPORTS:
        module_name, attr_name = _OPTIONAL_EXPORTS[name]
        optional = True
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = _load_module(module_name, feature=name, optional=optional)

    try:
        value = getattr(module, attr_name)
    except AttributeError as exc:
        raise ImportError(
            f"Module {module_name!r} does not define attribute {attr_name!r} "
            f"needed for public export {name!r}."
        ) from exc

    return _bind_export(name, value)


def __getattr__(name: str) -> Any:
    """Lazily resolve public PMARLO attributes."""
    return _resolve_export(name)


def __dir__() -> list[str]:
    """Return public names shown by dir(pmarlo)."""
    return sorted(set(globals()) | set(__all__))
