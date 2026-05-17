# Copyright (c) 2025 PMARLO Development Team
# SPDX-License-Identifier: GPL-3.0-or-later

"""PMARLO: Protein Markov State Model analysis for single-trajectory workflows."""

from __future__ import annotations

from importlib import import_module
from typing import Any, Dict, Tuple

from .utils.seed import quiet_external_loggers

try:
    from ._version import __version__
except ImportError:  # pragma: no cover
    __version__ = "0.0.0.dev0"

__author__ = "PMARLO Development Team"

_MANDATORY_EXPORTS: Dict[str, Tuple[str, str]] = {
    "Protein": ("pmarlo.protein.protein", "Protein"),
    "candidate_lag_ladder": (
        "pmarlo.markov_state_model._msm_utils",
        "candidate_lag_ladder",
    ),
}

_MODULE_EXPORTS: Dict[str, str] = {
    "api": "pmarlo.api",
    "visualization": "pmarlo.visualization",
}

_OPTIONAL_EXPORTS: Dict[str, Tuple[str, str]] = {
    "MarkovStateModel": ("pmarlo.markov_state_model.enhanced_msm", "EnhancedMSM"),
    "FESResult": ("pmarlo.markov_state_model.free_energy", "FESResult"),
    "PMFResult": ("pmarlo.markov_state_model.free_energy", "PMFResult"),
    "generate_1d_pmf": ("pmarlo.markov_state_model.free_energy", "generate_1d_pmf"),
    "generate_2d_fes": ("pmarlo.markov_state_model.free_energy", "generate_2d_fes"),
}

_ANALYSIS_HINT = "Install with `pip install 'pmarlo[analysis]'`."
_OPTIONAL_HINTS: Dict[str, str] = {
    "api": _ANALYSIS_HINT,
    "MarkovStateModel": _ANALYSIS_HINT,
    "FESResult": _ANALYSIS_HINT,
    "PMFResult": _ANALYSIS_HINT,
    "generate_1d_pmf": _ANALYSIS_HINT,
    "generate_2d_fes": _ANALYSIS_HINT,
}

__all__ = list(_MANDATORY_EXPORTS.keys()) + list(_MODULE_EXPORTS.keys())


def _bind_export(name: str, value: object) -> None:
    """Bind a lazily imported attribute into the module namespace."""

    globals()[name] = value


def _load_module(module_name: str, *, feature: str, optional: bool) -> Any:
    """Import ``module_name`` and raise with a clear message on failure."""

    try:
        return import_module(module_name)
    except Exception as exc:  # pragma: no cover - dependency missing
        hint = _OPTIONAL_HINTS.get(feature)
        if optional and hint is not None:
            raise ImportError(
                f"{feature} requires optional dependency {module_name!r}. {hint}"
            ) from exc
        raise ImportError(
            f"Failed to import required module {module_name!r} for {feature!r}."
        ) from exc


def _resolve_export(name: str) -> Any:
    if name in _MODULE_EXPORTS:
        module_name = _MODULE_EXPORTS[name]
        module = _load_module(
            module_name, feature=name, optional=name in _OPTIONAL_HINTS
        )
        _bind_export(name, module)
        return module

    if name in _MANDATORY_EXPORTS:
        module_name, attr_name = _MANDATORY_EXPORTS[name]
        module = _load_module(module_name, feature=name, optional=False)
    elif name in _OPTIONAL_EXPORTS:
        module_name, attr_name = _OPTIONAL_EXPORTS[name]
        module = _load_module(module_name, feature=name, optional=True)
    else:  # pragma: no cover - defensive guard
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    try:
        value = getattr(module, attr_name)
    except AttributeError as exc:  # pragma: no cover - defensive guard
        raise ImportError(
            f"Module {module_name!r} does not define attribute {attr_name!r} for {name!r}."
        ) from exc

    _bind_export(name, value)
    if name in _OPTIONAL_EXPORTS and name not in __all__:
        __all__.append(name)
    return value


def __getattr__(name: str) -> Any:
    return _resolve_export(name)


def __dir__() -> list[str]:
    names = set(__all__)
    names.update(_OPTIONAL_EXPORTS.keys())
    return sorted(names)


quiet_external_loggers()
