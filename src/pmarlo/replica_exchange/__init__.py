"""Replica-exchange conveniences with optional heavy dependencies."""

from __future__ import annotations

from importlib import import_module
from typing import Any, Dict, Tuple

__all__ = [
    "ExchangeRecord",
    "parse_temperature_ladder",
    "parse_exchange_log",
    "ReplicaExchange",
    "Simulation",
    "prepare_system",
    "production_run",
    "feature_extraction",
    "build_transition_model",
    "relative_energies",
    "plot_DG",
]

_MANDATORY_EXPORTS: Dict[str, Tuple[str, str]] = {
    "ExchangeRecord": ("pmarlo.replica_exchange.demux_compat", "ExchangeRecord"),
    "parse_temperature_ladder": (
        "pmarlo.replica_exchange.demux_compat",
        "parse_temperature_ladder",
    ),
    "parse_exchange_log": (
        "pmarlo.replica_exchange.demux_compat",
        "parse_exchange_log",
    ),
}

_OPTIONAL_EXPORTS: Dict[str, Tuple[str, str]] = {
    "ReplicaExchange": ("pmarlo.replica_exchange.replica_exchange", "ReplicaExchange"),
    "Simulation": ("pmarlo.replica_exchange.simulation", "Simulation"),
    "prepare_system": ("pmarlo.replica_exchange.simulation", "prepare_system"),
    "production_run": ("pmarlo.replica_exchange.simulation", "production_run"),
    "feature_extraction": ("pmarlo.replica_exchange.simulation", "feature_extraction"),
    "build_transition_model": (
        "pmarlo.replica_exchange.simulation",
        "build_transition_model",
    ),
    "relative_energies": ("pmarlo.replica_exchange.simulation", "relative_energies"),
    "plot_DG": ("pmarlo.replica_exchange.simulation", "plot_DG"),
}


def _resolve_export(name: str) -> Any:
    if name in _MANDATORY_EXPORTS:
        module_name, attr_name = _MANDATORY_EXPORTS[name]
    elif name in _OPTIONAL_EXPORTS:
        module_name, attr_name = _OPTIONAL_EXPORTS[name]
    else:  # pragma: no cover - defensive programming
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __getattr__(name: str) -> Any:
    return _resolve_export(name)


def __dir__() -> list[str]:
    return sorted(set(list(__all__) + list(_OPTIONAL_EXPORTS.keys())))
