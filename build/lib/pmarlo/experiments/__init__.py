"""Experiment framework for algorithm testing in PMARLO."""

from __future__ import annotations

from importlib import import_module
from typing import Any, Dict, Tuple

__all__ = [
    "run_simulation_experiment",
    "run_replica_exchange_experiment",
    "run_msm_experiment",
]

_EXPORTS: Dict[str, Tuple[str, str]] = {
    "run_simulation_experiment": (
        "pmarlo.experiments.simulation",
        "run_simulation_experiment",
    ),
    "run_replica_exchange_experiment": (
        "pmarlo.experiments.replica_exchange",
        "run_replica_exchange_experiment",
    ),
    "run_msm_experiment": ("pmarlo.experiments.msm", "run_msm_experiment"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError:  # pragma: no cover - defensive guard
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
