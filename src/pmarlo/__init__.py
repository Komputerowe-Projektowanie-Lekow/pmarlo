# Copyright (c) 2025 PMARLO Development Team
# SPDX-License-Identifier: GPL-3.0-or-later

"""
PMARLO: Protein Markov State Model Analysis with Replica Exchange

A Python package for protein simulation and Markov state model chain generation,
providing an OpenMM-like interface for molecular dynamics simulations.
"""

import logging
import sys
from importlib import import_module
from types import ModuleType
from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence, Tuple, Type

from .utils.seed import quiet_external_loggers

if TYPE_CHECKING:  # pragma: no cover - typing only
    # from .markov_state_model.enhanced_msm import EnhancedMSM as MarkovStateModelType
    # from .transform.pipeline import Pipeline as PipelineType
    pass

logger = logging.getLogger("pmarlo")


def _import_or_none(module_name: str) -> tuple[ModuleType | None, BaseException | None]:
    """Return ``(module, exc)`` for the requested module without raising."""
    try:
        module = import_module(module_name)
    except Exception as exc:  # pragma: no cover - optional dependency missing
        return None, exc
    return module, None


def _bind_export(name: str, value: object) -> None:
    """Bind a lazily imported attribute into the module namespace."""
    globals()[name] = value


def _warn_stub(name: str, exc: BaseException | None) -> None:
    """Emit a debug log when falling back to a stub export."""
    if exc is None:
        return
    logger.debug("Falling back to stub export for %s: %s", name, exc)


def _build_api_stub(module_name: str, exc: BaseException | None) -> ModuleType:
    """Create a lightweight stub for ``pmarlo.api`` when extras are missing."""
    stub = ModuleType("pmarlo.api")

    def _missing(*_args: object, **_kwargs: object) -> None:
        raise ImportError(
            (
                "pmarlo.api requires optional analysis dependencies."
                " Install with `pip install 'pmarlo[analysis]'`."
            )
        ) from exc

    stub.cluster_microstates = _missing  # type: ignore[attr-defined]
    try:
        import numpy as _np
    except Exception:  # pragma: no cover - numpy should be available
        _np = None

    def _trig_expand_periodic(X: object, periodic: Sequence[bool]) -> tuple[object, object]:  # type: ignore[override]
        if _np is None:
            raise ImportError("numpy is required for this helper")
        arr = _np.asarray(X, dtype=float)
        if arr.ndim != 2:
            raise ValueError("Input array must be 2D")
        flags = _np.asarray(periodic, dtype=bool)
        if flags.size != arr.shape[1]:
            flags = _np.resize(flags, arr.shape[1])
        cols: list[_np.ndarray] = []
        mapping: list[int] = []
        for idx in range(arr.shape[1]):
            col = arr[:, idx]
            if bool(flags[idx]):
                cols.append(_np.cos(col))
                cols.append(_np.sin(col))
                mapping.extend([idx, idx])
            else:
                cols.append(col)
                mapping.append(idx)
        expanded = _np.vstack(cols).T if cols else arr
        return expanded, _np.asarray(mapping, dtype=int)

    stub._trig_expand_periodic = _trig_expand_periodic  # type: ignore[attr-defined]
    sys.modules[module_name] = stub
    return stub


__version__ = "0.1.0"
__author__ = "PMARLO Development Team"

# Public API that is always available without optional heavy dependencies.
_MANDATORY_EXPORTS: Dict[str, Tuple[str, str]] = {
    "Protein": ("pmarlo.protein.protein", "Protein"),
    "ReplicaExchange": ("pmarlo.replica_exchange.replica_exchange", "ReplicaExchange"),
    "RemdConfig": ("pmarlo.replica_exchange.config", "RemdConfig"),
    "Simulation": ("pmarlo.replica_exchange.simulation", "Simulation"),
    "BuildOpts": ("pmarlo.transform.build", "BuildOpts"),
    "AppliedOpts": ("pmarlo.transform.build", "AppliedOpts"),
    "build_result": ("pmarlo.transform.build", "build_result"),
    "ShardMeta": ("pmarlo.data.shard", "ShardMeta"),
    "write_shard": ("pmarlo.data.shard", "write_shard"),
    "read_shard": ("pmarlo.data.shard", "read_shard"),
    "emit_shards_from_trajectories": (
        "pmarlo.data.emit",
        "emit_shards_from_trajectories",
    ),
    "aggregate_and_build": ("pmarlo.data.aggregate", "aggregate_and_build"),
    "DemuxDataset": ("pmarlo.data.demux_dataset", "DemuxDataset"),
    "build_demux_dataset": ("pmarlo.data.demux_dataset", "build_demux_dataset"),
    "power_of_two_temperature_ladder": (
        "pmarlo.utils.replica_utils",
        "power_of_two_temperature_ladder",
    ),
    "candidate_lag_ladder": (
        "pmarlo.markov_state_model._msm_utils",
        "candidate_lag_ladder",
    ),
    "pm_get_plan": ("pmarlo.transform", "pm_get_plan"),
    "pm_apply_plan": ("pmarlo.transform", "pm_apply_plan"),
}

_MODULE_EXPORTS: Dict[str, str] = {
    "api": "pmarlo.api",
}

# Optional exports depend on ML / analysis stacks. They are imported lazily when
# first accessed.
_OPTIONAL_EXPORTS: Dict[str, Tuple[str, str]] = {
    "Pipeline": ("pmarlo.transform.pipeline", "Pipeline"),
    "MarkovStateModel": ("pmarlo.markov_state_model.enhanced_msm", "EnhancedMSM"),
    "FESResult": ("pmarlo.markov_state_model.free_energy", "FESResult"),
    "PMFResult": ("pmarlo.markov_state_model.free_energy", "PMFResult"),
    "generate_1d_pmf": ("pmarlo.markov_state_model.free_energy", "generate_1d_pmf"),
    "generate_2d_fes": ("pmarlo.markov_state_model.free_energy", "generate_2d_fes"),
}

Pipeline: Optional[Type[Any]] = None
MarkovStateModel: Optional[Type[Any]] = None

# Attempt to eagerly expose optional exports when their dependencies are
# available.  Failures are ignored so that ``import pmarlo`` remains usable in
# lightweight environments (for example, unit tests that only need helpers).
for _name in ("Pipeline", "MarkovStateModel"):
    module_name, attr_name = _OPTIONAL_EXPORTS[_name]
    try:
        loaded_module = import_module(module_name)
        value = getattr(loaded_module, attr_name)
    except Exception:  # pragma: no cover - optional dependency missing
        continue
    else:
        globals()[_name] = value

__all__ = list(_MANDATORY_EXPORTS.keys()) + list(_MODULE_EXPORTS.keys())

for optional in ("Pipeline", "MarkovStateModel"):
    if optional in globals():
        __all__.append(optional)

# Free-energy exports are appended to ``__all__`` only when import succeeds.
for optional in ("FESResult", "PMFResult", "generate_1d_pmf", "generate_2d_fes"):
    module_name, attr_name = _OPTIONAL_EXPORTS[optional]
    optional_module, exc = _import_or_none(module_name)
    if optional_module is None:
        _warn_stub(optional, exc)
        continue
    value = getattr(optional_module, attr_name)
    _bind_export(optional, value)
    __all__.append(optional)


def _resolve_export(name: str) -> Any:
    if name in _MODULE_EXPORTS:
        module_name = _MODULE_EXPORTS[name]
        module, exc = _import_or_none(module_name)
        if module is None:
            if exc is None:
                raise ImportError(f"Could not import module {module_name!r} for {name!r}")
            if name != "api":  # pragma: no cover - defensive guard for other modules
                raise exc
            module = _build_api_stub(module_name, exc)
            _warn_stub(name, exc)
        _bind_export(name, module)
        return module

    if name in _MANDATORY_EXPORTS:
        module_name, attr_name = _MANDATORY_EXPORTS[name]
    elif name in _OPTIONAL_EXPORTS:
        module_name, attr_name = _OPTIONAL_EXPORTS[name]
    else:  # pragma: no cover - defensive guard
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module, exc = _import_or_none(module_name)
    if module is None:
        _warn_stub(name, exc)
        if exc is None:
            raise ImportError(f"Could not import optional dependency for {name!r}")
        raise ImportError(f"Could not import optional dependency for {name!r}") from exc
    value = getattr(module, attr_name)
    _bind_export(name, value)
    return value


def __getattr__(name: str) -> Any:
    return _resolve_export(name)


def __dir__() -> list[str]:
    return sorted(set(list(__all__) + ["Pipeline", "MarkovStateModel"]))


# Reduce noise from third-party libraries upon import
quiet_external_loggers()
