"""Feature (CV) layer: registry and built-in features.

The module used to eagerly import every deep learning helper, which pulled in
heavy optional dependencies such as PyTorch.  Tests in this kata only need the
balanced sampler utilities, so we expose everything lazily to keep
``import pmarlo.features`` lightweight.
"""

from __future__ import annotations

import os
from importlib import import_module
from typing import Any, Dict, Tuple

from .base import (
    FEATURE_REGISTRY,
    FeatureComputer,
)
from .base import get_feature as _base_get_feature
from .base import (
    register_feature,
)

__all__ = ["FEATURE_REGISTRY", "get_feature", "register_feature"]

_OPTIONAL_EXPORTS: Dict[str, Tuple[str, str]] = {
    "CVModel": ("pmarlo.features.collective_variables", "CVModel"),
    "LaggedPairs": ("pmarlo.features.data_loaders", "LaggedPairs"),
    "make_loaders": ("pmarlo.features.data_loaders", "make_loaders"),
    "DeepTICAConfig": ("pmarlo.features.deeptica", "DeepTICAConfig"),
    "DeepTICAModel": ("pmarlo.features.deeptica", "DeepTICAModel"),
    "train_deeptica": ("pmarlo.features.deeptica", "train_deeptica"),
    "PairDiagItem": ("pmarlo.features.diagnostics", "PairDiagItem"),
    "PairDiagReport": ("pmarlo.features.diagnostics", "PairDiagReport"),
    "diagnose_deeptica_pairs": (
        "pmarlo.features.diagnostics",
        "diagnose_deeptica_pairs",
    ),
    "make_training_pairs_from_shards": (
        "pmarlo.features.pairs",
        "make_training_pairs_from_shards",
    ),
    "scaled_time_pairs": ("pmarlo.features.pairs", "scaled_time_pairs"),
    "RamachandranResult": ("pmarlo.features.ramachandran", "RamachandranResult"),
    "compute_ramachandran": ("pmarlo.features.ramachandran", "compute_ramachandran"),
    "compute_ramachandran_fes": (
        "pmarlo.features.ramachandran",
        "compute_ramachandran_fes",
    ),
    "periodic_hist2d": ("pmarlo.features.ramachandran", "periodic_hist2d"),
}

__all__.extend(sorted(_OPTIONAL_EXPORTS.keys()))

_BUILTINS_IMPORT_ERROR: ModuleNotFoundError | None = None

_BUILTINS_IMPORT_ERROR: ModuleNotFoundError | None = None
_BUILTINS_READY = False
_FORCE_MDTRAJ_MISSING = bool(
    int(os.environ.get("PMARLO_FORCE_MDTRAJ_MISSING", "0"))  # type: ignore[arg-type]
    if os.environ.get("PMARLO_FORCE_MDTRAJ_MISSING") is not None
    else 0
)


def _ensure_builtins_loaded() -> None:
    """Import builtin molecular features on-demand.

    Importing at module import time eagerly pulled in mdtraj, breaking
    environments where the optional dependency is intentionally absent.  By
    deferring to first use we keep ``import pmarlo.features`` lightweight while
    still surfacing a descriptive error once a built-in feature is requested.
    """
    global _BUILTINS_READY, _BUILTINS_IMPORT_ERROR
    if _BUILTINS_READY or _BUILTINS_IMPORT_ERROR is not None:
        return
    if _FORCE_MDTRAJ_MISSING:
        _BUILTINS_IMPORT_ERROR = ModuleNotFoundError("mdtraj (forced missing)")
        return
    try:
        import_module("pmarlo.features.builtins")
    except ModuleNotFoundError as exc:  # pragma: no cover - optional deps
        if getattr(exc, "name", None) != "mdtraj":
            raise
        _BUILTINS_IMPORT_ERROR = exc
    else:
        _BUILTINS_READY = True


def get_feature(name: str) -> FeatureComputer:
    """Retrieve a registered feature, ensuring optional dependencies are present."""

    _ensure_builtins_loaded()
    try:
        return _base_get_feature(name)
    except KeyError as exc:
        if _BUILTINS_IMPORT_ERROR is not None:
            raise ModuleNotFoundError(
                "Built-in molecular features require the optional 'mdtraj' dependency. "
                "Install mdtraj and re-import pmarlo.features to access them."
            ) from _BUILTINS_IMPORT_ERROR
        raise


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _OPTIONAL_EXPORTS[name]
    except KeyError:  # pragma: no cover - defensive guard
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
