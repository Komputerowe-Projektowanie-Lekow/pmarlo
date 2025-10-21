"""DeepTICA feature helpers that require the full optional dependency stack."""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Any

from pmarlo.features.deeptica.cv_bias_potential import (
    CVBiasPotential,
    HarmonicExpansionBias,
    create_cv_bias_potential,
)

# Export standalone modules that don't require full training stack
from pmarlo.features.deeptica.export import (
    CVModelBundle,
    export_cv_bias_potential,
    export_cv_model,
    load_cv_model_info,
)
from pmarlo.features.deeptica.openmm_integration import (
    CVBiasForce,
    add_cv_bias_to_system,
    check_openmm_torch_available,
    create_cv_torch_force,
)

_STANDALONE_EXPORTS: dict[str, Any] = {
    "CVBiasPotential": CVBiasPotential,
    "HarmonicExpansionBias": HarmonicExpansionBias,
    "create_cv_bias_potential": create_cv_bias_potential,
    "CVModelBundle": CVModelBundle,
    "export_cv_model": export_cv_model,
    "export_cv_bias_potential": export_cv_bias_potential,
    "load_cv_model_info": load_cv_model_info,
    "CVBiasForce": CVBiasForce,
    "add_cv_bias_to_system": add_cv_bias_to_system,
    "check_openmm_torch_available": check_openmm_torch_available,
    "create_cv_torch_force": create_cv_torch_force,
}

__all__ = sorted(_STANDALONE_EXPORTS)

_FULL_MODULE: ModuleType | None = None
_EXPORTED_NAMES: tuple[str, ...] = ()


def _load_full() -> ModuleType:
    """Import the heavy implementation lazily to avoid circular imports."""

    global _FULL_MODULE, _EXPORTED_NAMES
    if _FULL_MODULE is None:
        module = import_module(f"{__name__}._full")
        _FULL_MODULE = module
        exported = getattr(module, "__all__", None)
        if exported is None:
            exported = tuple(name for name in vars(module) if not name.startswith("_"))
        else:
            exported = tuple(exported)
        _EXPORTED_NAMES = exported
        for name in exported:
            globals()[name] = getattr(module, name)
        # Merge with standalone exports
        all_exports = sorted(set(exported) | set(_STANDALONE_EXPORTS))
        globals()["__all__"] = all_exports
    return _FULL_MODULE


def __getattr__(name: str) -> Any:
    standalone = _STANDALONE_EXPORTS.get(name)
    if standalone is not None:
        globals()[name] = standalone
        return standalone

    # Not a standalone export, try the _full module
    module = _load_full()
    try:
        value = getattr(module, name)
    except AttributeError as exc:  # pragma: no cover - mirrors Python behaviour
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'") from exc
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    module = _load_full()
    return sorted(
        set(globals())
        | set(dir(module))
        | set(_EXPORTED_NAMES)
        | set(_STANDALONE_EXPORTS)
    )
