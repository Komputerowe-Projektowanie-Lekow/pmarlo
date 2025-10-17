"""DeepTICA feature helpers that require the full optional dependency stack."""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Any

# Export standalone modules that don't require full training stack
from pmarlo.features.deeptica.export import (
    CVModelBundle,
    export_cv_model,
    load_cv_model_info,
)

from pmarlo.features.deeptica.openmm_integration import (
    CVBiasForce,
    add_cv_bias_to_system,
    check_openmm_torch_available,
    create_cv_torch_force,
)

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
        all_exports = list(exported) + [
            "CVModelBundle",
            "export_cv_model",
            "load_cv_model_info",
            "CVBiasForce",
            "add_cv_bias_to_system",
            "check_openmm_torch_available",
            "create_cv_torch_force",
        ]
        globals()["__all__"] = all_exports
    return _FULL_MODULE


def __getattr__(name: str) -> Any:
    module = _load_full()
    try:
        value = getattr(module, name)
    except AttributeError as exc:  # pragma: no cover - mirrors Python behaviour
        raise AttributeError(
            f"module '{__name__}' has no attribute '{name}'"
        ) from exc
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    module = _load_full()
    standalone_exports = [
        "CVModelBundle",
        "export_cv_model",
        "load_cv_model_info",
        "CVBiasForce",
        "add_cv_bias_to_system",
        "check_openmm_torch_available",
        "create_cv_torch_force",
    ]
    return sorted(
        set(globals()) | set(dir(module)) | set(_EXPORTED_NAMES) | set(standalone_exports)
    )
