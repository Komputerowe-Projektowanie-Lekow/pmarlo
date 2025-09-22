"""Curriculum-based DeepTICA training utilities.

This subpackage exposes both lightweight whitening helpers and the
curriculum trainer implementation.  The trainer has a hard dependency on
PyTorch which is not required for many workflows (including the unit
tests in this repository).  Importing the trainer lazily keeps the base
``pmarlo`` package importable in minimal environments while preserving
the public API surface for downstream users that rely on it.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

from .whitening import apply_output_transform

if TYPE_CHECKING:  # pragma: no cover - typing-only imports
    from .trainer import CurriculumConfig as _CurriculumConfig
    from .trainer import DeepTICACurriculumTrainer as _DeepTICACurriculumTrainer

__all__ = [
    "apply_output_transform",
    "CurriculumConfig",
    "DeepTICACurriculumTrainer",
]


def __getattr__(name: str) -> Any:
    """Lazily import trainer components when requested.

    The trainer depends on PyTorch.  Delaying the import until attribute
    access avoids importing torch during ``import pmarlo`` in lightweight
    test environments that do not ship GPU-enabled dependencies.
    """

    if name in {"CurriculumConfig", "DeepTICACurriculumTrainer"}:
        module = import_module(".trainer", __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Provide ``dir(pmarlo.ml.deeptica)`` results consistent with ``__all__``."""

    return sorted(set(__all__))
