"""Curriculum-based DeepTICA training utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .whitening import apply_output_transform

if TYPE_CHECKING:
    from .trainer import CurriculumConfig, DeepTICACurriculumTrainer

__all__ = [
    "apply_output_transform",
    "CurriculumConfig",
    "DeepTICACurriculumTrainer",
]


def __dir__() -> list[str]:
    """Expose exported names for interactive tooling."""

    return sorted(set(__all__))


def __getattr__(name: str):
    """Lazy-load torch-dependent trainer classes on first access."""
    if name == "CurriculumConfig":
        from .trainer import CurriculumConfig

        return CurriculumConfig
    if name == "DeepTICACurriculumTrainer":
        from .trainer import DeepTICACurriculumTrainer

        return DeepTICACurriculumTrainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
