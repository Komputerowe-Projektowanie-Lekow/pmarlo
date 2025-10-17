"""Curriculum-based DeepTICA training utilities."""

from __future__ import annotations

from .trainer import CurriculumConfig, DeepTICACurriculumTrainer
from .whitening import apply_output_transform

__all__ = [
    "apply_output_transform",
    "CurriculumConfig",
    "DeepTICACurriculumTrainer",
]


def __dir__() -> list[str]:
    """Expose exported names for interactive tooling."""

    return sorted(set(__all__))
