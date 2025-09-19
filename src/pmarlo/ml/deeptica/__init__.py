"""Curriculum-based DeepTICA training utilities."""

from .trainer import CurriculumConfig, DeepTICACurriculumTrainer
from .whitening import apply_output_transform

__all__ = [
    "CurriculumConfig",
    "DeepTICACurriculumTrainer",
    "apply_output_transform",
]
