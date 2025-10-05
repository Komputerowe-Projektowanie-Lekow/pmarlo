"""Thin façade over the canonical DeepTICA curriculum trainer."""

from __future__ import annotations

from pmarlo.ml.deeptica.trainer import CurriculumConfig as TrainerConfig
from pmarlo.ml.deeptica.trainer import DeepTICACurriculumTrainer as DeepTICATrainer

__all__ = ["TrainerConfig", "DeepTICATrainer"]


# Re-exporting keeps historical import sites working while ensuring the
# heavyweight training logic lives in :mod:`pmarlo.ml.deeptica.trainer`.
