"""Compatibility façade around the canonical DeepTICA curriculum config."""

from __future__ import annotations

from typing import Iterable, List

from pmarlo.ml.deeptica.trainer import CurriculumConfig

__all__ = ["TrainerConfig", "resolve_curriculum"]

# Alias the canonical configuration so legacy imports continue to work.
TrainerConfig = CurriculumConfig


def resolve_curriculum(cfg: TrainerConfig) -> List[int]:
    """Return the ordered tau curriculum as positive integers."""

    schedule: Iterable[int] = getattr(cfg, "tau_schedule", ())
    return [max(1, int(step)) for step in schedule]
