from __future__ import annotations

import pytest

from pmarlo.features.deeptica_trainer.config import TrainerConfig, resolve_curriculum


def test_trainer_config_normalises_fields():
    cfg = TrainerConfig(
        tau_steps=5,
        tau_schedule=(4, 0, 2),
        grad_clip_mode="VALUE",
        log_every=0,
    )
    assert cfg.tau_schedule == (4, 2)
    assert cfg.grad_clip_mode == "value"
    assert cfg.log_every == 1


def test_trainer_config_requires_positive_tau():
    with pytest.raises(ValueError):
        TrainerConfig(tau_steps=0)


def test_trainer_config_cosine_requires_total_steps():
    with pytest.raises(ValueError):
        TrainerConfig(tau_steps=1, scheduler="cosine")

    cfg = TrainerConfig(tau_steps=1, scheduler="cosine", scheduler_total_steps=50)
    assert resolve_curriculum(cfg) == [1]


def test_resolve_curriculum_uses_schedule():
    cfg = TrainerConfig(tau_steps=4, tau_schedule=(3, 6, 9))
    assert resolve_curriculum(cfg) == [3, 6, 9]
