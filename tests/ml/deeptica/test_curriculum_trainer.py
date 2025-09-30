from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from pmarlo.ml.deeptica.trainer import CurriculumConfig, DeepTICACurriculumTrainer


class TinyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def _ar1_process(rho: float, length: int, rng: np.random.Generator) -> np.ndarray:
    noise_scale = np.sqrt(1.0 - rho**2)
    x = np.zeros(length, dtype=np.float32)
    for i in range(1, length):
        x[i] = rho * x[i - 1] + rng.normal(scale=noise_scale)
    return x


def _make_sequences(
    n_shards: int, length: int, rng: np.random.Generator
) -> list[np.ndarray]:
    sequences: list[np.ndarray] = []
    for _ in range(n_shards):
        slow = _ar1_process(0.995, length, rng)
        fast = _ar1_process(0.3, length, rng)
        stacked = np.stack([slow, fast], axis=1).astype(np.float32)
        sequences.append(stacked)
    return sequences


def _copy_state_dict(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: v.clone() for k, v in state.items()}


def test_curriculum_outperforms_single_tau(tmp_path, caplog):
    rng = np.random.default_rng(12345)
    sequences = _make_sequences(n_shards=4, length=400, rng=rng)

    torch.manual_seed(2024)
    base_state = _copy_state_dict(TinyNet().state_dict())

    single_cfg = CurriculumConfig(
        tau_schedule=(2,),
        val_tau=20,
        epochs_per_tau=4,
        batch_size=128,
        learning_rate=3e-3,
        weight_decay=1e-5,
        val_fraction=0.25,
        log_every=0,
        checkpoint_dir=tmp_path / "single",
        seed=123,
    )
    single_model = TinyNet()
    single_model.load_state_dict(base_state)
    single_trainer = DeepTICACurriculumTrainer(single_model, single_cfg)
    single_history = single_trainer.fit(sequences)
    baseline_score = single_history["best_val_score"]

    curriculum_cfg = CurriculumConfig(
        tau_schedule=(2, 5, 10, 20, 40),
        val_tau=20,
        epochs_per_tau=4,
        batch_size=128,
        learning_rate=3e-3,
        weight_decay=1e-5,
        val_fraction=0.25,
        log_every=1,
        checkpoint_dir=tmp_path / "curriculum",
        seed=456,
    )
    curriculum_model = TinyNet()
    curriculum_model.load_state_dict(base_state)
    curriculum_trainer = DeepTICACurriculumTrainer(curriculum_model, curriculum_cfg)

    caplog.set_level(logging.INFO)
    curriculum_history = curriculum_trainer.fit(sequences)
    improved_score = curriculum_history["best_val_score"]

    assert improved_score > baseline_score
    assert curriculum_history["val_tau"] == 20
    assert 20 in curriculum_history["per_tau_objective_curve"]
    assert hasattr(curriculum_model, "training_history")
    assert curriculum_model.training_history["val_tau"] == 20
    best_path = curriculum_history.get("best_checkpoint")
    assert best_path is not None and Path(best_path).exists()
    assert "val_tau=20" in caplog.text
    lr_curve = curriculum_history["learning_rate_curve"]
    assert len(lr_curve) == len(curriculum_history["epochs"])
    warmup_epochs = min(curriculum_cfg.warmup_epochs, len(lr_curve))
    if warmup_epochs:
        expected_first = curriculum_cfg.learning_rate / float(max(1, warmup_epochs))
        assert np.isclose(lr_curve[0], expected_first, rtol=1e-4)
        assert np.isclose(
            lr_curve[warmup_epochs - 1], curriculum_cfg.learning_rate, rtol=1e-4
        )
        if warmup_epochs > 1:
            assert np.all(np.diff(lr_curve[:warmup_epochs]) > 0)
    if len(lr_curve) > max(1, warmup_epochs):
        assert lr_curve[-1] < lr_curve[max(0, warmup_epochs - 1)]
    grad_max_curve = curriculum_history["grad_norm_max_curve"]
    assert len(grad_max_curve) == len(curriculum_history["epochs"])
    if curriculum_cfg.grad_clip_norm is not None:
        assert max(grad_max_curve) <= float(curriculum_cfg.grad_clip_norm) + 1e-5
    for block in curriculum_history["per_tau"]:
        assert "learning_rate_curve" in block
        assert "grad_norm_max_curve" in block
