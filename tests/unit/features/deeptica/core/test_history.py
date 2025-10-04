from __future__ import annotations

import numpy as np
import torch

from pmarlo.features.deeptica.core.history import (
    LossHistory,
    collect_history_metrics,
    project_model,
    summarize_history,
    vamp2_proxy,
)


def test_loss_history_tracks_train_and_val_curves():
    history = LossHistory()
    history.record_train({"loss": 1.2})
    history.record_val({"val_loss": 0.8, "val_score": 0.9})
    assert history.losses == [1.2]
    assert history.val_losses == [0.8]
    assert history.val_scores == [0.9]


def test_collect_history_metrics_normalises_types():
    metrics = collect_history_metrics({
        "loss_curve": ["1", 2.0],
        "val_loss_curve": ["0.5"],
        "val_score_curve": [0.75],
    })
    assert metrics["loss_curve"] == [1.0, 2.0]
    assert metrics["val_loss_curve"] == [0.5]
    assert metrics["val_score_curve"] == [0.75]


def test_vamp2_proxy_returns_scalar():
    rng = np.random.default_rng(0)
    Y = rng.normal(size=(20, 3)).astype(np.float32)
    idx_t = np.arange(0, 15, dtype=np.int64)
    idx_tau = idx_t + 1
    value = vamp2_proxy(Y, idx_t, idx_tau)
    assert isinstance(value, float)
    assert value >= 0.0


def test_project_model_converts_to_numpy():
    net = torch.nn.Linear(2, 1)
    Z = np.random.rand(5, 2).astype(np.float32)
    projected = project_model(net, Z)
    assert projected.shape == (5, 1)


def test_summarize_history_copies_lists():
    history = LossHistory(losses=[1.0], val_losses=[0.5], val_scores=[0.9])
    summary = summarize_history(history)
    assert summary["loss_curve"] == [1.0]
    assert summary["val_loss_curve"] == [0.5]
    assert summary["val_score_curve"] == [0.9]
