from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pmarlo.features.deeptica_trainer.loops import (
    checkpoint_if_better,
    compute_grad_norm,
    compute_loss_and_score,
    make_metrics,
    prepare_batch,
    record_metrics,
)


class _DummyLoss(torch.nn.Module):
    def forward(self, z_t, z_tau, weights):  # type: ignore[override]
        diff = z_t - z_tau
        loss = (diff * diff).mean()
        score = (z_t * z_tau).mean()
        return loss, score


class _Model:
    def __init__(self) -> None:
        self.training_history: dict[str, list[dict[str, float]]] = {}


def test_prepare_batch_and_metrics(tmp_path):
    x0 = np.arange(12, dtype=np.float32).reshape(6, 2)
    x1 = x0 + 1.0
    batch = [(x0, x1, None)]
    try:
        tensors = prepare_batch(
            batch, torch_mod=torch, device=torch.device("cpu"), use_weights=False
        )
    except NotImplementedError as exc:
        pytest.skip(f"DeepTICA training helpers unavailable: {exc}")
    assert tensors is not None
    x_t, x_tau, weights = tensors
    assert weights is None
    net = torch.nn.Linear(2, 2, bias=False)
    torch.nn.init.eye_(net.weight)
    loss_module = _DummyLoss()

    loss, score = compute_loss_and_score(net, loss_module, x_t, x_tau, weights)
    loss.backward()
    grad_norm = compute_grad_norm(net.parameters())
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
    metrics = make_metrics(
        loss=loss,
        score=score,
        tau=3,
        optimizer=optimizer,
        grad_norm=grad_norm,
    )
    history: list[dict[str, float]] = []
    model = _Model()
    record_metrics(history, metrics, model=model)
    assert history and model.training_history["steps"]

    checkpoint = tmp_path / "model.pt"
    best = checkpoint_if_better(
        torch_mod=torch,
        model_net=net,
        checkpoint_path=checkpoint,
        metrics=metrics,
        metric_name="vamp2",
        best_score=float("-inf"),
    )
    assert checkpoint.exists()
    assert best == metrics["vamp2"]
