from __future__ import annotations

from pmarlo.replica_exchange.diagnostics import compute_diffusion_metrics


def test_compute_diffusion_metrics_basic():
    # 3 replicas, simple up-down motion
    history = [
        [0, 1, 2],
        [1, 1, 1],
        [0, 1, 2],
        [1, 1, 1],
    ]
    # exchange_frequency_steps = 500
    m = compute_diffusion_metrics(history, 500)
    assert m["mean_abs_disp_per_sweep"] > 0
    assert m["mean_abs_disp_per_10k_steps"] == m["mean_abs_disp_per_sweep"] * (
        10000.0 / 500.0
    )
    assert isinstance(m["sparkline"], list)
