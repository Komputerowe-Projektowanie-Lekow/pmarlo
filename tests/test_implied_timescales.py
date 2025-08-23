"""Tests for implied timescale stabilization and plateau detection."""

from __future__ import annotations

import numpy as np

from pmarlo.markov_state_model.markov_state_model import EnhancedMSM


def _simulate_dtraj(T: np.ndarray, n_steps: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    traj = np.empty(n_steps, dtype=int)
    traj[0] = 0
    for i in range(1, n_steps):
        traj[i] = rng.choice(T.shape[0], p=T[traj[i - 1]])
    return traj


def test_implied_timescales_plateau(tmp_path):
    """Synthetic MSM recovers known timescale plateau within tolerance."""

    T = np.array([[0.95, 0.05], [0.05, 0.95]])
    dtraj = _simulate_dtraj(T, 50_000, seed=1)

    msm = EnhancedMSM(output_dir=str(tmp_path))
    msm.dtrajs = [dtraj]
    msm.n_states = 2

    result = msm.compute_implied_timescales(
        lag_times=list(range(1, 11)),
        n_timescales=1,
        n_samples=50,
        dirichlet_alpha=1.0,
        m=1,
        epsilon=0.1,
        random_seed=1,
    )

    assert result.recommended_lag_window is not None
    start, end = result.recommended_lag_window
    true_ts = -1.0 / np.log(np.linalg.eigvals(T)[1])
    idxs = [i for i, lag in enumerate(result.lag_times) if start <= lag <= end]
    est = result.timescales[idxs, 0]
    assert np.all(np.abs(est - true_ts) / true_ts < 0.2)
