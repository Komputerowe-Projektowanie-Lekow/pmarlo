"""Tests for the Chapman–Kolmogorov diagnostics."""

import sys
from pathlib import Path

import numpy as np
import pytest

from pmarlo.markov_state_model.enhanced_msm import EnhancedMSM


def _simulate_cycle(n_repeats: int = 1000) -> np.ndarray:
    """Generate a deterministic 3-state cycle trajectory."""
    return np.array([0, 1, 2] * n_repeats, dtype=int)


def test_ck_test_returns_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """CK MSE computed and plot contains content."""
    traj = _simulate_cycle(1000)
    msm = EnhancedMSM(output_dir=str(tmp_path))
    msm.dtrajs = [traj]
    msm.n_states = 3
    msm.lag_time = 1

    C = np.zeros((3, 3), dtype=float)
    for i in range(len(traj) - 1):
        C[traj[i], traj[i + 1]] += 1
    rows = C.sum(axis=1)
    rows[rows == 0] = 1.0
    msm.transition_matrix = C / rows[:, None]

    res = msm.compute_ck_test_micro()
    ks = sorted(res.mse.keys())
    assert ks == [2, 3, 4, 5]
    assert all(v >= 0.0 for v in res.mse.values())

    monkeypatch.setitem(sys.modules, "deeptime", None)
    plot_path = msm.plot_ck_test(save_file=tmp_path / "ck.png")
    assert plot_path is not None
    assert plot_path.exists() and plot_path.stat().st_size > 0
