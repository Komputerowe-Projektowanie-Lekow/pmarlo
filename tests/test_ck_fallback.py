from pathlib import Path

import numpy as np
from PIL import Image

from pmarlo.analysis.ck import run_ck


def _simulate_cycle(n_repeats: int = 1000) -> np.ndarray:
    return np.array([0, 1, 2] * n_repeats, dtype=int)


def test_ck_mse_decreases_and_plot(tmp_path: Path) -> None:
    traj = _simulate_cycle(1000)
    res = run_ck(
        [traj], lag_time=1, output_dir=tmp_path, macro_k=3, min_trans=5, top_n_micro=3
    )
    assert res.mse
    ks = sorted(res.mse.keys())
    assert ks == [2, 3, 4, 5]
    # Mild decrease: last MSE not greater than first
    assert res.mse[ks[-1]] <= res.mse[ks[0]] + 1e-12
    plot = tmp_path / "ck.png"
    assert plot.exists() and plot.stat().st_size > 0


def test_ck_insufficient_overlay(tmp_path: Path) -> None:
    traj = np.array([0, 1, 0, 1], dtype=int)
    res = run_ck(
        [traj], lag_time=1, output_dir=tmp_path, macro_k=2, min_trans=50, top_n_micro=2
    )
    assert not res.mse
    assert set(res.insufficient_k) == {2, 3, 4, 5}
    plot = tmp_path / "ck.png"
    assert plot.exists() and plot.stat().st_size > 0
    img = Image.open(plot)
    arr = np.array(img)
    assert float(arr.mean()) < 255.0
