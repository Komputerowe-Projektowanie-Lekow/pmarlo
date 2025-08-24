from pathlib import Path

import numpy as np

from pmarlo.markov_state_model.enhanced_msm import EnhancedMSM


def _pattern_traj(repeats: int = 500) -> np.ndarray:
    pattern = [0, 0, 1, 1]
    return np.array(pattern * repeats, dtype=int)


def test_ck_tau_selection(tmp_path: Path, capsys):
    traj = _pattern_traj()
    msm = EnhancedMSM(output_dir=str(tmp_path))
    msm.dtrajs = [traj]
    msm.n_states = 2
    selected = msm.select_lag_time_ck([1, 2, 3])
    assert selected == 2
    assert msm.lag_time == 2

    csv_path = tmp_path / "ck_mse.csv"
    png_path = tmp_path / "ck.png"
    assert csv_path.exists() and png_path.exists()

    out = capsys.readouterr().out
    assert "Selected τ =" in out
