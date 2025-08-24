import logging
from pathlib import Path

import mdtraj as md
import numpy as np
import pytest

from pmarlo.markov_state_model.enhanced_msm import EnhancedMSM


def _dummy_traj(n_frames: int) -> md.Trajectory:
    top = md.Topology()
    chain = top.add_chain()
    res = top.add_residue("ALA", chain)
    top.add_atom("CA", md.element.carbon, res)
    coords = np.zeros((n_frames, 1, 3))
    return md.Trajectory(coords, top)


def test_effective_frames_and_tau_guard(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="pmarlo")
    traj = _dummy_traj(1800)
    msm = EnhancedMSM(output_dir=str(tmp_path))
    msm.trajectories = [traj]
    msm.compute_features(feature_stride=8, tica_lag=200, tica_components=2)
    expected = (1800 // 8) - 200
    assert msm.effective_frames == expected
    with pytest.raises(ValueError):
        msm.build_msm(lag_time=200)
    assert f"effective frames after lag 200: {expected}" in caplog.text


def test_used_frames_with_tica_lag(tmp_path: Path) -> None:
    traj = _dummy_traj(1000)
    msm = EnhancedMSM(output_dir=str(tmp_path))
    msm.trajectories = [traj]
    msm.compute_features(tica_lag=100, tica_components=2)
    assert msm.effective_frames >= 900
