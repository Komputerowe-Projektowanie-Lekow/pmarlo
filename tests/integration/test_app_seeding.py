from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


@pytest.mark.skipif(__import__("importlib.util").util.find_spec("openmm") is None, reason="Requires OpenMM")
def test_fixed_seed_determinism_two_runs(test_fixed_pdb_file: Path, tmp_path: Path):
    from example_programs.app_usecase.app.backend import run_short_sim
    from pmarlo.io.trajectory_reader import MDTrajReader

    ws = tmp_path
    temps = [300.0, 320.0, 340.0]

    sim1 = run_short_sim(Path(test_fixed_pdb_file), ws, temps, steps=500, quick=True, random_seed=123456)
    sim2 = run_short_sim(Path(test_fixed_pdb_file), ws, temps, steps=500, quick=True, random_seed=123456)

    # Use returned demuxed or first trajectory file
    traj1 = Path(sim1.traj_files[0])
    traj2 = Path(sim2.traj_files[0])

    reader = MDTrajReader(topology_path=str(test_fixed_pdb_file))
    # Probe at least one frame and compare numerically
    frames1 = []
    for f in reader.iter_frames(str(traj1), start=0, stop=1, stride=1):
        frames1.append(np.asarray(f))
        break
    frames2 = []
    for f in reader.iter_frames(str(traj2), start=0, stop=1, stride=1):
        frames2.append(np.asarray(f))
        break
    assert frames1 and frames2
    assert frames1[0].shape == frames2[0].shape
    assert np.allclose(frames1[0], frames2[0])


@pytest.mark.skipif(__import__("importlib.util").util.find_spec("openmm") is None, reason="Requires OpenMM")
def test_auto_seed_varies_trajectories(test_fixed_pdb_file: Path, tmp_path: Path):
    from example_programs.app_usecase.app.backend import run_short_sim, choose_sim_seed
    from pmarlo.io.trajectory_reader import MDTrajReader

    ws = tmp_path
    temps = [300.0, 320.0, 340.0]

    seed1 = choose_sim_seed("auto")
    seed2 = choose_sim_seed("auto")
    assert seed1 != seed2

    sim1 = run_short_sim(Path(test_fixed_pdb_file), ws, temps, steps=400, quick=True, random_seed=seed1)
    sim2 = run_short_sim(Path(test_fixed_pdb_file), ws, temps, steps=400, quick=True, random_seed=seed2)

    traj1 = Path(sim1.traj_files[0])
    traj2 = Path(sim2.traj_files[0])
    reader = MDTrajReader(topology_path=str(test_fixed_pdb_file))
    f1 = next(reader.iter_frames(str(traj1), start=0, stop=1, stride=1))
    f2 = next(reader.iter_frames(str(traj2), start=0, stop=1, stride=1))
    # Very likely to differ; allow rare equality by negating allclose
    assert not np.allclose(f1, f2)

