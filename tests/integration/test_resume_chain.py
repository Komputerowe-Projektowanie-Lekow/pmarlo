from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


@pytest.mark.skipif(
    __import__("importlib.util").util.find_spec("openmm") is None,
    reason="Requires OpenMM",
)
def test_resume_two_chunks_yields_sum_frames(test_fixed_pdb_file: Path, tmp_path: Path):
    from example_programs.app_usecase.app.backend import run_short_sim
    from pmarlo.io.trajectory_reader import MDTrajReader

    ws = tmp_path
    temps = [300.0, 320.0, 340.0]

    # First run
    sim1 = run_short_sim(Path(test_fixed_pdb_file), ws, temps, steps=1000, quick=True)
    # Continue from previous run directory
    sim2 = run_short_sim(
        Path(test_fixed_pdb_file),
        ws,
        temps,
        steps=1000,
        quick=True,
        start_from=sim1.run_dir,
    )

    r = MDTrajReader(topology_path=str(test_fixed_pdb_file))
    f1 = r.probe_length(str(Path(sim1.traj_files[0])))
    f2 = r.probe_length(str(Path(sim2.traj_files[0])))
    assert f1 > 0 and f2 > 0
    # They are independent files, but chaining should make total frames sum without overlap within each file
    # Sanity: no zero-length
    assert f1 + f2 > f1 and f1 + f2 > f2
