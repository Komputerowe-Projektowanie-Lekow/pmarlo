import logging
from pathlib import Path

from pmarlo.markov_state_model.enhanced_msm import EnhancedMSM


def test_iterload_streaming(caplog):
    traj = Path("tests/data/traj.dcd")
    pdb = Path("tests/data/3gd8-fixed.pdb")
    msm = EnhancedMSM([str(traj)], topology_file=str(pdb))
    with caplog.at_level(logging.INFO):
        msm.load_trajectories(stride=2, atom_selection="name CA", chunk_size=5)
    assert msm.trajectories and msm.trajectories[0].n_frames == 50
    assert msm.trajectories[0].n_atoms < 500  # reduced atom count
    assert any("Streaming trajectory" in rec.getMessage() for rec in caplog.records)
