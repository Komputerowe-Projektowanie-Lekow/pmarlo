from pathlib import Path
from unittest.mock import patch

import mdtraj as md
import numpy as np

from pmarlo.markov_state_model.markov_state_model import EnhancedMSM


def _write_synthetic_traj(tmpdir: Path) -> tuple[Path, Path]:
    np.random.seed(0)
    top = md.Topology()
    chain = top.add_chain()
    res = top.add_residue("ALA", chain)
    for _ in range(3):
        top.add_atom("C", element=md.element.carbon, residue=res)
    xyz = np.random.rand(5, top.n_atoms, 3)
    traj = md.Trajectory(xyz, top)
    dcd = tmpdir / "test.dcd"
    pdb = tmpdir / "test.pdb"
    traj.save_dcd(str(dcd))
    traj[0].save_pdb(str(pdb))
    return dcd, pdb


def test_load_trajectories_uses_iterload(tmp_path: Path) -> None:
    dcd, pdb = _write_synthetic_traj(tmp_path)
    msm = EnhancedMSM(trajectory_files=[str(dcd)], topology_file=str(pdb))
    original = md.iterload
    with patch("mdtraj.iterload") as mock_iter:
        mock_iter.side_effect = original
        msm.load_trajectories(stride=2, atom_indices=[0, 1])
        assert mock_iter.called
    assert msm.trajectories[0].n_frames == 3
    assert msm.trajectories[0].n_atoms == 2
