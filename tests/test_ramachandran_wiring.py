import mdtraj as md
import numpy as np
from mdtraj.core.element import carbon

from pmarlo import api


def _tiny_traj():
    top = md.Topology()
    chain = top.add_chain()
    res0 = top.add_residue("ALA", chain)
    top.add_atom("C0", carbon, res0)
    top.add_atom("N0", carbon, res0)
    top.add_atom("CA0", carbon, res0)
    top.add_atom("C0b", carbon, res0)
    res1 = top.add_residue("ALA", chain)
    top.add_atom("N1", carbon, res1)
    top.add_atom("CA1", carbon, res1)
    top.add_atom("C1", carbon, res1)
    coords = np.zeros((1, top.n_atoms, 3), dtype=float)
    return md.Trajectory(coords, top)


def test_phi_psi_independent(monkeypatch):
    traj = _tiny_traj()

    def fake_phi(t):
        return np.deg2rad([[190.0]]), np.array([[0, 1, 2, 3]])

    def fake_psi(t):
        return np.deg2rad([[20.0]]), np.array([[1, 2, 3, 4]])

    monkeypatch.setattr(md, "compute_phi", fake_phi)
    monkeypatch.setattr(md, "compute_psi", fake_psi)

    X, cols, periodic = api.compute_features(traj, feature_specs=["phi_psi"])
    assert X.shape[1] == 2

    phi = X[:, 0]
    psi = X[:, 1]
    assert phi is not psi
    assert not np.allclose(phi, psi)

    phi_deg = np.degrees(phi)
    psi_deg = np.degrees(psi)
    assert np.all(phi_deg > -180) and np.all(phi_deg <= 180)
    assert np.all(psi_deg > -180) and np.all(psi_deg <= 180)
