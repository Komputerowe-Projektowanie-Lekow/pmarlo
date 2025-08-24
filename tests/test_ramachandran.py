import mdtraj as md
import numpy as np
from mdtraj.core.element import carbon

from pmarlo.fes import compute_ramachandran, periodic_hist2d


def _ramachandran_traj() -> md.Trajectory:
    top = md.Topology()
    chain = top.add_chain()
    res0 = top.add_residue("ALA", chain)
    top.add_atom("C0", carbon, res0)
    res1 = top.add_residue("ALA", chain)
    top.add_atom("N1", carbon, res1)
    top.add_atom("CA1", carbon, res1)
    top.add_atom("C1", carbon, res1)
    res2 = top.add_residue("ALA", chain)
    top.add_atom("N2", carbon, res2)
    coords = np.zeros((1, 5, 3), dtype=float)
    return md.Trajectory(coords, top)


def test_angle_wrapping_invariance(monkeypatch) -> None:
    traj = _ramachandran_traj()

    def fake_phi(t):
        return np.deg2rad([[190.0]]), np.array([[0, 1, 2, 3]])

    def fake_psi(t):
        return np.deg2rad([[-190.0]]), np.array([[1, 2, 3, 4]])

    monkeypatch.setattr(md, "compute_phi", fake_phi)
    monkeypatch.setattr(md, "compute_psi", fake_psi)
    ang1 = compute_ramachandran(traj)

    def fake_phi2(t):
        return np.deg2rad([[-170.0]]), np.array([[0, 1, 2, 3]])

    def fake_psi2(t):
        return np.deg2rad([[170.0]]), np.array([[1, 2, 3, 4]])

    monkeypatch.setattr(md, "compute_phi", fake_phi2)
    monkeypatch.setattr(md, "compute_psi", fake_psi2)
    ang2 = compute_ramachandran(traj)

    np.testing.assert_allclose(ang1, ang2)


def test_periodic_histogram_mass_conservation() -> None:
    phi = np.array([179.0, -179.0])
    psi = np.array([10.0, 10.0])
    H, xedges, yedges = periodic_hist2d(phi, psi, bins=(10, 10))
    assert H.sum() == 2.0
    col = np.digitize([10.0], yedges) - 1
    assert H[0, col[0]] == 2.0
