import mdtraj as md
import numpy as np
import pytest
from mdtraj.core.element import carbon

from pmarlo.features import get_feature
from pmarlo.features.base import parse_feature_spec
from pmarlo.features.builtins import ContactsPairFeature, DistancePairFeature


def _simple_traj(distance: float = 0.5, *, nan: bool = False) -> md.Trajectory:
    top = md.Topology()
    chain = top.add_chain()
    res = top.add_residue("GLY", chain)
    top.add_atom("A", carbon, res)
    top.add_atom("B", carbon, res)
    coords = np.array([[[0.0, 0.0, 0.0], [distance, 0.0, 0.0]]], dtype=float)
    if nan:
        coords[0, 1, :] = np.nan
    return md.Trajectory(coords, top)


def test_case_insensitive_feature_lookup() -> None:
    name, kwargs = parse_feature_spec("rg")
    feat = get_feature(name)
    traj = _simple_traj()
    X = feat.compute(traj, **kwargs)
    assert X.shape == (1, 1)


def test_distance_pair_validation_and_nan() -> None:
    traj = _simple_traj(nan=True)
    feat = DistancePairFeature()
    X = feat.compute(traj, i=0, j=1)
    assert np.all(np.isfinite(X))
    with pytest.raises(ValueError):
        feat.compute(traj, i=0, j=2)


def test_contacts_pair_boundary_and_validation() -> None:
    traj = _simple_traj(distance=0.5)
    feat = ContactsPairFeature()
    X = feat.compute(traj, i=0, j=1, rcut=0.5)
    assert X[0, 0] == 1.0
    traj_nan = _simple_traj(nan=True)
    X_nan = feat.compute(traj_nan, i=0, j=1, rcut=0.5)
    assert X_nan[0, 0] == 0.0
    with pytest.raises(ValueError):
        feat.compute(traj, i=0, j=2, rcut=0.5)
    with pytest.raises(ValueError):
        feat.compute(traj, i=0, j=1, rcut=-1.0)
