"""Test molecular features parsing and computation."""

from __future__ import annotations

import os
import tempfile

import mdtraj as md
import numpy as np


def test_feature_spec_parsing():
    """Test that feature specs are parsed correctly."""
    from pmarlo.features.base import parse_feature_spec

    test_specs = [
        ("distance([0, 1])", "distance", {"indices": [0, 1], "atoms": [0, 1]}),
        ("angle([0, 1, 2])", "angle", {"indices": [0, 1, 2], "atoms": [0, 1, 2]}),
        (
            "dihedral([0, 1, 2, 3])",
            "dihedral",
            {"indices": [0, 1, 2, 3], "atoms": [0, 1, 2, 3]},
        ),
    ]

    for spec, expected_name, expected_kwargs in test_specs:
        feat_name, kwargs = parse_feature_spec(spec)
        assert feat_name == expected_name, f"Feature name mismatch for {spec}"
        assert kwargs == expected_kwargs, f"Kwargs mismatch for {spec}"


def test_feature_registration():
    """Test that molecular features are registered."""
    from pmarlo.features import get_feature

    for name in ["distance", "angle", "dihedral"]:
        fc = get_feature(name)
        assert fc is not None, f"Feature '{name}' not found"
        assert fc.name == name


def test_feature_computation():
    """Test feature computation with a dummy trajectory."""
    from pmarlo.features.base import get_feature, parse_feature_spec

    pdb_content = """ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00  0.00           C
ATOM      4  O   ALA A   1       3.227   1.420   0.000  1.00  0.00           O
ATOM      5  CB  ALA A   1       1.967  -0.729  -1.232  1.00  0.00           C
END
"""

    temp_pdb: str | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False) as f:
            f.write(pdb_content)
            temp_pdb = f.name

        traj = md.load_pdb(temp_pdb)

        n_frames = 10
        xyz_stack = []
        for _ in range(n_frames):
            noise = np.random.randn(*traj.xyz[0].shape) * 0.01
            xyz_stack.append(traj.xyz[0] + noise)

        traj.xyz = np.array(xyz_stack)
        traj.time = np.arange(n_frames)

        test_specs = [
            "distance([0, 1])",
            "angle([0, 1, 2])",
            "dihedral([0, 1, 2, 3])",
        ]

        for spec in test_specs:
            feat_name, kwargs = parse_feature_spec(spec)
            fc = get_feature(feat_name)
            X = fc.compute(traj, **kwargs)
            periodic = fc.is_periodic()

            assert (
                X.shape[0] == n_frames
            ), f"Feature {spec} should return {n_frames} values"
            assert len(periodic) > 0, f"Feature {spec} should return periodic info"
    finally:
        if temp_pdb and os.path.exists(temp_pdb):
            os.unlink(temp_pdb)


def test_periodic_flags():
    """Test that periodic flags are set correctly for different feature types."""
    from pmarlo.features import get_feature

    distance_fc = get_feature("distance")
    assert not distance_fc.is_periodic()[0], "Distance should not be periodic"

    angle_fc = get_feature("angle")
    assert angle_fc.is_periodic()[0], "Angle should be periodic"

    dihedral_fc = get_feature("dihedral")
    assert dihedral_fc.is_periodic()[0], "Dihedral should be periodic"
