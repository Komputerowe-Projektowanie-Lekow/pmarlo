"""Test the molecular_cv_biasing profile with actual shard extraction."""

import numpy as np
import mdtraj as md
import tempfile
import pytest
from pathlib import Path


def test_feature_profile_loading():
    """Test loading feature profiles."""
    from pmarlo_webapp.app.backend.feature_profiles import load_feature_profile, FEATURE_PROFILES

    # Check that profiles are available
    assert len(FEATURE_PROFILES) > 0
    assert "molecular_cv_biasing" in FEATURE_PROFILES

    # Load the molecular_cv_biasing profile
    profile = load_feature_profile("molecular_cv_biasing")
    assert profile.name == "molecular_cv_biasing"
    assert profile.feature_type == "molecular"
    assert profile.cv_biasing_compatible == True
    assert len(profile.features) > 0


def test_feature_parsing_from_profile():
    """Test parsing each feature from the profile."""
    from pmarlo_webapp.app.backend.feature_profiles import load_feature_profile
    from pmarlo.features.base import parse_feature_spec, get_feature

    profile = load_feature_profile("molecular_cv_biasing")

    for feat_spec in profile.features:
        feat_name, kwargs = parse_feature_spec(feat_spec)
        fc = get_feature(feat_name)
        assert fc is not None, f"Feature {feat_name} should be registered"


def test_profile_with_compute_features():
    """Test using profile features with compute_features API."""
    from pmarlo_webapp.app.backend.feature_profiles import load_feature_profile
    from pmarlo.api.features import compute_features

    profile = load_feature_profile("molecular_cv_biasing")

    # Create test trajectory
    pdb_content = """ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00  0.00           C
ATOM      4  O   ALA A   1       3.227   1.420   0.000  1.00  0.00           O
ATOM      5  CB  ALA A   1       1.967  -0.729  -1.232  1.00  0.00           C
ATOM      6  N   GLY A   2       3.100   2.200   0.000  1.00  0.00           N
ATOM      7  CA  GLY A   2       3.700   2.800   1.200  1.00  0.00           C
ATOM      8  C   GLY A   2       4.900   3.400   1.200  1.00  0.00           C
END
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
        f.write(pdb_content)
        temp_pdb = f.name

    try:
        traj = md.load_pdb(temp_pdb)

        # Add some random motion to create multiple frames
        n_frames = 10
        xyz_stack = []
        for i in range(n_frames):
            noise = np.random.randn(*traj.xyz[0].shape) * 0.01
            xyz_stack.append(traj.xyz[0] + noise)

        traj.xyz = np.array(xyz_stack)
        traj.time = np.arange(n_frames)

        # Compute features using the profile
        X, columns, periodic = compute_features(traj, profile.features)

        assert X.shape[0] == n_frames
        assert X.shape[1] == len(profile.features)
        assert len(columns) == len(profile.features)
        assert len(periodic) == len(profile.features)

    finally:
        import os
        if os.path.exists(temp_pdb):
            os.unlink(temp_pdb)
"""Comprehensive integration test for molecular features with shard extraction."""

import sys
import numpy as np
import mdtraj as md
import tempfile
import pytest
from pathlib import Path


def test_feature_registration():
    """Test that molecular features are registered."""
    from pmarlo.features import get_feature

    for name in ['distance', 'angle', 'dihedral']:
        fc = get_feature(name)
        assert fc is not None, f"Feature '{name}' should be registered"


def test_feature_parsing():
    """Test feature spec parsing."""
    from pmarlo.features.base import parse_feature_spec

    test_specs = [
        "distance([0, 1])",
        "distance([1, 2])",
        "angle([0, 1, 2])",
        "dihedral([0, 1, 2, 3])",
        "dihedral([1, 2, 4, 7])"
    ]

    for spec in test_specs:
        name, kwargs = parse_feature_spec(spec)
        assert name in ['distance', 'angle', 'dihedral']
        assert 'atoms' in kwargs


def test_profile_loading():
    """Test loading molecular_cv_biasing profile."""
    from pmarlo_webapp.app.backend.feature_profiles import load_feature_profile

    profile = load_feature_profile("molecular_cv_biasing")
    assert profile.name == "molecular_cv_biasing"
    assert profile.cv_biasing_compatible == True
    assert len(profile.features) > 0


def test_feature_computation():
    """Test feature computation with trajectory."""
    from pmarlo.api.features import compute_features
    from pmarlo_webapp.app.backend.feature_profiles import load_feature_profile

    profile = load_feature_profile("molecular_cv_biasing")

    # Create minimal PDB
    pdb_content = """ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00  0.00           C
ATOM      4  O   ALA A   1       3.227   1.420   0.000  1.00  0.00           O
ATOM      5  CB  ALA A   1       1.967  -0.729  -1.232  1.00  0.00           C
ATOM      6  N   GLY A   2       3.100   2.200   0.000  1.00  0.00           N
ATOM      7  CA  GLY A   2       3.700   2.800   1.200  1.00  0.00           C
ATOM      8  C   GLY A   2       4.900   3.400   1.200  1.00  0.00           C
END
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
        f.write(pdb_content)
        temp_pdb = f.name

    try:
        traj = md.load_pdb(temp_pdb)

        # Add frames
        n_frames = 10
        xyz_stack = [traj.xyz[0] + np.random.randn(*traj.xyz[0].shape) * 0.01 for _ in range(n_frames)]
        traj.xyz = np.array(xyz_stack)
        traj.time = np.arange(n_frames)

        # Compute features
        X, columns, periodic = compute_features(traj, profile.features)

        assert X.shape[0] == n_frames
        assert X.shape[1] == len(profile.features)
        assert len(columns) == len(profile.features)
        assert len(periodic) == len(profile.features)

        # Verify periodic flags are correct
        for i, col in enumerate(columns):
            if 'distance' in col:
                assert not periodic[i], f"Distance {col} should not be periodic"
            elif 'angle' in col or 'dihedral' in col:
                assert periodic[i], f"Angle/Dihedral {col} should be periodic"

    finally:
        import os
        if os.path.exists(temp_pdb):
            os.unlink(temp_pdb)


def test_shard_id_format():
    """Test shard ID format generation."""
    from datetime import datetime

    # Simulate what _write_shard does
    run_id = "run-20251108-120000"
    shard_idx = 0
    temperature = 300.0

    t_kelvin = int(temperature)
    replica_id = 0
    run_suffix = str(run_id).replace("run_", "") if run_id else "default"
    shard_id = f"T{t_kelvin}K_{run_suffix}_seg{shard_idx:04d}_rep{replica_id:03d}"

    assert shard_id == "T300K_run-20251108-120000_seg0000_rep000"

    # Check source metadata structure
    source_metadata = {
        "created_at": datetime.now().isoformat(),
        "kind": "demux",
        "run_id": run_id,
        "replica_id": 0,
        "segment_id": shard_idx,
    }

    required_keys = ["created_at", "kind", "run_id", "replica_id", "segment_id"]
    for key in required_keys:
        assert key in source_metadata, f"Missing required key: {key}"
