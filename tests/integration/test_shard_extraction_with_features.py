"""Test shard extraction with molecular_cv_biasing profile."""

import numpy as np
import mdtraj as md
import tempfile
import json
import pytest
from pathlib import Path


def test_shard_extraction_with_molecular_features():
    """Test complete shard extraction workflow with molecular features."""
    from pmarlo_webapp.app.backend.shard_extraction import extract_shards_with_features
    from pmarlo_webapp.app.backend.feature_profiles import load_feature_profile

    # Load the molecular_cv_biasing profile
    profile = load_feature_profile("molecular_cv_biasing")

    # Create a test PDB file
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

    # Create temporary files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)

        # Write PDB
        pdb_file = temp_dir_path / "test.pdb"
        with open(pdb_file, 'w') as f:
            f.write(pdb_content)

        # Create a trajectory with 50 frames
        traj = md.load_pdb(str(pdb_file))

        n_frames = 50
        xyz_stack = []
        for i in range(n_frames):
            noise = np.random.randn(*traj.xyz[0].shape) * 0.01
            xyz_stack.append(traj.xyz[0] + noise)

        traj.xyz = np.array(xyz_stack)
        traj.time = np.arange(n_frames) * 0.002  # 2 ps per frame

        # Save trajectory as DCD
        dcd_file = temp_dir_path / "test.dcd"
        traj.save_dcd(str(dcd_file))

        # Create output directory for shards
        shard_dir = temp_dir_path / "shards"
        shard_dir.mkdir()

        # Extract shards
        shard_paths = extract_shards_with_features(
            pdb_file=pdb_file,
            traj_files=[dcd_file],
            out_dir=shard_dir,
            feature_specs=profile.features,
            stride=1,
            temperature=300.0,
            seed_start=0,
            frames_per_shard=20,
            provenance={
                "feature_profile": profile.name,
                "description": profile.description,
            }
        )

        assert len(shard_paths) > 0, "Should create at least one shard"

        # Inspect the first shard
        first_shard = shard_paths[0]

        with open(first_shard, 'r') as f:
            shard_data = json.load(f)

        assert "shard_id" in shard_data
        assert "source" in shard_data

        source = shard_data["source"]
        assert "temperature_K" in source
        assert source["temperature_K"] == 300.0
        assert "n_frames" in source
        assert "columns" in source
        assert "periodic" in source

        # Check if periodic flags are correct
        periodic_dict = source.get("periodic", {})
        for col, is_periodic in periodic_dict.items():
            expected_periodic = 'angle' in col or 'dihedral' in col
            assert is_periodic == expected_periodic, \
                f"Periodic flag mismatch for {col}: expected {expected_periodic}, got {is_periodic}"

        # Load the NPZ file
        npz_file = first_shard.parent / (first_shard.stem + ".npz")
        assert npz_file.exists(), "NPZ file should exist"

        npz_data = np.load(npz_file)
        assert "X" in npz_data
        assert npz_data["X"].shape[1] == len(profile.features)
