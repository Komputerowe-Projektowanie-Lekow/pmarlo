"""Test that shards have proper metadata for conformation analysis."""

import numpy as np
import mdtraj as md
import tempfile
import json
import pytest
from pathlib import Path


def test_shard_metadata_for_conformation_analysis():
    """Test that extracted shards contain metadata required for conformation analysis."""
    from pmarlo_webapp.app.backend.shard_extraction import extract_shards_with_features
    from pmarlo_webapp.app.backend.feature_profiles import load_feature_profile

    # Load profile
    profile = load_feature_profile("molecular_cv_biasing")

    # Create test data
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

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)

        # Write PDB
        pdb_file = temp_dir_path / "test.pdb"
        with open(pdb_file, 'w') as f:
            f.write(pdb_content)

        # Create trajectory
        traj = md.load_pdb(str(pdb_file))

        n_frames = 50
        xyz_stack = []
        for i in range(n_frames):
            noise = np.random.randn(*traj.xyz[0].shape) * 0.01
            xyz_stack.append(traj.xyz[0] + noise)

        traj.xyz = np.array(xyz_stack)
        traj.time = np.arange(n_frames) * 0.002

        # Save trajectory
        dcd_file = temp_dir_path / "test.dcd"
        traj.save_dcd(str(dcd_file))

        # Create shards
        shard_dir = temp_dir_path / "shards"
        shard_dir.mkdir()

        stride = 5
        frames_per_shard = 20

        shard_paths = extract_shards_with_features(
            pdb_file=pdb_file,
            traj_files=[dcd_file],
            out_dir=shard_dir,
            feature_specs=profile.features,
            stride=stride,
            temperature=300.0,
            seed_start=0,
            frames_per_shard=frames_per_shard,
        )

        assert len(shard_paths) > 0, "Should create at least one shard"

        # Validate metadata for conformation analysis
        for shard_path in shard_paths:
            with open(shard_path, 'r') as f:
                shard_data = json.load(f)

            # Check source exists
            assert "source" in shard_data, "Shard must have 'source' metadata"
            source = shard_data["source"]

            # Check frame_range
            frame_range = source.get("frame_range") or source.get("range")
            assert frame_range is not None, "Shard must have frame_range or range"
            assert isinstance(frame_range, (list, tuple)), "Frame range must be list or tuple"
            assert len(frame_range) == 2, "Frame range must have start and stop"

            # Verify frame count makes sense
            n_frames_in_shard = source.get("n_frames", 0)
            expected_frames = (frame_range[1] - frame_range[0]) // stride
            assert expected_frames == n_frames_in_shard, \
                f"Frame count mismatch: expected {expected_frames}, got {n_frames_in_shard}"

            # Check trajectory files
            traj_files_found = []
            for key in ["traj_files", "trajectories", "traj", "trajectory", "path"]:
                val = source.get(key)
                if val:
                    if isinstance(val, (list, tuple)):
                        traj_files_found.extend([str(v) for v in val])
                    elif isinstance(val, str):
                        traj_files_found.append(val)

            assert len(traj_files_found) > 0, "Shard must reference trajectory files"

            # Check other required fields
            required_fields = ["created_at", "kind", "run_id", "replica_id", "segment_id"]
            for field in required_fields:
                assert field in source, f"Missing required field: {field}"
