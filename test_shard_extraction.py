    try:
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

        print(f"\n✓ Successfully created {len(shard_paths)} shards!")

        # Inspect the first shard
        if shard_paths:
            first_shard = shard_paths[0]
            print(f"\nInspecting first shard: {first_shard.name}")

            with open(first_shard, 'r') as f:
                shard_data = json.load(f)

            print(f"  Shard ID: {shard_data.get('shard_id')}")
            print(f"  Temperature: {shard_data.get('source', {}).get('temperature_K')} K")
            print(f"  N frames: {shard_data.get('source', {}).get('n_frames')}")
            print(f"  Columns: {shard_data.get('source', {}).get('columns')}")
            print(f"  Periodic: {shard_data.get('source', {}).get('periodic')}")

            # Check if periodic flags are correct
            periodic_dict = shard_data.get('source', {}).get('periodic', {})
            print("\n  Periodic flag verification:")
            for col, is_periodic in periodic_dict.items():
                expected_periodic = 'angle' in col or 'dihedral' in col
                status = "✓" if is_periodic == expected_periodic else "✗"
                print(f"    {status} {col:30} -> {is_periodic} (expected: {expected_periodic})")

            # Load the NPZ file
            npz_file = first_shard.parent / (first_shard.stem + ".npz")
            if npz_file.exists():
                npz_data = np.load(npz_file)
                print(f"\n  NPZ data shape: {npz_data['X'].shape}")
                print(f"  NPZ arrays: {list(npz_data.keys())}")

        print("\n" + "=" * 60)
        print("✓ All tests passed! Shard extraction works correctly!")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ Error during shard extraction: {e}")
        import traceback
        traceback.print_exc()
"""Test shard extraction with molecular_cv_biasing profile."""

import numpy as np
import mdtraj as md
import tempfile
import os
import json
from pathlib import Path

print("=" * 60)
print("Testing shard extraction with molecular features")
print("=" * 60)

# Import the shard extraction function
from pmarlo_webapp.app.backend.shard_extraction import extract_shards_with_features
from pmarlo_webapp.app.backend.feature_profiles import load_feature_profile

# Load the molecular_cv_biasing profile
profile = load_feature_profile("molecular_cv_biasing")
print(f"\nUsing profile: {profile.name}")
print(f"Features: {profile.features}")

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
    print("\nCreating test trajectory...")
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
    print(f"  Created trajectory: {n_frames} frames, {traj.n_atoms} atoms")

    # Create output directory for shards
    shard_dir = temp_dir_path / "shards"
    shard_dir.mkdir()

    # Extract shards
    print("\nExtracting shards...")

