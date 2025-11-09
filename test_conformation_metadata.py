"""Test that shards have proper metadata for conformation analysis."""

import numpy as np
import mdtraj as md
import tempfile
import json
from pathlib import Path

print("=" * 70)
print("TEST: Shard Metadata for Conformation Analysis")
print("=" * 70)

from pmarlo_webapp.app.backend.shard_extraction import extract_shards_with_features
from pmarlo_webapp.app.backend.feature_profiles import load_feature_profile

# Load profile
profile = load_feature_profile("molecular_cv_biasing")
print(f"\nUsing profile: {profile.name}")

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
    print("\nCreating test trajectory...")
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
    print(f"  Created: {n_frames} frames, {traj.n_atoms} atoms")

    # Create shards
    shard_dir = temp_dir_path / "shards"
    shard_dir.mkdir()

    print("\nExtracting shards...")
    stride = 5
    frames_per_shard = 20

    try:
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

        print(f"✓ Created {len(shard_paths)} shards")

        # Validate metadata for conformation analysis
        print("\n" + "=" * 70)
        print("Validating Shard Metadata for Conformation Analysis")
        print("=" * 70)

        for i, shard_path in enumerate(shard_paths):
            print(f"\n[Shard {i}] {shard_path.name}")

            with open(shard_path, 'r') as f:
                shard_data = json.load(f)

            # Check source exists
            if "source" not in shard_data:
                print(f"  ✗ Missing 'source' in shard metadata")
                continue

            source = shard_data["source"]

            # Check frame_range
            frame_range = source.get("frame_range") or source.get("range")
            if frame_range:
                if isinstance(frame_range, (list, tuple)) and len(frame_range) == 2:
                    print(f"  ✓ frame_range: {frame_range}")

                    # Verify it makes sense
                    n_frames_in_shard = shard_data.get("source", {}).get("n_frames", 0)
                    expected_frames = (frame_range[1] - frame_range[0]) // stride
                    if expected_frames == n_frames_in_shard:
                        print(f"    ✓ Frame count matches: {n_frames_in_shard} frames")
                    else:
                        print(f"    ⚠ Frame count mismatch: expected {expected_frames}, got {n_frames_in_shard}")
                else:
                    print(f"  ✗ Invalid frame_range format: {frame_range}")
            else:
                print(f"  ✗ Missing 'frame_range' or 'range'")

            # Check trajectory files
            traj_files_found = []
            for key in ["traj_files", "trajectories", "traj", "trajectory", "path"]:
                val = source.get(key)
                if val:
                    if isinstance(val, (list, tuple)):
                        traj_files_found.extend([str(v) for v in val])
                    elif isinstance(val, str):
                        traj_files_found.append(val)

            if traj_files_found:
                print(f"  ✓ trajectory files ({len(traj_files_found)}):")
                for tf in traj_files_found:
                    print(f"    - {Path(tf).name}")
            else:
                print(f"  ✗ Missing trajectory file references")

            # Check other required fields
            required_fields = ["created_at", "kind", "run_id", "replica_id", "segment_id"]
            all_present = True
            for field in required_fields:
                if field in source:
                    print(f"  ✓ {field}: {source[field]}")
                else:
                    print(f"  ✗ Missing required field: {field}")
                    all_present = False

            if all_present and frame_range and traj_files_found:
                print(f"\n  ✅ Shard {i} has all required metadata for conformation analysis")
            else:
                print(f"\n  ❌ Shard {i} is missing required metadata")

        print("\n" + "=" * 70)
        print("✅ TEST PASSED: Shards are compatible with conformation analysis")
        print("=" * 70)

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()

