"""Test script to verify molecular features work with the new format."""

import numpy as np
import mdtraj as md
from pmarlo.features.base import parse_feature_spec, get_feature

# Test parsing
print("Testing feature spec parsing...")
print("-" * 60)

test_specs = [
    "distance([0, 1])",
    "angle([0, 1, 2])",
    "dihedral([0, 1, 2, 3])",
]

for spec in test_specs:
    try:
        feat_name, kwargs = parse_feature_spec(spec)
        print(f"✓ {spec:30} -> {feat_name:15} with kwargs={kwargs}")
    except Exception as e:
        print(f"✗ {spec:30} -> ERROR: {e}")

print("\n" + "=" * 60)
print("Testing feature registry...")
print("-" * 60)

for spec in test_specs:
    try:
        feat_name, kwargs = parse_feature_spec(spec)
        fc = get_feature(feat_name)
        print(f"✓ {feat_name:15} is registered: {fc.name}")
    except Exception as e:
        print(f"✗ {feat_name:15} -> ERROR: {e}")

print("\n" + "=" * 60)
print("Testing feature computation with dummy trajectory...")
print("-" * 60)

# Create a minimal test trajectory
try:
    import tempfile
    import os

    pdb_content = """ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00  0.00           C
ATOM      4  O   ALA A   1       3.227   1.420   0.000  1.00  0.00           O
ATOM      5  CB  ALA A   1       1.967  -0.729  -1.232  1.00  0.00           C
END
"""

    # Write to a temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
        f.write(pdb_content)
        temp_pdb = f.name

    # Create a simple trajectory with 10 frames
    traj = md.load_pdb(temp_pdb)

    # Add some random motion to create multiple frames
    n_frames = 10
    xyz_stack = []
    for i in range(n_frames):
        noise = np.random.randn(*traj.xyz[0].shape) * 0.01
        xyz_stack.append(traj.xyz[0] + noise)

    traj.xyz = np.array(xyz_stack)
    traj.time = np.arange(n_frames)

    print(f"Created test trajectory: {traj.n_frames} frames, {traj.n_atoms} atoms")

    # Test each feature
    for spec in test_specs:
        try:
            feat_name, kwargs = parse_feature_spec(spec)
            fc = get_feature(feat_name)
            X = fc.compute(traj, **kwargs)
            periodic = fc.is_periodic()
            print(f"✓ {spec:30} -> shape={X.shape}, periodic={periodic[0] if len(periodic) > 0 else 'N/A'}")
        except Exception as e:
            print(f"✗ {spec:30} -> ERROR: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("All tests completed!")

    # Clean up temp file
    if os.path.exists(temp_pdb):
        os.unlink(temp_pdb)

except Exception as e:
    print(f"ERROR creating test trajectory: {e}")
    import traceback
    traceback.print_exc()

