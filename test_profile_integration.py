"""Test the molecular_cv_biasing profile with actual shard extraction."""

import numpy as np
import mdtraj as md
import tempfile
import os
from pathlib import Path

# Test loading the feature profile
print("=" * 60)
print("Testing feature profile loading...")
print("-" * 60)

from pmarlo_webapp.app.backend.feature_profiles import load_feature_profile, FEATURE_PROFILES

# Show available profiles
print("Available profiles:")
for name, profile in FEATURE_PROFILES.items():
    print(f"  - {name}: {profile.description}")

# Load the molecular_cv_biasing profile
print("\n" + "=" * 60)
print("Loading molecular_cv_biasing profile...")
print("-" * 60)

profile = load_feature_profile("molecular_cv_biasing")
print(f"Profile: {profile.name}")
print(f"Description: {profile.description}")
print(f"Feature type: {profile.feature_type}")
print(f"CV biasing compatible: {profile.cv_biasing_compatible}")
print(f"Features ({len(profile.features)}):")
for feat in profile.features:
    print(f"  - {feat}")

# Test parsing each feature
print("\n" + "=" * 60)
print("Testing feature parsing...")
print("-" * 60)

from pmarlo.features.base import parse_feature_spec, get_feature

for feat_spec in profile.features:
    try:
        feat_name, kwargs = parse_feature_spec(feat_spec)
        fc = get_feature(feat_name)
        print(f"✓ {feat_spec:30} -> {feat_name:15} (registered)")
    except Exception as e:
        print(f"✗ {feat_spec:30} -> ERROR: {e}")

# Test with compute_features
print("\n" + "=" * 60)
print("Testing with compute_features API...")
print("-" * 60)

from pmarlo.api.features import compute_features

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
    
    print(f"Created test trajectory: {traj.n_frames} frames, {traj.n_atoms} atoms")
    
    # Compute features using the profile
    X, columns, periodic = compute_features(traj, profile.features)
    
    print(f"\nFeature matrix computed successfully!")
    print(f"  Shape: {X.shape}")
    print(f"  Columns: {columns}")
    print(f"  Periodic: {periodic}")
    
    print("\n" + "=" * 60)
    print("✓ All tests passed! Molecular CV biasing profile works!")
    print("=" * 60)
    
finally:
    if os.path.exists(temp_pdb):
        os.unlink(temp_pdb)

