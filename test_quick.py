"""Quick test of molecular features."""
import sys
print("Python version:", sys.version)

print("\n1. Testing feature import...")
try:
    from pmarlo.features import get_feature
    print("   ✓ Feature import OK")
except Exception as e:
    print(f"   ✗ Error: {e}")
    sys.exit(1)

print("\n2. Testing feature registration...")
try:
    for name in ['distance', 'angle', 'dihedral']:
        fc = get_feature(name)
        print(f"   ✓ {name} -> {fc.name}")
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n3. Testing feature parsing...")
try:
    from pmarlo.features.base import parse_feature_spec
    test_specs = ["distance([0, 1])", "angle([0, 1, 2])", "dihedral([0, 1, 2, 3])"]
    for spec in test_specs:
        name, kwargs = parse_feature_spec(spec)
        print(f"   ✓ {spec} -> {name} {kwargs}")
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n4. Testing profile loading...")
try:
    from pmarlo_webapp.app.backend.feature_profiles import load_feature_profile
    profile = load_feature_profile("molecular_cv_biasing")
    print(f"   ✓ Loaded profile: {profile.name}")
    print(f"   Features: {profile.features}")
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✓ All basic tests passed!")

