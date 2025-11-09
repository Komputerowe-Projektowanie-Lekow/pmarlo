"""Test model file discovery for CV export."""

from pathlib import Path

print("=" * 80)
print("TEST: Model File Discovery for CV Export")
print("=" * 80)

# Simulate the directory structure from the user's case
models_dir = Path("pmarlo_webapp/app_output/models")
checkpoint_dir = models_dir / "training-20251108-193156"

print(f"\nCheckpoint directory: {checkpoint_dir}")
print(f"Models directory: {models_dir}")

# Extract checkpoint timestamp
checkpoint_timestamp = checkpoint_dir.name.replace("training-", "")
print(f"Checkpoint timestamp: {checkpoint_timestamp}")

# Find all .pt files
all_pt_files = sorted(models_dir.glob("deeptica-*.pt"))
print(f"\nAll .pt files found: {len(all_pt_files)}")
for mf in all_pt_files:
    is_scaler = mf.name.endswith(".scaler.pt")
    print(f"  - {mf.name} {'(scaler - skip)' if is_scaler else ''}")

# Filter out .scaler.pt files
all_model_files = [f for f in all_pt_files if not f.name.endswith(".scaler.pt")]
print(f"\nFiltered model files (excluding .scaler.pt): {len(all_model_files)}")
for mf in all_model_files:
    print(f"  - {mf.name}")

# Find matching models (timestamp >= checkpoint timestamp)
matching_models = []
for mf in all_model_files:
    model_timestamp = mf.stem.replace("deeptica-", "")
    matches = model_timestamp >= checkpoint_timestamp
    print(f"\n  {mf.name}")
    print(f"    Model timestamp: {model_timestamp}")
    print(f"    >= {checkpoint_timestamp}? {matches}")
    if matches:
        matching_models.append(mf)

print(f"\nMatching models: {len(matching_models)}")
for mf in matching_models:
    model_timestamp = mf.stem.replace("deeptica-", "")
    print(f"  ✓ {mf.name} (timestamp: {model_timestamp})")

if matching_models:
    selected_model = matching_models[0]
    print(f"\n✓ Selected model: {selected_model.name}")

    # Check if companion files exist
    base_path = selected_model.with_suffix("")
    pt_file = base_path.with_suffix(".pt")
    scaler_file = base_path.with_suffix(".scaler.pt")
    json_file = base_path.with_suffix(".json")

    print(f"\nChecking companion files:")
    print(f"  .pt file:      {'✓' if pt_file.exists() else '✗'} {pt_file.name}")
    print(f"  .scaler.pt:    {'✓' if scaler_file.exists() else '✗'} {scaler_file.name}")
    print(f"  .json file:    {'✓' if json_file.exists() else '✗'} {json_file.name}")

    all_exist = pt_file.exists() and scaler_file.exists() and json_file.exists()
    if all_exist:
        print(f"\n✅ All required files found! Model can be loaded.")
    else:
        print(f"\n❌ Some required files are missing!")
else:
    print(f"\n❌ No matching models found!")

print("\n" + "=" * 80)
print("Expected behavior:")
print("  1. Checkpoint: training-20251108-193156")
print("  2. Model saved at completion: deeptica-20251108-195911.pt")
print("  3. Logic: 195911 >= 193156 → Match! ✓")
print("=" * 80)

