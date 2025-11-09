import pickle
from pathlib import Path

bundle_path = Path(r"C:\Users\konrad_guest\Documents\GitHub\pmarlo\pmarlo_webapp\app_output\models\deeptica-20251108-173316.pbz")

print(f"Checking bundle: {bundle_path}")
print(f"Exists: {bundle_path.exists()}")
print(f"Size: {bundle_path.stat().st_size if bundle_path.exists() else 'N/A'}")

try:
    with open(bundle_path, 'rb') as f:
        data = pickle.load(f)

    print(f"\nBundle keys: {list(data.keys())}")

    network = data.get('network')
    scaler = data.get('scaler')

    print(f"\nNetwork type: {type(network)}")
    print(f"Network is None: {network is None}")

    print(f"\nScaler type: {type(scaler)}")
    print(f"Scaler is None: {scaler is None}")

    if scaler is not None:
        print(f"Scaler has mean_: {hasattr(scaler, 'mean_')}")
        print(f"Scaler has scale_: {hasattr(scaler, 'scale_')}")
        if hasattr(scaler, 'mean_'):
            print(f"Scaler mean_ shape: {scaler.mean_.shape if hasattr(scaler.mean_, 'shape') else 'N/A'}")

    if network is not None:
        print(f"Network class: {network.__class__.__name__}")

except Exception as e:
    print(f"\nError loading bundle: {e}")
    import traceback
    traceback.print_exc()

