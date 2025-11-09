"""
Diagnostic script to check openmm-torch availability.
Run this from your (ommtorch) environment to see what's happening.
"""
import sys
import os
from pathlib import Path

print("=" * 80)
print("Python Environment Diagnostics")
print("=" * 80)
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")
print(f"sys.prefix: {sys.prefix}")
print()

print("=" * 80)
print("Environment Variables")
print("=" * 80)
for var in ("CONDA_PREFIX", "CONDA_DEFAULT_ENV", "MAMBA_ROOT_PREFIX",
            "MICROMAMBA_ROOT_PREFIX", "OPENMM_PLUGIN_DIR", "PMARLO_OPENMM_PLUGIN_DIR"):
    value = os.environ.get(var, "<not set>")
    print(f"{var}: {value}")
print()

print("=" * 80)
print("Checking imports")
print("=" * 80)

# Check OpenMM
try:
    import openmm
    print(f"✓ OpenMM installed: {openmm.__version__}")
    print(f"  OpenMM location: {openmm.__file__}")
except ImportError as e:
    print(f"✗ OpenMM not available: {e}")
    openmm = None

# Check openmmtorch direct import
try:
    import openmmtorch
    print(f"✓ openmmtorch module found")
    print(f"  openmmtorch location: {openmmtorch.__file__}")
    try:
        from openmmtorch import TorchForce
        print(f"✓ TorchForce imported directly from openmmtorch")
    except ImportError as e:
        print(f"✗ Cannot import TorchForce from openmmtorch: {e}")
except ImportError as e:
    print(f"✗ openmmtorch module not found: {e}")

# Check PyTorch
try:
    import torch
    print(f"✓ PyTorch installed: {torch.__version__}")
    print(f"  PyTorch location: {torch.__file__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
except ImportError as e:
    print(f"✗ PyTorch not available: {e}")

print()
print("=" * 80)
print("Checking OpenMM Plugin Directories")
print("=" * 80)

if openmm:
    from openmm import Platform

    try:
        default_dir = Platform.getDefaultPluginsDirectory()
        print(f"Default plugins directory: {default_dir}")
        if default_dir:
            default_path = Path(default_dir)
            if default_path.exists():
                print(f"  Directory exists: ✓")
                plugins = list(default_path.glob("*Torch*"))
                if plugins:
                    print(f"  Torch plugins found:")
                    for p in plugins:
                        print(f"    - {p.name}")
                else:
                    print(f"  No Torch plugins found in directory")
                    print(f"  All files:")
                    for p in default_path.iterdir():
                        print(f"    - {p.name}")
            else:
                print(f"  Directory does not exist: ✗")
    except Exception as e:
        print(f"Error getting default plugins directory: {e}")

    # Check if TorchForce is in openmm namespace
    print()
    print("Checking openmm namespace for TorchForce:")
    torch_force = getattr(openmm, "TorchForce", None)
    if torch_force:
        print(f"✓ openmm.TorchForce found: {torch_force}")
    else:
        print(f"✗ openmm.TorchForce not found")
        torch_force_nested = getattr(getattr(openmm, "openmm", object()), "TorchForce", None)
        if torch_force_nested:
            print(f"✓ openmm.openmm.TorchForce found: {torch_force_nested}")
        else:
            print(f"✗ openmm.openmm.TorchForce not found")

print()
print("=" * 80)
print("Testing pmarlo's check_openmm_torch_available()")
print("=" * 80)

try:
    from pmarlo.features.deeptica import check_openmm_torch_available
    result = check_openmm_torch_available()
    if result:
        print(f"✓ check_openmm_torch_available() returned True")
    else:
        print(f"✗ check_openmm_torch_available() returned False")
except Exception as e:
    print(f"✗ Error checking openmm-torch availability: {e}")
    import traceback
    traceback.print_exc()

print()
print("=" * 80)
print("Potential conda environment paths")
print("=" * 80)

# Check common conda locations
conda_prefix = os.environ.get("CONDA_PREFIX")
if conda_prefix:
    conda_path = Path(conda_prefix)
    plugin_locations = [
        conda_path / "Library" / "plugins",
        conda_path / "lib" / "plugins",
        conda_path / "lib64" / "plugins",
    ]

    for loc in plugin_locations:
        if loc.exists():
            print(f"✓ {loc} exists")
            torch_plugins = list(loc.glob("*Torch*"))
            if torch_plugins:
                print(f"  Torch plugins:")
                for p in torch_plugins:
                    print(f"    - {p.name}")
        else:
            print(f"✗ {loc} does not exist")

