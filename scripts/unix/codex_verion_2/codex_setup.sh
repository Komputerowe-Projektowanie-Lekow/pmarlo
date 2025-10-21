#!/usr/bin/env bash
set -euxo pipefail

echo "=========================================="
echo "PMARLO Codex Environment Setup2"
echo "=========================================="

# 0) Install OS dependencies
echo "[Step 0/8] Installing OS dependencies..."
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
    python3-venv \
    python3-dev \
    build-essential \
    git \
    ca-certificates \
    wget \
    libssl-dev \
    libffi-dev

# 1) Create and activate Python virtual environment
echo "[Step 1/8] Creating Python virtual environment..."
VENV=/workspace/.venv
if [ -d "$VENV" ]; then
    echo "  Virtual environment already exists, removing..."
    rm -rf "$VENV"
fi
python3 -m venv "$VENV"
source "$VENV/bin/activate"

# 2) Upgrade pip, setuptools, and wheel
echo "[Step 2/8] Upgrading pip, setuptools, and wheel..."
python -m pip install --upgrade pip setuptools wheel

# 3) Install Poetry
echo "[Step 3/8] Installing Poetry..."
python -m pip install "poetry>=2.1.3,<3.0"

# 4) Configure repository
echo "[Step 4/8] Configuring repository..."
cd /workspace/pmarlo
git remote get-url origin >/dev/null 2>&1 || git remote add origin https://github.com/Komputerowe-Projektowanie-Lekow/pmarlo.git
git fetch origin --tags --prune
git checkout -B development origin/development

# Ensure Poetry uses THIS venv for this project
poetry config --local virtualenvs.create false

# 5) Install CPU-only PyTorch first (lightning comes from PyPI later)
echo "[Step 5/8] Installing CPU-only PyTorch..."
python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.2,<3.0"

# 6) Install project dependencies with pip (better for restricted networks)
echo "[Step 6/8] Installing project dependencies..."
PYTHON_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "  Detected Python version: $PYTHON_VERSION"

# Install build dependencies
echo "  Installing build system requirements..."
python -m pip install --upgrade "setuptools>=69" "wheel" "setuptools_scm[toml]>=8.0"

# Install the project with pip (reads pyproject.toml directly)
echo "  Installing pmarlo with all dependencies..."
if python -c "import sys; exit(0 if sys.version_info < (3, 12) else 1)"; then
    echo "  Installing with fixer extra (includes pdbfixer)..."
    python -m pip install -e ".[mlcv,app,analysis,fixer]"
else
    echo "  Installing with fixer extra (pdbfixer will be skipped on Python ${PYTHON_VERSION})..."
    python -m pip install -e ".[mlcv,app,analysis,fixer]"
fi

# Install dev and test dependencies (not in pyproject.toml extras)
echo "  Installing dev and test dependencies..."
python -m pip install \
    "pytest>=8.2" \
    "pytest-xdist>=3.5" \
    "pytest-randomly>=3.15" \
    "pytest-testmon>=2.1" \
    "pytest-picked>=0.5" \
    "tox>=4.14" \
    "pre-commit>=3.7" \
    "mypy>=1.11" \
    "hypothesis>=6.112" \
    "filelock>=3.20"

echo "  ✓ All dependencies installed successfully"

# 7) Re-pin CPU PyTorch to ensure no CUDA builds were pulled
echo "[Step 7/8] Re-pinning CPU-only PyTorch..."
python -m pip install --force-reinstall --index-url https://download.pytorch.org/whl/cpu "torch>=2.2,<3.0"

# 8) Run comprehensive smoke tests
echo "[Step 8/9] Running smoke tests..."
python - <<'PY'
import sys

# Track failures
failures = []

# Test core dependencies
print("\n=== Core Dependencies ===")
try:
    import numpy as np
    print(f"✓ numpy: {np.__version__}")
except ImportError as e:
    print(f"✗ numpy: FAILED - {e}")
    failures.append("numpy")

try:
    import scipy
    print(f"✓ scipy: {scipy.__version__}")
except ImportError as e:
    print(f"✗ scipy: FAILED - {e}")
    failures.append("scipy")

try:
    import pandas as pd
    print(f"✓ pandas: {pd.__version__}")
except ImportError as e:
    print(f"✗ pandas: FAILED - {e}")
    failures.append("pandas")

try:
    import openmm
    print(f"✓ openmm: {openmm.__version__}")
except ImportError as e:
    print(f"✗ openmm: FAILED - {e}")
    failures.append("openmm")

try:
    import mdtraj
    print(f"✓ mdtraj: {mdtraj.__version__}")
except ImportError as e:
    print(f"✗ mdtraj: FAILED - {e}")
    failures.append("mdtraj")

try:
    import rdkit
    print(f"✓ rdkit: {rdkit.__version__}")
except ImportError as e:
    print(f"✗ rdkit: FAILED - {e}")
    failures.append("rdkit")

try:
    import sklearn
    print(f"✓ scikit-learn: {sklearn.__version__}")
except ImportError as e:
    print(f"✗ scikit-learn: FAILED - {e}")
    failures.append("scikit-learn")

try:
    import deeptime
    print(f"✓ deeptime: {deeptime.__version__}")
except ImportError as e:
    print(f"✗ deeptime: FAILED - {e}")
    failures.append("deeptime")

try:
    import mlcolvar
    print(f"✓ mlcolvar: {mlcolvar.__version__}")
except ImportError as e:
    print(f"✗ mlcolvar: FAILED - {e}")
    failures.append("mlcolvar")

# Test ML/CV dependencies
print("\n=== ML/CV Dependencies ===")
try:
    import torch
    print(f"✓ torch: {torch.__version__}")
    print(f"  - CUDA available: {torch.cuda.is_available()}")
    print(f"  - CPU build: {not torch.cuda.is_available()}")
except ImportError as e:
    print(f"✗ torch: FAILED - {e}")
    failures.append("torch")

try:
    import lightning
    print(f"✓ lightning: {lightning.__version__}")
except ImportError as e:
    print(f"✗ lightning: FAILED - {e}")
    failures.append("lightning")

# Test analysis dependencies
print("\n=== Analysis Dependencies ===")
try:
    import matplotlib
    print(f"✓ matplotlib: {matplotlib.__version__}")
except ImportError as e:
    print(f"✗ matplotlib: FAILED - {e}")
    failures.append("matplotlib")

try:
    import plotly
    print(f"✓ plotly: {plotly.__version__}")
except ImportError as e:
    print(f"✗ plotly: FAILED - {e}")
    failures.append("plotly")

# Test dev tools
print("\n=== Dev Tools ===")
try:
    import pytest
    print(f"✓ pytest: {pytest.__version__}")
except ImportError as e:
    print(f"✗ pytest: FAILED - {e}")
    failures.append("pytest")

try:
    import tox
    print(f"✓ tox: {tox.__version__}")
except ImportError as e:
    print(f"✗ tox: FAILED - {e}")
    failures.append("tox")

try:
    import black
    print(f"✓ black: {black.__version__}")
except ImportError as e:
    print(f"✗ black: FAILED - {e}")
    failures.append("black")

try:
    import isort
    print(f"✓ isort: {isort.__version__}")
except ImportError as e:
    print(f"✗ isort: FAILED - {e}")
    failures.append("isort")

try:
    import ruff
    # Ruff doesn't expose __version__, check via CLI
    import subprocess
    try:
        result = subprocess.run(["ruff", "--version"], capture_output=True, text=True, timeout=5)
        version = result.stdout.strip().split()[-1] if result.returncode == 0 else "installed"
        print(f"✓ ruff: {version}")
    except:
        print(f"✓ ruff: installed")
except ImportError as e:
    print(f"✗ ruff: FAILED - {e}")
    failures.append("ruff")

# Test pmarlo itself
print("\n=== PMARLO ===")
try:
    import pmarlo
    version = getattr(pmarlo, "__version__", "unknown")
    print(f"✓ pmarlo: {version}")

    # Test main API imports
    from pmarlo import Protein, ReplicaExchange, Simulation, Pipeline
    print("✓ Main API imports successful")
except ImportError as e:
    print(f"✗ pmarlo: FAILED - {e}")
    failures.append("pmarlo")

# Summary
print("\n" + "="*50)
if failures:
    print(f"✗ Setup completed with {len(failures)} failures:")
    for pkg in failures:
        print(f"  - {pkg}")
    print("\nPlease review the errors above.")
    sys.exit(1)
else:
    print("✓ All dependencies installed successfully!")
    print("\n🎉 PMARLO environment is ready!")
    sys.exit(0)
PY

# 9) Verify pytest can find OpenMM and pmarlo
echo "[Step 9/9] Verifying pytest environment..."
python - <<'PY'
import sys
import os

print("\n=== Pytest Environment Verification ===")

# 1. Check sys.path
print(f"\nPython executable: {sys.executable}")
print(f"Python version: {sys.version}")
print(f"\nPython path (first 5 entries):")
for i, path in enumerate(sys.path[:5]):
    print(f"  {i}: {path}")

# 2. Verify OpenMM import (critical for replica exchange tests)
print("\n=== Critical Imports for Replica Exchange ===")
try:
    import openmm
    from openmm import app, unit
    print(f"✓ openmm: {openmm.__version__}")
    print(f"  - Location: {openmm.__file__}")
except ImportError as e:
    print(f"✗ openmm import FAILED: {e}")
    sys.exit(1)

# 3. Verify PMARLO import
try:
    import pmarlo
    print(f"✓ pmarlo: {getattr(pmarlo, '__version__', 'unknown')}")
    print(f"  - Location: {pmarlo.__file__}")
except ImportError as e:
    print(f"✗ pmarlo import FAILED: {e}")
    sys.exit(1)

# 4. Verify pytest can run
print("\n=== Pytest Availability ===")
try:
    import pytest
    print(f"✓ pytest: {pytest.__version__}")
    print(f"  - Location: {pytest.__file__}")
except ImportError as e:
    print(f"✗ pytest import FAILED: {e}")
    sys.exit(1)

# 5. Test that pytest can discover tests
print("\n=== Test Discovery Check ===")
test_file = "tests/integration/replica_exchange/test_simulation.py"
if os.path.exists(test_file):
    print(f"✓ Found test file: {test_file}")
else:
    print(f"⚠ Test file not found: {test_file}")

print("\n" + "="*50)
print("✓ Pytest environment is correctly configured!")
print("  OpenMM, pmarlo, and pytest are all importable.")
print("="*50)
PY

echo ""
echo "Running quick pytest verification..."
cd /workspace/pmarlo

# Try to collect tests (don't run them, just verify pytest can see them)
if poetry run python -m pytest --collect-only tests/integration/replica_exchange/test_simulation.py 2>&1 | grep -q "no tests ran\|ERROR"; then
    echo "⚠ Warning: pytest collection had issues, but continuing..."
else
    echo "✓ poetry can discover replica exchange tests"
fi

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "To use the environment:"
echo "  source /workspace/.venv/bin/activate"
echo ""
echo "To run tests:"
echo "  poetry run pytest"
echo "  poetry run pytest tests/unit -n auto"
echo ""
echo "To run replica exchange tests:"
echo "  poetry run pytest tests/integration/replica_exchange/"
echo "  poetry run pytest tests/integration/replica_exchange/test_simulation.py -v"
echo ""
echo "To run quality checks:"
echo "  poetry run tox"
echo ""
