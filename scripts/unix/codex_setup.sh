#!/usr/bin/env bash
set -euxo pipefail

echo "=========================================="
echo "PMARLO Codex Environment Setup"
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

# 5) Install CPU-only PyTorch first (lightning comes from PyPI later)
echo "[Step 5/8] Installing CPU-only PyTorch..."
python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.2,<3.0"

# 6) Configure Poetry and install project dependencies
echo "[Step 6/8] Installing project with Poetry..."
poetry config virtualenvs.create false
poetry config virtualenvs.in-project false

# Lock dependencies (remove old lock first to regenerate with current pyproject.toml)
echo "  Locking dependencies..."
if [ -f poetry.lock ]; then
    echo "  Removing old poetry.lock to regenerate with updated constraints..."
    rm poetry.lock
fi

MAX_RETRIES=3
for i in $(seq 1 $MAX_RETRIES); do
    if poetry lock ; then
        break
    elif [ $i -eq $MAX_RETRIES ]; then
        echo "  Failed to lock dependencies after $MAX_RETRIES attempts"
        echo "  Python version: $(python --version)"
        echo "  This may be due to Python version incompatibility."
        exit 1
    else
        echo "  Lock attempt $i failed, retrying..."
        sleep 2
    fi
done

# Install with all extras (Poetry will skip pdbfixer on Python >= 3.12 via markers)
echo "  Installing dependencies..."
PYTHON_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "  Detected Python version: $PYTHON_VERSION"

# Note: fixer extra includes pdbfixer (Python < 3.12 only), black, isort, and ruff
# Poetry will automatically skip pdbfixer on Python >= 3.12 due to version markers
poetry install \
    --with dev,tests \
    --extras "mlcv app analysis fixer" \
    --no-interaction \
    --no-ansi

# 7) Re-pin CPU PyTorch to ensure no CUDA builds were pulled
echo "[Step 7/8] Re-pinning CPU-only PyTorch..."
python -m pip install --force-reinstall --index-url https://download.pytorch.org/whl/cpu "torch>=2.2,<3.0"

# 8) Run comprehensive smoke tests
echo "[Step 8/8] Running smoke tests..."
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
    print(f"✓ ruff: {ruff.__version__}")
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
echo ""
echo "To run quality checks:"
echo "  poetry run tox"
echo ""
