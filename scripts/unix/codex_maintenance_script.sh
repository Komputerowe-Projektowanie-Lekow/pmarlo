#!/usr/bin/env bash
set -euxo pipefail

echo "=========================================="
echo "PMARLO Codex Environment Maintenance"
echo "=========================================="

# 0) Ensure we're in the right directory
cd /workspace/pmarlo

# 1) Activate virtual environment
echo "[Step 1/6] Activating virtual environment..."
VENV=/workspace/.venv
if [ ! -d "$VENV" ]; then
    echo "Error: Virtual environment not found at $VENV"
    echo "Please run codex_setup.sh first to initialize the environment."
    exit 1
fi
source "$VENV/bin/activate"

# 2) Update repository
echo "[Step 2/6] Updating repository..."
git fetch origin --tags --prune
CURRENT_BRANCH=$(git branch --show-current)
echo "  Current branch: $CURRENT_BRANCH"

# Ask user if they want to pull latest changes (auto-yes in non-interactive mode)
if git pull origin "$CURRENT_BRANCH" --ff-only; then
    echo "  ✓ Repository updated successfully"
else
    echo "  ⚠ Could not fast-forward. You may need to resolve conflicts manually."
    echo "  Continuing with current state..."
fi

# 3) Update pip, setuptools, wheel, and Poetry
echo "[Step 3/6] Updating core Python tools..."
python -m pip install --upgrade pip setuptools wheel
python -m pip install --upgrade "poetry>=2.1.3,<3.0"

# 4) Update project dependencies with pip
echo "[Step 4/6] Updating project dependencies..."
PYTHON_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "  Detected Python version: $PYTHON_VERSION"

# Upgrade pip and build tools
echo "  Upgrading build tools..."
python -m pip install --upgrade "setuptools>=69" "wheel" "setuptools_scm[toml]>=8.0"

# Reinstall project with all extras (this updates dependencies)
echo "  Reinstalling pmarlo with all dependencies..."
if python -c "import sys; exit(0 if sys.version_info < (3, 12) else 1)"; then
    echo "  Installing with fixer extra (includes pdbfixer)..."
    python -m pip install --upgrade --force-reinstall --no-deps -e "."
    python -m pip install -e ".[mlcv,app,analysis,fixer]"
else
    echo "  Installing with fixer extra (pdbfixer will be skipped on Python ${PYTHON_VERSION})..."
    python -m pip install --upgrade --force-reinstall --no-deps -e "."
    python -m pip install -e ".[mlcv,app,analysis,fixer]"
fi

# Update dev and test dependencies
echo "  Updating dev and test dependencies..."
python -m pip install --upgrade \
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

echo "  ✓ Dependencies updated successfully"

# 5) Ensure CPU-only PyTorch (lightning comes from PyPI)
echo "[Step 5/6] Ensuring CPU-only PyTorch..."
python -m pip install --force-reinstall --index-url https://download.pytorch.org/whl/cpu "torch>=2.2,<3.0"

# 6) Verify installation
echo "[Step 6/6] Verifying installation..."
python - <<'PY'
import sys

failures = []

print("\n=== Quick Verification ===")

# Core dependencies
required_imports = [
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("pandas", "pandas"),
    ("openmm", "openmm"),
    ("mdtraj", "mdtraj"),
    ("rdkit", "rdkit"),
    ("sklearn", "scikit-learn"),
    ("deeptime", "deeptime"),
    ("mlcolvar", "mlcolvar"),
    ("torch", "torch"),
    ("lightning", "lightning"),
    ("matplotlib", "matplotlib"),
    ("pytest", "pytest"),
    ("tox", "tox"),
]

for module_name, display_name in required_imports:
    try:
        module = __import__(module_name)
        version = getattr(module, "__version__", "unknown")
        print(f"✓ {display_name}: {version}")
    except ImportError as e:
        print(f"✗ {display_name}: FAILED - {e}")
        failures.append(display_name)

# Test pmarlo
try:
    import pmarlo
    version = getattr(pmarlo, "__version__", "unknown")
    print(f"✓ pmarlo: {version}")

    # Test main API
    from pmarlo import Protein, ReplicaExchange, Simulation, Pipeline
    print("✓ Main API imports successful")
except ImportError as e:
    print(f"✗ pmarlo: FAILED - {e}")
    failures.append("pmarlo")

# Test PyTorch CPU
try:
    import torch
    if torch.cuda.is_available():
        print("⚠ Warning: CUDA build detected. Expected CPU-only build.")
        print("  This may cause issues on CPU-only systems.")
    else:
        print("✓ PyTorch CPU-only build confirmed")
except Exception as e:
    print(f"⚠ Could not verify PyTorch build type: {e}")

# Summary
print("\n" + "="*50)
if failures:
    print(f"✗ Verification completed with {len(failures)} failures:")
    for pkg in failures:
        print(f"  - {pkg}")
    print("\nSome dependencies are missing. Consider re-running setup.")
    sys.exit(1)
else:
    print("✓ All dependencies verified successfully!")
    print("\n🎉 Environment is up to date!")
    sys.exit(0)
PY

echo ""
echo "=========================================="
echo "Maintenance Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  - Run tests: poetry run pytest"
echo "  - Run quality checks: poetry run tox"
echo ""
