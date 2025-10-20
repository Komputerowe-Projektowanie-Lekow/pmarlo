# Testing in Codex Environment

## ⚠️ Important: Use `python -m` Not `poetry run`

In the Codex environment, dependencies are installed with `pip` (not Poetry) for better network reliability. This means you should use Python directly, not through Poetry.

## Quick Commands

### ✅ Correct (use these):

```bash
# Activate environment first
source /workspace/.venv/bin/activate

# Run all tests
python -m pytest

# Run unit tests in parallel (fast)
python -m pytest tests/unit -n auto

# Run integration tests
python -m pytest tests/integration

# Run replica exchange tests specifically
python -m pytest tests/integration/replica_exchange/

# Run specific test file
python -m pytest tests/integration/replica_exchange/test_simulation.py

# Run with verbose output
python -m pytest -v tests/unit

# Run quality checks
tox
```

### ❌ Incorrect (don't use these in Codex):

```bash
# DON'T use poetry run - it may use a different environment
poetry run pytest  # ❌ May fail with "No module named 'openmm'"
```

## Why This Matters

### The Setup

1. **Codex has network restrictions** that prevent Poetry from downloading packages
2. **Solution:** We use `pip install -e .` to install PMARLO and dependencies
3. **Result:** Everything is in `/workspace/.venv` managed by pip

### The Problem

- `poetry run` tries to use Poetry's environment management
- Even with `poetry config virtualenvs.create false`, there can be conflicts
- Poetry may not see pip-installed packages correctly

### The Solution

- Use `python -m pytest` which directly uses the venv's Python
- This ensures pytest runs in the same environment where pip installed packages
- All dependencies (numpy, openmm, etc.) are found correctly

## Verification

To verify your environment is working:

```bash
# Check Python is from venv
which python
# Should show: /workspace/.venv/bin/python

# Check OpenMM is importable
python -c "import openmm; print(openmm.__version__)"
# Should show: 8.3.1 (or similar)

# Check pmarlo is importable
python -c "import pmarlo; print(pmarlo.__version__)"
# Should show: 0.1.0 (or similar)
```

## Running Specific Test Categories

```bash
# Unit tests only
python -m pytest tests/unit

# Integration tests only
python -m pytest tests/integration

# Tests with markers
python -m pytest -m "unit"
python -m pytest -m "integration"
python -m pytest -m "replica"

# Specific test patterns
python -m pytest -k "test_simulation"
python -m pytest -k "replica_exchange"
```

## Parallel Execution

Use `-n auto` to run tests in parallel (faster):

```bash
# Parallel unit tests
python -m pytest tests/unit -n auto

# Parallel with verbose output
python -m pytest tests/unit -n auto -v
```

## Coverage

```bash
# Run with coverage
python -m pytest --cov=pmarlo --cov-report=html

# View coverage report
# Open htmlcov/index.html in a browser
```

## Debugging Failed Tests

```bash
# Run with detailed output
python -m pytest -vv tests/integration/replica_exchange/test_simulation.py

# Stop on first failure
python -m pytest -x tests/unit

# Drop into debugger on failure
python -m pytest --pdb tests/unit
```

## Common Issues

### Issue: "No module named 'openmm'"

**Cause:** Using `poetry run pytest` instead of `python -m pytest`

**Solution:**
```bash
source /workspace/.venv/bin/activate
python -m pytest  # ✅ Use this
```

### Issue: "ModuleNotFoundError: No module named 'pmarlo'"

**Cause:** Not in the right directory or venv not activated

**Solution:**
```bash
cd /workspace/pmarlo
source /workspace/.venv/bin/activate
python -m pytest
```

### Issue: Tests run but skip OpenMM tests

**Cause:** OpenMM not installed or wrong environment

**Solution:**
```bash
# Verify OpenMM
python -c "import openmm; print('✓ OpenMM works')"

# If that fails, re-run setup
cd /workspace/pmarlo/scripts/unix
bash codex_setup.sh
```

## CI/CD Integration

If you're running tests in CI/CD:

```yaml
# Example GitHub Actions / GitLab CI
script:
  - source /workspace/.venv/bin/activate
  - python -m pytest tests/unit -n auto
  - python -m pytest tests/integration
```

## Summary

| Command | Use Case | Speed |
|---------|----------|-------|
| `python -m pytest` | All tests | Medium |
| `python -m pytest tests/unit -n auto` | Fast feedback | Fast |
| `python -m pytest tests/integration` | Full validation | Slow |
| `tox` | Quality gate | Slow |

**Remember:** Always use `python -m pytest`, not `poetry run pytest` in Codex! 🎯

