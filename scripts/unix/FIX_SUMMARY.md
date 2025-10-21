# Codex Environment Scripts - Fix Summary

## Issues Fixed

### 1. Lightning Installation Failure ❌→✅
**Problem:**
```
ERROR: Could not find a version that satisfies the requirement lightning<3.0,>=2.0
```

**Root Cause:**
The PyTorch CPU index (`https://download.pytorch.org/whl/cpu`) only contains `torch` packages, not `lightning`.

**Solution:**
- Install only `torch` from the CPU index
- Let Poetry install `lightning` from PyPI during the main installation step
- This ensures both packages are properly installed without conflicts

### 2. Black and Ruff Missing on Python 3.12+ ❌→✅
**Problem:**
```
✗ black: FAILED - No module named 'black'
✗ ruff: FAILED - No module named 'ruff'
```

**Root Cause:**
The script was skipping the entire `fixer` extra on Python >= 3.12 to avoid installing `pdbfixer` (which isn't compatible with Python 3.12+). However, this also skipped `black`, `isort`, and `ruff`, which DO work on Python 3.12+.

**Solution:**
- Always install the `fixer` extra
- Poetry automatically respects the `python_version < '3.12'` marker on `pdbfixer` in `pyproject.toml`
- This means:
  - Python 3.10-3.11: Installs pdbfixer, black, isort, ruff
  - Python 3.12+: Skips pdbfixer, installs black, isort, ruff

### 3. Python 3.13 Support ❌→✅
**Problem:**
Python 3.13 wasn't supported in `pyproject.toml`.

**Solution:**
Updated version constraints from `>=3.10,<3.13` to `>=3.10,<3.14` in:
- `[project]` section: `requires-python`
- `[tool.poetry.dependencies]` section: `python`
- `[tool.black]` section: `target-version` (added "py313")

## Files Modified

1. **`scripts/unix/codex_setup.sh`**
   - Fixed PyTorch/Lightning installation strategy
   - Simplified fixer extra logic (always install, let Poetry handle markers)

2. **`scripts/unix/codex_maintenance_script.sh`**
   - Same fixes as setup script

3. **`pyproject.toml`**
   - Updated Python version constraints to support 3.10-3.13
   - Updated Black target versions to include py313

4. **`README.md`** (main)
   - Updated Python version range to 3.10-3.13
   - Clarified fixer extra contents

5. **`scripts/unix/README.md`**
   - Updated virtual environment Python version range
   - Added note about pdbfixer availability

6. **`scripts/unix/CHANGES.md`**
   - Documented all fixes

## Expected Results After These Fixes

When running `codex_setup.sh`, you should now see:

```
=== Core Dependencies ===
✓ numpy: <version>
✓ scipy: <version>
✓ pandas: <version>
✓ openmm: <version>
✓ mdtraj: <version>
✓ rdkit: <version>
✓ scikit-learn: <version>
✓ deeptime: <version>
✓ mlcolvar: <version>

=== ML/CV Dependencies ===
✓ torch: 2.9.0+cpu
  - CUDA available: False
  - CPU build: True
✓ lightning: <version>

=== Analysis Dependencies ===
✓ matplotlib: <version>
✓ plotly: <version>

=== Dev Tools ===
✓ pytest: <version>
✓ tox: <version>
✓ black: <version>        ← NOW WORKS!
✓ isort: <version>
✓ ruff: <version>         ← NOW WORKS!

=== PMARLO ===
✓ pmarlo: <version>
✓ Main API imports successful

==================================================
✓ All dependencies installed successfully!

🎉 PMARLO environment is ready!
```

## Python Version Compatibility Matrix

| Python Version | pdbfixer | black | isort | ruff | torch | lightning |
|---------------|----------|-------|-------|------|-------|-----------|
| 3.10          | ✓        | ✓     | ✓     | ✓    | ✓     | ✓         |
| 3.11          | ✓        | ✓     | ✓     | ✓    | ✓     | ✓         |
| 3.12          | ✗        | ✓     | ✓     | ✓    | ✓     | ✓         |
| 3.13          | ✗        | ✓     | ✓     | ✓    | ✓     | ✓         |

## Testing the Fix

On a fresh Codex environment:

```bash
# Run setup script
cd /workspace/pmarlo/scripts/unix
bash codex_setup.sh

# Should complete with:
# ✓ All dependencies installed successfully!
# 🎉 PMARLO environment is ready!

# Verify manually
source /workspace/.venv/bin/activate
python -c "import black, ruff, torch, lightning; print('✓ All critical imports work!')"

# Run tests
poetry run pytest tests/unit -n auto

# Run integration tests (including replica exchange)
poetry run pytest tests/integration/replica_exchange/test_simulation.py

# Full quality gate
poetry run tox
```

## Key Takeaways

1. **PyTorch CPU Index:** Only use it for `torch`, install everything else from PyPI
2. **Poetry Markers:** Trust Poetry to handle `python_version` markers - don't manually skip extras
3. **Simplicity:** Less conditional logic = fewer bugs
4. **Python 3.13:** Now fully supported across the project

## Next Steps

If you encounter any issues:

1. Check Python version: `python --version`
2. Check installed packages: `pip list | grep -E "(black|ruff|torch|lightning)"`
3. Re-run setup: `bash codex_setup.sh` (it's idempotent)
4. Check Poetry config: `poetry config --list`
5. Manual verification: `poetry install --extras "fixer" --verbose`
