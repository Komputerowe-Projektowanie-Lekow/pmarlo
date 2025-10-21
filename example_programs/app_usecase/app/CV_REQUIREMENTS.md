# CV-Informed Sampling Requirements & Technical Details

## Status: ✅ CV Biasing Fully Implemented

**CV-biased sampling is production-ready** with TorchScript-based feature extraction and strict validation.

### What's Working
- ✅ **TorchScript Feature Extraction**: Positions + box → molecular features (distances, angles, dihedrals)
- ✅ **Embedded Feature Extraction**: All computation inside TorchScript (no Python per-step)
- ✅ **Periodic Boundary Conditions**: Minimum-image convention for distances under PBC
- ✅ **Strict Validation**: Hash-locked feature specs, dtype checks, dimension validation
- ✅ **CPU-Viable Performance**: 2–3× slowdown vs unbiased on CPU (acceptable for production)
- ✅ **Harmonic Restraint Physics**: E = k·Σ(cv²) implemented correctly
- ✅ **Periodic Logging**: CV mean/std and bias energy logged every 1000 steps
- ✅ **Configuration-Driven**: Explicit enable/disable with no fallbacks

### Current Status

```
🟢 Feature Extraction:     TorchScript module (CPU/GPU native)
🟢 CV Computation:         Deep-TICA network (TorchScript)
🟢 Bias Potential:         Harmonic restraint in CV space
🟢 Force Computation:      Automatic differentiation (OpenMM)
🟢 Validation:             Strict, hash-locked, no fallbacks
🟢 Performance:            2-3× slowdown on CPU (acceptable)
🟢 Logging:                Periodic CV/bias stats (every 1000 steps)
🟢 Configuration:          Explicit keys, strict validation
```

See `CV_INTEGRATION_GUIDE.md` for comprehensive usage guide.

---

## Technical Architecture

### TorchScript Feature Extraction

**Input Contract**: TorchForce provides only:
- `positions`: FloatTensor[N, 3] in nanometers
- `box`: FloatTensor[3, 3] box vectors in nanometers

**All feature computation happens inside TorchScript**—no Python callbacks.

### Feature Extraction Module

`TorchScriptFeatureExtractor` computes molecular features from atomic positions:

**Supported Feature Types:**
1. **Pairwise distances** (with minimum-image PBC)
2. **Bond angles** (three-body)
3. **Dihedral angles** (four-body torsion)

**Key Implementation Details:**

```python
# Minimum-image distance under PBC
fractional = positions @ inv_box          # Convert to fractional coords
wrapped = fractional - round(fractional)  # Wrap to [-0.5, 0.5)
min_image = wrapped @ box                 # Back to Cartesian
distance = ||min_image||                  # Euclidean norm
```

**Vectorized Operations**: All computations use PyTorch tensor ops (no loops).

**PBC Per-Feature**: Each feature can enable/disable PBC independently via `pbc: true/false` in spec.

### Feature Specification Format

```yaml
use_pbc: true   # Global PBC flag (informational)
features:
  - name: dist_N_CA
    type: distance          # distance | angle | dihedral
    atom_indices: [0, 1]    # Atom indices (0-based)
    pbc: true               # Use minimum-image for this feature
    weight: 1.0             # Feature weight (multiplicative)
```

**Hash-Locked**: The spec's SHA-256 hash is embedded in the exported model. Any change to the spec (adding, removing, reordering features) will cause a validation error.

### Export Pipeline

The export process creates a **single TorchScript module** containing:

1. **Feature Extractor**: `TorchScriptFeatureExtractor(spec)`
2. **Scaler**: Mean/std normalization (embedded as buffers)
3. **CV Model**: Deep-TICA network
4. **Bias Potential**: Harmonic restraint (k·Σ(cv²))

**Optimization**: `torch.jit.optimize_for_inference()` applied before saving.

**Metadata Embedded**:
```python
model.feature_spec_sha256                    # str (SHA-256 hash)
model.uses_periodic_boundary_conditions      # bool
model.atom_count                             # int
```

### Load-Time Validation

`system_builder.py` performs strict validation when loading a CV model:

1. **Configuration Validation**:
   - All required keys present (`enable_cv_bias`, `bias_mode`, `torch_threads`, `precision`)
   - `bias_mode` must be `"harmonic"` (only supported mode)
   - `precision` must be `"single"` or `"double"`
   - `torch_threads` must be positive integer

2. **File Validation**:
   - Model file exists at specified path
   - Scaler file exists
   - Config JSON exists and parseable

3. **Hash Validation**:
   - Model's embedded `feature_spec_sha256` matches current `feature_spec.yaml` hash
   - Prevents using mismatched models

4. **Dtype Validation**:
   - All model parameters are `float32`
   - All floating-point buffers are `float32`
   - Prevents mixed-precision models

5. **Dimension Validation**:
   - Input feature count matches scaler dimension
   - CV dimension matches metadata
   - Atom count matches metadata

6. **Method Validation**:
   - Model has `compute_cvs()` method (required for monitoring)

7. **Dry-Run Test**:
   - Dummy forward pass with zeros to verify model works
   - Confirms CV dimension matches expected

**No Fallbacks**: Any validation failure raises an exception and terminates before simulation starts.

## Installation

### Minimal Setup (CPU Testing)

```bash
# Install OpenMM-Torch
conda install -c conda-forge openmm-torch

# Verify installation
python -c "import openmmtorch; print('✓ openmm-torch installed')"
```

### Production Setup (GPU)

```bash
# 1. Check CUDA version
nvidia-smi  # Note CUDA version (e.g., 11.8 or 12.1)

# 2. Install CUDA-enabled PyTorch
# For CUDA 11.8:
pip install torch --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1:
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 3. Install OpenMM-Torch
conda install -c conda-forge openmm-torch

# 4. Verify CUDA available
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### Verify Configuration

```python
from pmarlo.settings import load_defaults, load_feature_spec

# Load configuration
config = load_defaults()
print(f"enable_cv_bias: {config['enable_cv_bias']}")
print(f"bias_mode: {config['bias_mode']}")
print(f"torch_threads: {config['torch_threads']}")
print(f"precision: {config['precision']}")

# Load and hash feature spec
spec, spec_hash = load_feature_spec()
print(f"Feature spec hash: {spec_hash}")
print(f"Features defined: {len(spec['features'])}")
```

## Performance Expectations

### CPU Performance

Run the benchmark harness for accurate measurements on your hardware:

```bash
# Baseline (no bias)
poetry run python scripts/bench_openmm.py --platform CPU --with-bias no --steps 5000

# With CV bias (4 threads)
poetry run python scripts/bench_openmm.py --platform CPU --with-bias yes \
    --model path/to/model.pt --steps 5000 --torch-threads 4

# With CV bias (8 threads)
poetry run python scripts/bench_openmm.py --platform CPU --with-bias yes \
    --model path/to/model.pt --steps 5000 --torch-threads 8
```

**Expected Results:**

| Configuration | Steps/sec | Slowdown | Notes |
|---------------|-----------|----------|-------|
| Unbiased (CPU) | 50–100 | 1× (baseline) | Depends on system size |
| CV-biased (4 threads) | 20–40 | 2–3× | Recommended thread count |
| CV-biased (8 threads) | 25–50 | 2–2.5× | Diminishing returns >8 |
| CV-biased (GPU CUDA) | 80–150 | 1.5–2× | Production recommended |

**Why Slower?**
1. Feature extraction (vectorized, but still adds work)
2. Neural network forward pass
3. Automatic differentiation for force gradients
4. OpenMM ↔ PyTorch data transfer

**Optimization Guidelines:**
- **`torch_threads`**: Set to 4–8 for best CPU performance
- **`precision`**: Use `single` (double adds ~2× overhead)
- **Avoid oversubscription**: Don't set `torch_threads` higher than physical core count

### GPU Performance

GPU drastically improves CV bias performance:

```bash
# Ensure CUDA PyTorch is installed
python -c "import torch; print(torch.cuda.is_available())"

# Run benchmark on GPU
poetry run python scripts/bench_openmm.py --platform CUDA --with-bias yes \
    --model path/to/model.pt --steps 5000
```

**GPU Advantages:**
- Parallel feature computation
- Fast neural network inference
- Minimal impact on OpenMM simulation (already on GPU)

**Expected**: 1.5–2× slower than unbiased (vs 2–3× on CPU).

## Configuration Reference

### Required Configuration Keys

All keys are **required** in `src/pmarlo/settings/defaults.yaml`:

```yaml
# ✅ Enable or disable CV biasing (explicit)
enable_cv_bias: true

# ✅ Bias mode (only 'harmonic' supported)
bias_mode: harmonic

# ✅ PyTorch thread count (4-8 recommended for CPU)
torch_threads: 4

# ✅ Precision (single recommended, double adds overhead)
precision: single

# ✅ Path to feature specification (relative or absolute)
feature_spec_path: feature_spec.yaml
```

**Missing any key → `ConfigurationError` at startup.**

### Configuration Validation

The `load_defaults()` function validates:
- All required keys present
- `bias_mode` in `{"harmonic"}` (enum constraint)
- `precision` in `{"single", "double"}` (enum constraint)
- `torch_threads` > 0 (positive integer)

### Feature Spec Validation

The `canonicalize_feature_spec()` function validates:
- All features have `type`, `atom_indices`
- Types in `{"distance", "angle", "dihedral"}`
- Atom indices are non-negative integers
- Correct number of indices per type (2 for distance, 3 for angle, 4 for dihedral)
- No duplicate feature names (if provided)

## Logging and Monitoring

### Startup Logs

When CV bias is enabled, `system_builder.py` logs:

```
CV bias enabled (mode=harmonic)
  Torch threads: 4
  Torch precision: single
  Model feature hash: a1b2c3...
  Specification hash: a1b2c3...
  Force group: 1
  Uses periodic boundary conditions: True
```

### Runtime Logs

`replica_exchange.py` logs CV statistics every 1000 steps:

```
Bias stats after 1000 steps: energy mean=12.345678 std=3.456789; CV mean=[0.1234 -0.0567] std=[0.4321  0.2345]
Bias stats after 2000 steps: energy mean=14.567890 std=4.123456; CV mean=[0.1456 -0.0678] std=[0.4532  0.2456]
...
```

**What to Monitor:**
- **Bias energy**: Should be non-zero and vary over time
- **CV mean**: Should shift as system explores CV space
- **CV std**: Indicates spread in CV values (broader = more diverse sampling)

### Benchmark Output

The benchmark harness prints:

```
Benchmark summary
------------------
Platform           : CPU
Bias enabled       : True
Steps              : 5000
Torch threads      : 4
Precision          : single
Steps / second     : 35.42
Bias energy mean   : 12.345678 kJ/mol
Bias energy std    : 3.456789 kJ/mol
CV mean            : [0.1234 -0.0567]
CV std             : [0.4321  0.2345]
```

## Troubleshooting

### ConfigurationError: "missing required keys"

**Cause**: `defaults.yaml` is missing one or more required keys.

**Fix**: Add all required keys:
```yaml
enable_cv_bias: true
bias_mode: harmonic
torch_threads: 4
precision: single
feature_spec_path: feature_spec.yaml
```

### RuntimeError: "Feature specification mismatch"

**Cause**: Model's embedded hash doesn't match current `feature_spec.yaml`.

**Root Cause**: You modified the feature spec after exporting the model.

**Fix**:
1. Restore original `feature_spec.yaml`, OR
2. Re-export model with new spec

### RuntimeError: "Model parameter must be float32"

**Cause**: Model was exported with mixed precision or double precision.

**Fix**: Re-train and re-export ensuring all parameters are `float32`.

### RuntimeError: "CV dimension mismatch"

**Cause**: Model's CV dimension doesn't match metadata.

**Fix**: Re-export model with correct metadata.

### Slow Performance (>5× slowdown)

**Diagnostic Steps**:

1. **Check thread count**:
   ```python
   import torch
   print(f"Torch threads: {torch.get_num_threads()}")
   ```
   Should match `torch_threads` in config (4–8).

2. **Check precision**:
   ```python
   config = load_defaults()
   print(f"Precision: {config['precision']}")
   ```
   Should be `"single"`.

3. **Run benchmark**:
   ```bash
   poetry run python scripts/bench_openmm.py --platform CPU --with-bias yes \
       --model path/to/model.pt --steps 5000 --torch-threads 4
   ```
   Compare `Steps / second` to expected range (20–40 for CPU).

4. **Consider GPU**:
   If slowdown is unacceptable, install CUDA PyTorch for ~4× speedup.

### NaN or Zero Bias Energy

**Diagnostic**:
```python
# Check if model loads
import torch
model = torch.jit.load("path/to/model.pt", map_location="cpu")
print(hasattr(model, "compute_cvs"))  # Should be True

# Test forward pass
pos = torch.zeros(22, 3, dtype=torch.float32)  # Replace 22 with atom_count
box = torch.eye(3, dtype=torch.float32)
energy = model(pos, box)
print(f"Test energy: {energy}")  # Should be non-NaN scalar
```

**Common Causes**:
- Model file corrupted or incomplete
- Wrong atom count in dummy test
- Scaler parameters not loaded correctly

**Fix**: Re-export model, ensuring all steps complete without errors.

## Physics Terminology

### Harmonic Restraint vs Exploration Bias

**What It Is**: The implemented bias is a **harmonic restraint in CV space**:

```
E_bias = k · Σ(cv_i²)
```

This adds a quadratic penalty when CVs deviate from zero.

**What It Is NOT**:
- ❌ **Not metadynamics**: Metadynamics fills visited regions with Gaussian hills to discourage revisiting
- ❌ **Not adaptive biasing**: ABF/eABD compute free energy on-the-fly and flatten it
- ❌ **Not guaranteed exploration**: The bias pushes away from CV=0, but doesn't ensure better sampling of physical states

**Correct Description**: "Harmonic restraint that encourages non-zero CV values."

**Incorrect Description**: "Exploration bias that enhances sampling." ← This is misleading.

### When This Bias Helps

- Your CVs are informative and cover the relevant slow modes
- You want to sample regions with non-zero CV values (e.g., extended conformations)
- You're willing to reweight the bias out during analysis

### When This Bias Doesn't Help

- Your CVs are not informative (bias won't magically fix bad CVs)
- You need to map the full free energy landscape (use metadynamics or umbrella sampling)
- You want to find transition paths (use string method or milestoning)

**Bottom Line**: This is a **restraint tool**, not a universal sampling enhancer.

## References

- **TorchScript Documentation**: https://pytorch.org/docs/stable/jit.html
- **OpenMM-Torch**: https://github.com/openmm/openmm-torch
- **Deep-TICA**: Schwantes & Pande, *J. Chem. Theory Comput.* **9**, 2000 (2013)
- **Feature Extraction**: `src/pmarlo/features/deeptica/ts_feature_extractor.py`
- **System Builder**: `src/pmarlo/replica_exchange/system_builder.py`
- **Export Logic**: `src/pmarlo/features/deeptica/export.py`

## Support

For issues or questions:
1. Check simulation logs: `app_output/sims/*/replica_exchange/`
2. Run benchmark harness: `scripts/bench_openmm.py`
3. Verify configuration: `python -c "from pmarlo.settings import load_defaults; print(load_defaults())"`
4. Consult `CV_INTEGRATION_GUIDE.md` for usage examples
