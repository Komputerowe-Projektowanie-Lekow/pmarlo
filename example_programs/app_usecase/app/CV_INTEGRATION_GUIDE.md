# CV-Informed Sampling Integration Guide

## Overview

The `pmarlo` package supports **CV-informed sampling** using trained Deep-TICA models to apply harmonic restraints in collective variable (CV) space during molecular dynamics simulations. This implements a **restraint potential** that encourages the system to sample regions with non-zero CV values.

## How It Works

### 1. Training Phase

Train a Deep-TICA model on simulation data to learn collective variables:

```python
# In the app: Model Training tab
# or via API:
from pmarlo.shards import build_from_shards

result, ds_hash = build_from_shards(
    shard_jsons=["shard1.json", "shard2.json"],
    out_bundle="model.pbz",
    bins={"Rg": 64, "RMSD_ref": 64},
    lag=5,
    learn_cv=True,
    deeptica_params={"hidden": [64, 32], "max_epochs": 200},
)
```

### 2. Export Phase

The model is automatically exported as a **TorchScript module** with embedded feature extraction:

```
app_output/models/training-20250117-120000/
├── deeptica_cv_bias.pt          # TorchScript module (positions+box → energy)
├── deeptica_cv_bias_scaler.npz  # Scaler parameters
├── deeptica_cv_bias_config.json # Configuration
└── deeptica_cv_bias_metadata.json # Usage info
```

**Key Innovation**: The exported model is a complete TorchScript module that:

1. **Extracts molecular features** from atomic positions (distances, angles, dihedrals)
2. **Scales features** using fitted scaler parameters
3. **Computes CVs** using the Deep-TICA network
4. **Applies harmonic restraint**: **E_bias = k · Σ(cv_i²)**
5. **Returns energy** in kJ/mol (OpenMM computes forces via automatic differentiation)

All computation happens **inside TorchScript** at MD step time—no Python callbacks, no per-step feature extraction overhead.

### 3. Simulation Phase

OpenMM uses the exported TorchScript module via `TorchForce`:

```python
from openmmtorch import TorchForce
import torch

# Load the complete CV bias module
model = torch.jit.load("deeptica_cv_bias.pt")

# Add as custom force to OpenMM system
cv_force = TorchForce("deeptica_cv_bias.pt")
cv_force.setForceGroup(1)
cv_force.setUsesPeriodicBoundaryConditions(True)
system.addForce(cv_force)

# OpenMM automatically computes forces: F = -∇E_bias
```

## Physics of the Bias

### Harmonic Restraint in CV Space

The CV bias potential applies a **harmonic restraint around the CV origin**:

```
E_bias = k · (cv₁² + cv₂² + ... + cvₙ²)

where:
- k = bias strength (default: 10.0 kJ/mol)
- cv_i = i-th collective variable value
```

**Important:** This is a **restraint**, not an "exploration" bias. The potential:
- Adds a repulsive penalty proportional to |CV|²
- Encourages sampling of states with non-zero CV values
- Does NOT guarantee enhanced conformational exploration
- Is NOT metadynamics (which fills visited regions with hills)

For true **enhanced exploration**, use metadynamics or adaptive biasing schemes (out of scope for this release).

### Example

If CV1 represents RMSD from a reference structure:
- Small CV1 → Low bias → System comfortable near reference
- Large CV1 → High bias → Energetic penalty pushes away

**Result**: System samples a broader range of CV values, but this does not automatically translate to better conformational coverage—it depends on how informative your CVs are.

## Architecture

### TorchScript Pipeline (All Operations GPU/CPU Native)

```
Atomic Positions (N × 3 tensor, nm) + Box Vectors (3 × 3 tensor, nm)
         ↓
TorchScriptFeatureExtractor
    ├─ Compute pairwise distances (with PBC)
    ├─ Compute angles
    ├─ Compute dihedrals
    └─ Apply feature weights
         ↓
    Feature Vector (F-dimensional)
         ↓
CVBiasPotential.forward()
    ├─ Scale features: (features - mean) / scale
    ├─ Compute CVs: Deep-TICA network
    └─ Apply bias: k · Σ(cv²)
         ↓
    Energy (scalar, kJ/mol)
         ↓
OpenMM: F = -∇E via automatic differentiation
         ↓
    Biasing Forces
         ↓
Biased MD Simulation
```

**All steps run in TorchScript**—no Python loops, no per-step callbacks. This enables CPU-viable performance.

## CPU Viability

### Performance Benchmarks

Run the benchmark harness to measure performance on your hardware:

```bash
# Unbiased baseline
poetry run python scripts/bench_openmm.py --platform CPU --with-bias no --steps 5000

# With CV bias
poetry run python scripts/bench_openmm.py --platform CPU --with-bias yes \
    --model path/to/deeptica_cv_bias.pt --steps 5000 --torch-threads 4
```

**Expected Performance (CPU):**

| Configuration | Steps/second (approx) | Slowdown vs Unbiased |
|---------------|----------------------|----------------------|
| Unbiased MD (CPU) | 50–100 | 1× (baseline) |
| CV-biased (4 threads) | 20–40 | ~2–3× |
| CV-biased (8 threads) | 25–50 | ~2–2.5× |

**GPU Performance:**
- With CUDA: ~1.5–2× slower than unbiased (much faster than CPU bias)
- Recommended for production workflows

**Slowdown Factors:**
1. Feature extraction (distances, angles, dihedrals)
2. Neural network forward pass
3. Automatic differentiation for force computation
4. Data transfer between OpenMM and PyTorch tensors

**Optimization Tips:**
- Set `torch_threads` to 4–8 for best CPU throughput (see Configuration below)
- Avoid `torch_threads > physical_cores` to prevent oversubscription
- Use `precision: single` (double precision adds ~2× overhead)

### Quick 10 ps Smoke Test

```bash
# Create a minimal test configuration
cat > test_config.yaml <<EOF
enable_cv_bias: true
bias_mode: harmonic
torch_threads: 4
precision: single
feature_spec_path: feature_spec.yaml
EOF

# Run a short simulation (5000 steps × 2 fs = 10 ps)
export PMARLO_CONFIG_FILE=test_config.yaml
poetry run python scripts/bench_openmm.py --platform CPU --with-bias yes \
    --model path/to/deeptica_cv_bias.pt --steps 5000 --torch-threads 4
```

Expected output:
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

## Configuration

### Required Keys (`src/pmarlo/settings/defaults.yaml`)

All CV bias behavior is controlled via configuration—**no silent defaults**:

```yaml
enable_cv_bias: true          # Must be explicit; no auto-enable
bias_mode: harmonic           # Only 'harmonic' supported (enum)
torch_threads: 4              # Thread count for PyTorch (default: 4)
precision: single             # 'single' or 'double' (single recommended for CPU)
feature_spec_path: feature_spec.yaml  # Path to feature specification
```

**Missing any required key will cause a `ConfigurationError` at startup.**

### Configuration Behavior

| `enable_cv_bias` | `cv_model_path` | Behavior |
|------------------|-----------------|----------|
| `false` | `None` | Unbiased simulation (normal) |
| `false` | Provided | **Error**: Model provided but bias disabled |
| `true` | `None` | **Error**: Bias enabled but no model |
| `true` | Provided | CV-biased simulation |

**No fallbacks.** Misconfiguration terminates immediately with a clear error message.

### Feature Specification

The `feature_spec.yaml` defines molecular features extracted by TorchScript:

```yaml
use_pbc: true
features:
  - name: dist_N_CA
    type: distance
    atom_indices: [0, 1]
    pbc: true
    weight: 1.0
  - name: angle_N_CA_C
    type: angle
    atom_indices: [0, 1, 2]
    pbc: true
    weight: 1.0
  - name: dihedral_phi
    type: dihedral
    atom_indices: [4, 6, 8, 14]
    pbc: true
    weight: 1.0
```

**Hash-locked:** The spec's SHA-256 hash is embedded in the exported model. Changing the spec or reordering features will cause a validation error at load time.

## Requirements

### Essential
- **`openmm-torch`**: Interface between PyTorch and OpenMM
  ```bash
  conda install -c conda-forge openmm-torch
  ```
- **PyTorch**: CPU or CUDA version
  ```bash
  # CPU-only (for testing)
  pip install torch

  # CUDA version (for production; check your CUDA version first)
  pip install torch --index-url https://download.pytorch.org/whl/cu118
  ```

### Verification

```python
import torch
import openmmtorch
from pmarlo.settings import load_defaults, load_feature_spec

# Check PyTorch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Check configuration
config = load_defaults()
print(f"CV bias enabled: {config['enable_cv_bias']}")
print(f"Bias mode: {config['bias_mode']}")
print(f"Torch threads: {config['torch_threads']}")

# Verify feature spec
spec, spec_hash = load_feature_spec()
print(f"Feature spec hash: {spec_hash}")
print(f"Number of features: {len(spec['features'])}")
```

## Usage in the App

### Step 1: Train Model
1. Go to "Model Training" tab
2. Configure training parameters
3. Click "Train Deep-TICA model"
4. Wait for training to complete
5. Model is automatically exported as TorchScript with CV bias wrapper

### Step 2: Configure CV Bias

Edit `src/pmarlo/settings/defaults.yaml`:
```yaml
enable_cv_bias: true
bias_mode: harmonic
torch_threads: 4
precision: single
```

### Step 3: Run CV-Informed Sampling
1. Go to "Sampling" tab
2. Select your trained model
3. Configure simulation parameters
4. Click "Run replica exchange"

### Step 4: Monitor

Simulation logs every 1000 steps:
```
Bias stats after 1000 steps: energy mean=12.345 std=3.456; CV mean=[0.12 -0.05] std=[0.43  0.23]
Bias stats after 2000 steps: energy mean=14.567 std=4.123; CV mean=[0.15 -0.07] std=[0.45  0.25]
...
```

## Tuning the Bias Strength

The default bias strength (`k = 10.0 kJ/mol`) is a starting point. Adjust based on your system:

### Too Weak (`k < 5`)
- Bias energy small compared to thermal fluctuations (~kT ≈ 2.5 kJ/mol at 300 K)
- System behavior similar to unbiased
- **Fix**: Increase `k`

### Too Strong (`k > 50`)
- Bias dominates thermal energy
- System forced into high-CV regions regardless of physical stability
- May sample unphysical structures
- **Fix**: Decrease `k`

### Just Right (`k ≈ 10–20`)
- Bias comparable to thermal energy
- System samples a broader CV range while remaining physically reasonable
- Balance between restraint and thermal exploration

**How to adjust**: Currently set during model export in `backend.py`. Future versions will expose this as a runtime configuration parameter.

## Validation

### Check if Bias is Working

1. **Inspect logs**: Verify bias energy is non-zero and changing over time
   ```
   Bias stats after 1000 steps: energy mean=12.345 std=3.456
   ```

2. **Compare CV distributions**:
   ```python
   # Unbiased: narrow CV distribution
   # Biased: broader CV distribution (but not necessarily better sampling!)
   ```

3. **Monitor acceptance rates** (replica exchange):
   - Biased simulations may have different exchange acceptance
   - Adjust temperature ladder if needed

## Troubleshooting

### "openmm-torch not available"
```bash
conda install -c conda-forge openmm-torch
```

### "Configuration missing required keys"
Add all required keys to `defaults.yaml`:
```yaml
enable_cv_bias: true
bias_mode: harmonic
torch_threads: 4
precision: single
feature_spec_path: feature_spec.yaml
```

### "Feature specification mismatch"
The model's embedded feature spec hash doesn't match your current `feature_spec.yaml`.
- **Cause**: You changed the spec after exporting the model
- **Fix**: Re-export the model or restore the original feature spec

### "Model parameter must be float32"
Your model was trained/exported with mixed precision.
- **Fix**: Re-export ensuring all tensors are `float32`

### "Simulation extremely slow"
- **Check 1**: Set `torch_threads` to 4–8 (not too high!)
- **Check 2**: Use `precision: single` (not `double`)
- **Check 3**: Consider GPU-enabled PyTorch for production

### "Bias energy is zero or NaN"
- **Check 1**: Verify `enable_cv_bias: true` in config
- **Check 2**: Confirm model file exists at specified path
- **Check 3**: Check logs for load-time errors
- **Check 4**: Run benchmark harness to isolate the issue

## Implementation Details

### TorchScript Feature Extraction

The `TorchScriptFeatureExtractor` module computes molecular features entirely within TorchScript:

- **Distances**: Minimum-image convention under PBC
  ```python
  # Fractional coordinates
  frac = positions @ inv_box
  # Wrap to [-0.5, 0.5)
  wrapped = frac - round(frac)
  # Back to Cartesian
  min_image = wrapped @ box
  ```

- **Angles**: Three-body angles with PBC-aware vectors
- **Dihedrals**: Four-body torsion angles using cross products

**Vectorized**: All operations use PyTorch tensor ops (no Python loops).

**PBC-aware**: Each feature can enable/disable PBC handling via the `pbc` flag in the spec.

### Metadata Embedding

The exported TorchScript module carries metadata:
```python
model.feature_spec_sha256       # Hash of feature spec
model.uses_periodic_boundary_conditions  # PBC flag
model.atom_count                # Number of atoms required
```

OpenMM's `system_builder.py` validates these attributes at load time.

## References

- **OpenMM-Torch**: https://github.com/openmm/openmm-torch
- **Deep-TICA**: Schwantes & Pande, *J. Chem. Theory Comput.* (2013)
- **TorchScript**: https://pytorch.org/docs/stable/jit.html
- **PMARLO Documentation**: `CV_REQUIREMENTS.md` for setup and troubleshooting

## Contact

For questions about CV-informed sampling:
- Check simulation logs in `app_output/sims/*/replica_exchange/`
- Review training logs in `app_output/models/training-*/training.log`
- Consult `CV_REQUIREMENTS.md` for setup issues
- Run the benchmark harness to diagnose performance problems
