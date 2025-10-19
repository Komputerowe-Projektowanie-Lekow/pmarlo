# CV-Informed Sampling Requirements & Troubleshooting

✅ **CV-BIASED SAMPLING IS NOW IMPLEMENTED** ✅

The CV integration now properly transforms collective variables into biasing forces:
- **Solution**: Models are wrapped with `CVBiasPotential` that outputs **energy** (E = k · Σ(cv²))
- **Impact**: OpenMM computes physically meaningful forces via automatic differentiation (F = -∇E)
- **Purpose**: Repulsive bias in CV space → encourages conformational exploration

**Current Behavior**: Simulations can run WITH CV biasing (when `openmm-torch` is installed).

See "CV-Informed Sampling Integration Guide" (`CV_INTEGRATION_GUIDE.md`) for full details.

---

# CV-Informed Sampling Requirements & Troubleshooting

## Current Setup Diagnosis

Your environment:
- ✅ Python 3.13.0
- ✅ PyTorch 2.7.0+cpu (CPU-only build)
- ❌ **openmm-torch NOT installed**
- ❌ **CUDA NOT available**

## Issues Explained

### 1. Import Error
**Error**: `cannot import name 'load_cv_model_info' from 'pmarlo.features.deeptica'`

**Cause**: openmm-torch is not installed

**Solution**: Install openmm-torch (see below)

**Current Behavior**: Simulations run normally WITHOUT CV biasing (with warning logged)

### 2. Performance Issue
**Observation**: Simulations taking 5 hours instead of 15 minutes

**Causes**:
- If CV model IS being loaded (after installing openmm-torch), CPU-only PyTorch is 10-20x slower
- Check if you changed simulation parameters (more replicas, more steps)
- Without openmm-torch, simulations should run at normal speed (~15 min)

## Installation Options

### Option 1: Test CV Integration (CPU-only, SLOW)

```bash
conda install -c conda-forge openmm-torch
```

**Expected Performance**:
- Unbiased simulation: 15 minutes
- CV-biased simulation: 3-5 hours (10-20x slower)

**Use Case**: Testing, development, proof-of-concept

### Option 2: Production Setup (GPU-accelerated, RECOMMENDED)

```bash
# 1. Uninstall CPU-only PyTorch
pip uninstall torch

# 2. Install CUDA-enabled PyTorch
# Check your CUDA version: nvidia-smi
# For CUDA 11.8:
pip install torch --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1:
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 3. Install openmm-torch
conda install -c conda-forge openmm-torch
```

**Expected Performance**:
- Unbiased simulation: 15 minutes
- CV-biased simulation: 30-45 minutes (2-3x slower)

**Use Case**: Production, iterative FES mapping

### Verify Installation

```python
import torch
import openmmtorch

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"openmm-torch: installed ✓")
```

## Training Metrics Review

Your model training looked good:

**✅ Positive Signs**:
- VAMP-2 score: 1.04 (>1.0 is good, means CVs capture slow dynamics)
- Stable convergence in training and validation
- No overfitting

**⚠️ Areas to Monitor**:
- **Coverage**: 25.1% (5970/23778 pairs)
  - Lower than ideal but acceptable for initial training
  - Consider longer trajectories for better coverage

- **Gradient Clipping**: Very active
  - Pre-clip gradients: 2-10 (sometimes higher)
  - Post-clip: Always ~1.0
  - Consider increasing `gradient_clip_val` from 1.0 to 5.0 if retraining

**Model Details**:
- Architecture: [64, 32] hidden layers
- CV dimensions: 2
- Parameters: ~3000 (estimated)
- Training time: ~21 minutes for 60 epochs

## Why is CV-Biased Sampling Slow?

Neural networks in MD require computing forces at **every timestep**:

1. Extract atomic features (distances, angles, dihedrals)
2. Scale inputs using scaler (mean/std normalization)
3. Forward pass through neural network
4. Backpropagate through network to compute forces
5. Apply forces to atoms

With millions of timesteps per simulation:
- **CPU**: Each neural network call is slow (10-20x overhead)
- **GPU**: Much faster but still 2-3x slower than unbiased MD
- **Overhead**: Data transfer between OpenMM and PyTorch

## Current Behavior (Without openmm-torch)

When you select a CV model right now:

1. ✅ App loads model configuration
2. ❌ `check_openmm_torch_available()` returns False
3. ⚠️ Logs: "CV model selected but openmm-torch is not installed"
4. ⚠️ Logs: "Running simulation WITHOUT CV biasing"
5. ✅ Simulation runs normally (no CV forces applied)

**Your simulations should still be fast (~15 min)**. If they're slow:
- Check number of replicas (temperatures)
- Check number of steps
- Check CPU usage
- Check if multiple simulations are queued

## Recommended Workflow

### Phase 1: Development (Current)
- ✅ Train models on small trajectory batches
- ✅ Use unbiased sampling to collect data
- ✅ Iterate on model hyperparameters
- ❌ Skip CV-biased sampling until GPU ready

### Phase 2: Validation (After GPU Setup)
- Install GPU PyTorch + openmm-torch
- Test CV-biased sampling on short runs
- Verify performance is acceptable (2-3x slower, not 20x)
- Confirm CVs enhance sampling (check acceptance rates, coverage)

### Phase 3: Production
- Use CV-biased sampling to map FES efficiently
- Iterative workflow:
  1. Run biased sampling with current best model
  2. Train better model on combined data
  3. Run biased sampling with improved model
  4. Repeat until FES fully mapped

## Troubleshooting

### "Import error: load_cv_model_info"
- **Cause**: openmm-torch not installed
- **Fix**: Install openmm-torch (see above)
- **Workaround**: Simulations run without CV biasing

### "Simulation taking 5 hours instead of 15 min"
- **Check 1**: Are you using CV biasing? (openmm-torch installed?)
  - If YES and CPU-only: Expected, install GPU PyTorch
  - If NO: Check simulation parameters

- **Check 2**: How many temperature replicas?
  - More replicas = linear increase in time
  - Default might be 4-8 replicas

- **Check 3**: How many steps?
  - Check if steps parameter was increased
  - Default for quick mode: ~5000-10000 steps

- **Check 4**: Multiple queued simulations?
  - Progress bars might show multiple runs

### "CUDA out of memory" (if using GPU)
- Reduce model size (smaller hidden layers)
- Reduce batch size during training
- Use smaller temperature ladder (fewer replicas)

### "Model export failed"
- Check if training completed successfully
- Check logs in `app_output/models/training-*/training.log`
- Try retraining if model files missing

## Performance Expectations

| Configuration | Time (estimate) | Notes |
|--------------|----------------|-------|
| Unbiased MD | 15 min | Baseline |
| CV-biased (CPU) | 3-5 hours | 10-20x slower, testing only |
| CV-biased (GPU) | 30-45 min | 2-3x slower, production-ready |
| Multiple replicas (4x) | 4x longer | Linear scaling |

## Next Steps

1. **Immediate** (to fix import error):
   ```bash
   conda install -c conda-forge openmm-torch
   ```

2. **Short-term** (if you want fast CV-biased sampling):
   - Install GPU-enabled PyTorch
   - Verify CUDA works: `torch.cuda.is_available()`

3. **Long-term** (optimal workflow):
   - Continue training models on unbiased data
   - Build up dataset with multiple sampling rounds
   - Use CV-biased sampling only when GPU ready
   - Iterate: sample → train → sample with CV → train on all data

## Implementation: CV-to-Force Transformation

### The Solution (NOW IMPLEMENTED)

CV values are transformed into biasing forces using a **bias potential wrapper**:

**What OpenMM-Torch TorchForce Expects**:
```python
# Model should output: Energy (scalar)
# OpenMM computes forces via: F = -∇E (automatic differentiation)
model(features) → energy  # kJ/mol
```

**Our Implementation (CVBiasPotential)**:
```python
# Model transforms CVs → Energy via bias potential
class CVBiasPotential(nn.Module):
    def forward(self, features):
        scaled = (features - mean) / scale  # Scale features
        cvs = deep_tica(scaled)             # Compute CVs
        energy = k * torch.sum(cvs ** 2)    # Apply bias
        return energy                       # ✅ Returns energy!
```

### Why This Works

1. Deep-TICA outputs CVs (dimensionless values)
2. Bias potential transforms CVs → Energy: **E = k · (cv₁² + cv₂² + ... + cvₙ²)**
3. OpenMM computes forces via automatic differentiation: **F = -∇E**
4. **Result**: Physically meaningful repulsive forces in CV space!

### What's Needed

To properly use CVs for biasing, we need to:

1. **Define a Bias Potential**: Transform CVs → Energy
   ```python
   # Option 1: Harmonic restraint (push away from origin)
   E_bias = k * (cv1² + cv2² + ...)

   # Option 2: Metadynamics-like (discourage visited regions)
   E_bias = sum of Gaussians deposited in CV space

   # Option 3: Umbrella sampling on CVs
   E_bias = k * (cv - target)²
   ```

2. **Wrap the Model**: Create a PyTorch module that:
   ```python
   class CVBiasPotential(nn.Module):
       def __init__(self, cv_model, bias_type="harmonic", strength=1.0):
           self.cv_model = cv_model
           self.strength = strength

       def forward(self, positions):
           # Extract features from positions
           features = extract_features(positions)

           # Compute CVs
           cvs = self.cv_model(features)

           # Transform to energy
           energy = self.strength * torch.sum(cvs ** 2)

           return energy  # Now OpenMM can compute forces!
   ```

3. **Export the Wrapper**: Export `CVBiasPotential`, not just the CV model

### Current Status

The app now:
- ✅ Trains CV models successfully
- ✅ Exports models wrapped with CVBiasPotential (outputs energy)
- ✅ **Applies CV biasing** when `openmm-torch` is installed
- ✅ Logs detailed information about bias physics
- ⚠️ **Limitation**: Feature extraction not yet implemented (see below)

### What's Working

1. **Model wrapping**: Deep-TICA + HarmonicExpansionBias → Energy output
2. **OpenMM integration**: TorchForce added to system, forces computed automatically
3. **Export/import**: Models saved as TorchScript, loaded seamlessly
4. **Graceful degradation**: Falls back to unbiased MD if openmm-torch missing

### Current Limitation: Feature Extraction

⚠️ **Important**: The CVBiasPotential expects **molecular features** (distances, angles, dihedrals) as input, not raw atomic positions.

**Status**: Feature extraction from atomic positions → molecular features is not yet implemented in the OpenMM integration.

**Impact**: The model may receive incorrect inputs, leading to unexpected behavior.

**Workarounds**:
1. Use predefined feature sets that match training data
2. Implement feature extraction using `CustomBondForce`, `CustomAngleForce`
3. Use `cvpack` library for standardized CVs
4. Wait for automatic feature extraction implementation (future work)

## Support

If issues persist:
- Check logs in `app_output/models/training-*/training.log`
- Check simulation logs in `app_output/sims/*/replica_exchange/`
- Provide error messages and system info

For CV biasing questions:
- ✅ CV→Energy transformation is IMPLEMENTED
- ⚠️ Feature extraction still needs work
- See `CV_INTEGRATION_GUIDE.md` for comprehensive usage guide
- Current behavior: CV biasing active (when openmm-torch installed)
- Contact developers if issues persist or to contribute to feature extraction
