# CV Biasing Performance Issue - CRITICAL

## Problem Summary
The CV-biased sampling implementation (added in recent commits) causes catastrophic performance degradation, making simulations 10-100x slower than unbiased simulations.

## Root Causes

### 1. Feature Extraction Not Implemented
**Location**: `src/pmarlo/replica_exchange/system_builder.py:92-105`

The `TorchForce` is added to the system, but:
- The CV model expects **molecular features** (distances, angles, dihedrals) as input
- `TorchForce` by default passes **raw atomic positions** (3N floats for N atoms)
- The model is either:
  - Getting wrong input and producing nonsense forces
  - OR trying to compute features in Python on every MD step (extremely slow)

```python
# Current code (BROKEN):
cv_force = TorchForce(model)  # model expects features, gets positions!
```

### 2. PyTorch on CPU
**Location**: Multiple files

- Simulations with `random_seed` set use Reference (CPU) platform for determinism
- PyTorch runs on CPU → 10-20x slower than GPU
- **Every single MD integration step** calls the PyTorch model
- For a 50,000 step simulation with 3 replicas = 150,000 PyTorch forward passes on CPU

### 3. No Optimization
- No feature caching
- No batching of replica evaluations
- Forces recomputed from scratch at every step

## Performance Impact

### Observed:
- **Equilibration (100 steps)**: ~30+ seconds per replica (should be <1 second)
- **Expected total runtime** for 50,000 steps: **Hours to days** (should be minutes)

### Calculation:
- Normal simulation: ~1000-5000 steps/second on CPU
- With CV biasing: ~3-10 steps/second on CPU
- **Slowdown: 100-1000x**

## Solutions

### Immediate (DONE)
✅ Disabled CV biasing in `app.py` by setting `cv_model_bundle=None`

### Short-term (Required for Production)

#### Option A: Proper Feature Extraction in OpenMM
Implement feature extraction using OpenMM's custom forces:

```python
# Pseudo-code
from openmm import CustomBondForce, CustomAngleForce, CustomTorsionForce

# Define features in OpenMM
distance_force = CustomBondForce("r")
angle_force = CustomAngleForce("theta")
dihedral_force = CustomTorsionForce("theta")

# Add feature atoms
for feature_spec in feature_definitions:
    if feature_spec.type == "distance":
        distance_force.addBond(i, j)
    elif feature_spec.type == "angle":
        angle_force.addAngle(i, j, k)
    # etc.

# Combine features → TorchForce
# This requires openmm-torch to support feature extraction callbacks
```

#### Option B: GPU-Accelerated PyTorch
1. Install CUDA-enabled PyTorch
2. Ensure TorchForce uses GPU for inference
3. Add batch processing for multiple replicas
4. Still need to fix feature extraction!

#### Option C: Precompute CVs
For small systems, precompute CV values on a grid and use interpolation:
```python
# Expensive: Train CV model
cv_model.eval()

# One-time: Compute CV grid
cv_grid = precompute_cv_grid(topology, cv_model, resolution=0.1)

# Fast: Lookup during simulation
cv_value = interpolate_cv(positions, cv_grid)
```

### Long-term (Recommended)

#### Use PLUMED for CV Biasing
PLUMED is battle-tested and optimized for this:

```python
# Use PLUMED instead of custom TorchForce
from openmm import PlumedForce

plumed_script = """
# Define CVs using PLUMED's optimized implementations
d1: DISTANCE ATOMS=1,10
a1: ANGLE ATOMS=5,10,15
t1: TORSION ATOMS=1,5,10,15

# Apply metadynamics bias
METAD ARG=d1,a1,t1 HEIGHT=1.0 PACE=500 SIGMA=0.1,0.1,0.1
"""

plumed_force = PlumedForce(plumed_script)
system.addForce(plumed_force)
```

## Files Affected

### Modified (Need Fixing):
- `src/pmarlo/replica_exchange/system_builder.py` - Broken TorchForce integration
- `src/pmarlo/features/deeptica/cv_bias_potential.py` - CV model expects features
- `src/pmarlo/features/deeptica/export.py` - Export function doesn't validate inputs
- `example_programs/app_usecase/app/backend.py` - Backend passes CV model

### Disabled (Temporary):
- `example_programs/app_usecase/app/app.py` - CV biasing disabled

## Testing Checklist

Before re-enabling CV biasing:
- [ ] Verify feature extraction works correctly
- [ ] Benchmark: CV-biased simulation should be <2x slower than unbiased
- [ ] Verify forces are physically reasonable
- [ ] Test on GPU with CUDA-enabled PyTorch
- [ ] Add integration test for CV-biased REMD
- [ ] Document performance characteristics in CV_INTEGRATION_GUIDE.md

## References
- `example_programs/app_usecase/app/CV_INTEGRATION_GUIDE.md` - Original integration guide
- `src/pmarlo/features/deeptica/cv_bias_potential.py` - CV model implementation
- OpenMM TorchForce docs: http://docs.openmm.org/latest/userguide/application/05_creating_ffs.html#torchforce

## Status
- **Current**: CV biasing DISABLED (2025-10-19)
- **Blocker**: Feature extraction not implemented
- **Priority**: P1 (blocks production use of CV-informed sampling)
