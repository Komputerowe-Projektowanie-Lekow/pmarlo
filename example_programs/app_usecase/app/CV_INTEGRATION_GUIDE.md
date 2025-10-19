# CV-Informed Sampling Integration Guide

## Overview

The `pmarlo` package now supports **CV-informed sampling** using trained Deep-TICA models to bias molecular dynamics simulations. This enables enhanced sampling by applying repulsive forces in collective variable (CV) space, encouraging exploration of diverse conformational states.

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

The model is automatically exported with a **CVBiasPotential wrapper**:

```
app_output/models/training-20250117-120000/
├── deeptica_cv_model.pt          # TorchScript model (outputs energy)
├── deeptica_cv_model_scaler.npz  # Scaler parameters (embedded in model)
├── deeptica_cv_model_config.json # Configuration
└── deeptica_cv_model_metadata.json # Usage info
```

**Key Innovation**: The exported model is NOT just the Deep-TICA network. It's wrapped in a `CVBiasPotential` that:

1. Takes molecular features as input
2. Scales features using fitted scaler
3. Computes CVs using Deep-TICA
4. Applies bias potential: **E_bias = k · Σ(cv_i²)**
5. Returns energy (kJ/mol)

### 3. Simulation Phase

OpenMM uses the exported model to compute biasing forces:

```python
from openmmtorch import TorchForce
import torch

# Load the CVBiasPotential model
model = torch.jit.load("deeptica_cv_model.pt")

# Add as custom force
cv_force = TorchForce(model)
system.addForce(cv_force)

# OpenMM automatically computes forces via automatic differentiation:
# F = -∇E_bias
```

## Physics of the Bias

### Harmonic Expansion Bias

The CVBiasPotential applies a **harmonic expansion bias**:

```
E_bias = k · (cv₁² + cv₂² + ... + cvₙ²)

where:
- k = bias strength (default: 10.0 kJ/mol)
- cv_i = i-th collective variable value
```

### Why This Works for Conformational Search

1. **Repulsive in CV space**: Positive energy when CVs ≠ 0
2. **Pushes away from reference**: System explores states with non-zero CV values
3. **Encourages diversity**: Higher CV values = more diverse conformations
4. **Physically meaningful**: Energy → Forces via calculus (F = -∇E)

### Example

If CV1 represents RMSD from reference structure:
- Small CV1 → Low bias → System comfortable near reference
- Large CV1 → High bias → Pushes to explore further away

**Result**: System explores a wider range of conformations than unbiased MD.

## Architecture

### CVBiasPotential Module

```python
class CVBiasPotential(nn.Module):
    def __init__(self, cv_model, scaler_mean, scaler_scale, bias_strength=10.0):
        # Wraps Deep-TICA model with bias potential
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # 1. Scale features
        scaled = (features - self.scaler_mean) / self.scaler_scale
        
        # 2. Compute CVs
        cvs = self.cv_model(scaled)
        
        # 3. Apply bias
        energy = self.bias_strength * torch.sum(cvs ** 2)
        
        return energy  # OpenMM computes F = -∇energy
```

### Integration Flow

```
Molecular Structure (PDB)
         ↓
Extract Features (distances, angles) ← ⚠️ NOT YET IMPLEMENTED
         ↓
CVBiasPotential.forward(features)
    ├─ Scale features
    ├─ Compute CVs (Deep-TICA)
    └─ Apply bias (E = k·Σ(cv²))
         ↓
    Energy (kJ/mol)
         ↓
OpenMM: F = -∇E  (automatic differentiation)
         ↓
    Biasing Forces
         ↓
Enhanced MD Simulation
```

## Current Limitations

⚠️ **Feature Extraction Not Implemented**

The CVBiasPotential expects **molecular features** (distances, angles, dihedrals) as input, not raw atomic positions. Currently, this feature extraction is not implemented in the OpenMM integration.

**Workaround Options**:
1. Use `mdtraj` to precompute features and pass them to the model
2. Implement feature extraction using OpenMM's `CustomBondForce` / `CustomAngleForce`
3. Use the `cvpack` library for predefined CVs

**Future Work**: Implement automatic feature extraction that matches the training data.

## Requirements

### Essential
- `openmm-torch`: Interface between PyTorch and OpenMM
  ```bash
  conda install -c conda-forge openmm-torch
  ```

### Recommended
- **CUDA-enabled PyTorch**: For acceptable performance
  ```bash
  # Check current setup
  python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
  
  # Install CUDA version (if needed)
  # See: https://pytorch.org/get-started/locally/
  ```

### Performance Impact

| Configuration | Speed vs Unbiased MD |
|--------------|----------------------|
| GPU (CUDA) | ~1.2-2x slower | 
| CPU only | ~10-20x slower |

**Why slower?**: Each MD step requires:
1. Feature extraction from positions
2. Forward pass through neural network
3. Backpropagation to compute gradients w.r.t. positions
4. Force evaluation

## Usage in the App

### Step 1: Train Model
1. Go to "Model Training" tab
2. Configure training parameters
3. Click "Train Deep-TICA model"
4. Wait for training to complete
5. Model is automatically exported with CVBiasPotential wrapper

### Step 2: Run CV-Informed Sampling
1. Go to "Sampling" tab
2. Enable "Use trained CV model to inform sampling"
3. Select your trained model
4. Configure simulation parameters
5. Click "Run replica exchange"

### Step 3: Monitor
- Check logs for CV bias information
- Simulation logs show bias energy contributions
- Analyze trajectory for enhanced sampling

## Tuning the Bias Strength

The default bias strength (k = 10.0 kJ/mol) is a starting point. Adjust based on:

### Too Weak (k < 5)
- System doesn't explore enough
- Conformational coverage similar to unbiased
- **Fix**: Increase k

### Too Strong (k > 50)
- System explores too aggressively
- May sample unphysical states
- Energy barriers overwhelmed
- **Fix**: Decrease k

### Just Right (k ≈ 10-20)
- Enhanced exploration of relevant conformations
- System spends time in biophysically meaningful states
- Free energy landscape well-sampled

**How to adjust**: Currently hardcoded in `backend.py`. Future versions will have UI control.

## Validation

### Check if Bias is Working

1. **Compare trajectories**:
   ```python
   # Unbiased: cluster in few conformations
   # CV-biased: spread across CV space
   ```

2. **Monitor CV values**:
   ```python
   # Should see larger CV values over time
   # Indicates exploration away from reference
   ```

3. **Free energy landscape**:
   ```python
   # Should be more evenly sampled
   # Fewer deep wells, more exploration
   ```

## Troubleshooting

### "openmm-torch not available"
```bash
conda install -c conda-forge openmm-torch
```

### "PyTorch running on CPU"
- Install CUDA-enabled PyTorch
- Check GPU drivers
- Accept slower performance (10-20x)

### "CV model files not found"
- Ensure training completed successfully
- Check `checkpoint_dir` exists
- Verify `.pt` file was created

### "Simulation extremely slow"
- Check PyTorch CUDA availability
- Consider using CPU-only for testing (small systems)
- For production, GPU is essential

### "Forces seem wrong"
- **Feature extraction not implemented**: This is a known limitation
- Model receives wrong inputs → meaningless forces
- **Fix**: Wait for feature extraction implementation or implement manually

## Future Enhancements

1. **Automatic feature extraction**: Match training data features
2. **Tunable bias strength**: UI control for k parameter
3. **Alternative bias types**: Metadynamics-on-CVs, umbrella sampling
4. **Adaptive biasing**: Adjust k based on CV coverage
5. **Multi-model ensembles**: Combine multiple CV models
6. **Real-time CV monitoring**: Plot CV trajectories during simulation

## References

- OpenMM-Torch: https://github.com/openmm/openmm-torch
- Deep-TICA: https://mlcolvar.readthedocs.io/
- Enhanced Sampling: Valsson & Parrinello, Phys. Rev. Lett. (2014)
- VAMP-2 Score: Wu & Noé, J. Chem. Phys. (2020)

## Contact

For questions about CV-informed sampling:
- Check logs in `app_output/sims/*/replica_exchange/`
- Review training logs in `app_output/models/training-*/training.log`
- Consult `CV_REQUIREMENTS.md` for setup issues

