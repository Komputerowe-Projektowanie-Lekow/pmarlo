# PMARLO: Enhanced Protein Markov State Model Analysis

## Overview

PMARLO (Protein Markov State Model Analysis with Replica Exchange) is an enhanced version of your original protein analysis pipeline. It integrates **Replica Exchange Molecular Dynamics (REMD)** and **TRAM/dTRAM** analysis to provide better conformational sampling and more accurate free energy landscapes.

## What's New

Based on your original pipeline concept, we've added:

### 🔄 **Replica Exchange Molecular Dynamics (REMD)**
- Parallel simulations at multiple temperatures
- Enhanced conformational sampling
- Automatic temperature ladder generation
- Exchange statistics and diagnostics

### 📊 **Enhanced MSM Analysis**
- TRAM/dTRAM for multi-temperature data integration
- 2D/3D free energy surface generation
- Comprehensive state tables with representative structures
- Implied timescales analysis and validation
- Interactive visualization capabilities

### 📋 **Comprehensive Reporting**
- Automated generation of publication-ready plots
- State summary tables (CSV export)
- Representative PDB structures for each state
- Free energy surfaces (PNG and interactive HTML)
- Analysis validation metrics

## Pipeline Comparison

| Stage | Original Pipeline | Enhanced Pipeline |
|-------|------------------|-------------------|
| **1. Build System** | Single protein preparation | Same + enhanced validation |
| **2. Production** | Single-T MD + Metadynamics | **Multi-T REMD + Metadynamics** |
| **3. Data Collection** | Single trajectory | **Multi-temperature demultiplexing** |
| **4. Featurization** | Basic phi clustering | Enhanced feature extraction (phi/psi, distances, contacts) |
| **5. MSM Construction** | Standard MSM | **TRAM/dTRAM enhanced MSM** |
| **6. Analysis** | Basic free energies | **Complete reporting suite** |

## Quick Start

### Basic Usage

```python
# Run the enhanced pipeline
python src/main.py --mode remd

# Compare with original pipeline
python src/main.py --mode compare

# Run specific examples
python src/examples.py --example 3
```

### Command Line Options

```bash
# Available modes
python src/main.py --mode original    # Original single-T pipeline
python src/main.py --mode remd       # New REMD + enhanced MSM pipeline
python src/main.py --mode compare    # Run both for comparison
python src/main.py --mode test       # Test protein preparation only

# Customize parameters
python src/main.py --mode remd --steps 100000 --states 100
```

## Step-by-Step Workflow

### 1. Protein Preparation
```python
from protein import Protein

# Prepare protein structure
protein = Protein("input.pdb", ph=7.0)
protein.save("prepared.pdb")

# Get properties
properties = protein.get_properties(detailed=True)
print(f"Atoms: {properties['num_atoms']}, Residues: {properties['num_residues']}")
```

### 2. Replica Exchange Simulation
```python
from replica_exchange import run_remd_simulation

# Run REMD with custom settings
demux_trajectory = run_remd_simulation(
    pdb_file="prepared.pdb",
    output_dir="remd_output",
    total_steps=100000,
    temperatures=[300, 320, 342, 366, 392],  # Custom T ladder
    use_metadynamics=True
)
```

### 3. Enhanced MSM Analysis
```python
from enhanced_msm import run_complete_msm_analysis

# Complete MSM analysis pipeline
msm = run_complete_msm_analysis(
    trajectory_files=["trajectory.dcd"],
    topology_file="prepared.pdb",
    output_dir="msm_analysis",
    n_clusters=100,
    lag_time=20,
    feature_type="phi_psi"
)
```

## Output Files

The enhanced pipeline generates comprehensive outputs:

### REMD Output (`remd_output/`)
```
remd_output/
├── replica_00.dcd              # Individual replica trajectories
├── replica_01.dcd
├── ...
├── demux_T300K.dcd            # Demultiplexed trajectory at 300K
├── remd_results.pkl           # Exchange statistics
└── bias/                      # Metadynamics bias files
    └── bias_*.npy
```

### MSM Analysis Output (`msm_analysis/`)
```
msm_analysis/
├── msm_analysis_state_table.csv           # State summary table
├── msm_analysis_fes.npy                   # Free energy surface data
├── msm_analysis_transition_matrix.npz     # Transition matrix (sparse)
├── msm_analysis_free_energies.npy         # Free energies per state
├── free_energy_surface.png               # FES plot
├── implied_timescales.png                # Validation plot
├── free_energy_profile.png               # 1D energy profile
├── state_000_representative.pdb          # Representative structures
├── state_001_representative.pdb
└── ...
```

## Key Features

### 🌡️ **Temperature Ladder Optimization**
```python
# Automatic exponential spacing
temperatures = remd._generate_temperature_ladder(
    min_temp=300.0,
    max_temp=450.0,
    n_replicas=8
)
# Result: [300.0, 315.4, 331.6, 348.5, 366.2, 384.8, 404.4, 425.0]
```

### 📈 **Free Energy Surface Generation**
```python
# Generate 2D FES in phi/psi space
fes_data = msm.generate_free_energy_surface(
    cv1_name="phi",
    cv2_name="psi",
    bins=50,
    temperature=300.0
)

# Plot with matplotlib or interactive plotly
msm.plot_free_energy_surface(save_file="fes", interactive=True)
```

### 📊 **State Analysis**
```python
# Get comprehensive state information
state_table = msm.create_state_table()

# Top states by population
top_states = state_table.nlargest(10, 'population')
for _, state in top_states.iterrows():
    print(f"State {state['state_id']}: "
          f"ΔG = {state['free_energy_kJ_mol']:.2f} kJ/mol, "
          f"Pop = {state['population']:.4f}")
```

### 🔍 **Model Validation**
```python
# Implied timescales analysis
msm.compute_implied_timescales(lag_times=range(1, 51))
msm.plot_implied_timescales(save_file="validation")

# Exchange diagnostics for REMD
stats = remd.get_exchange_statistics()
print(f"Exchange acceptance: {stats['overall_acceptance_rate']:.3f}")
```

## Advanced Usage

### Custom Feature Engineering
```python
# Use distance-based features instead of dihedrals
msm.compute_features(feature_type="distances", n_features=100)

# Or contact-based features
msm.compute_features(feature_type="contacts")
```

### Multi-Temperature Analysis (TRAM)
```python
# Analyze multiple replica trajectories simultaneously
msm = EnhancedMSM(
    trajectory_files=["replica_00.dcd", "replica_01.dcd", ...],
    temperatures=[300, 320, 342, 366],  # Enable TRAM
    output_dir="tram_analysis"
)

# Build TRAM-enhanced MSM
msm.build_msm(lag_time=20, method="tram")
```

### Representative Structure Extraction
```python
# Extract and save representative structures
structures = msm.extract_representative_structures(save_pdb=True)

# Each state gets a representative PDB file
# state_000_representative.pdb, state_001_representative.pdb, etc.
```

## Integration with Existing Tools

The enhanced pipeline is designed to work alongside your existing workflow:

### With Your Current Simulation Code
```python
# Use your existing prepare_system and production_run
from simulation import prepare_system, production_run

# Then enhance with REMD
from replica_exchange import ReplicaExchange

# And analyze with enhanced MSM
from enhanced_msm import EnhancedMSM
```

### Backwards Compatibility
```python
# Original workflow still works
from src.main import original_pipeline_with_dg
original_pipeline_with_dg()  # Runs your original pipeline

# OR use the new clean API:
from src import run_pmarlo
results = run_pmarlo("protein.pdb", temperatures=[300, 310, 320], steps=1000)

# New enhanced workflow
from main import run_remd_pipeline
run_remd_pipeline()  # Runs REMD + enhanced MSM
```

## Examples

See `src/examples.py` for detailed usage examples:

1. **Basic REMD** - Simple replica exchange setup
2. **Custom Temperatures** - Manual temperature ladder configuration
3. **Enhanced MSM Analysis** - Complete analysis pipeline
4. **Multi-Temperature TRAM** - Advanced multi-temperature analysis
5. **Protein Preparation** - Structure preparation workflow

Run examples:
```bash
python src/examples.py                    # All examples
python src/examples.py --example 3       # Specific example
```

## Performance Notes

### Computational Requirements
- **REMD**: Scales linearly with number of replicas
- **MSM Analysis**: Memory scales with n_states²
- **Representative structures**: I/O intensive for large systems

### Optimization Tips
- Use stride when loading large trajectories: `msm.load_trajectories(stride=5)`
- Reduce states for initial testing: `n_clusters=30`
- Use shorter lag times for faster convergence: `lag_time=10`

## Troubleshooting

### Common Issues

**1. "No phi dihedrals found"**
```python
# Solution: Check protein structure or use distance features
msm.compute_features(feature_type="distances")
```

**2. "Low exchange acceptance rate"**
```python
# Solution: Adjust temperature spacing
temperatures = remd._generate_temperature_ladder(min_temp=300, max_temp=400)
```

**3. "MSM not converged"**
```python
# Solution: Check implied timescales
msm.compute_implied_timescales()
msm.plot_implied_timescales()
# Increase lag_time if timescales are not flat
```

### File Path Issues
Make sure your file paths are correct:
```python
from pathlib import Path
pdb_file = Path("tests/data/3gd8-fixed.pdb")
assert pdb_file.exists(), f"File not found: {pdb_file}"
```

## Dependencies

Core requirements:
- `openmm` - Molecular dynamics simulation
- `mdtraj` - Trajectory analysis
- `scikit-learn` - Clustering
- `numpy`, `scipy` - Numerical computation
- `matplotlib` - Plotting
- `pandas` - Data analysis

Optional:
- `plotly` - Interactive visualization
- `rdkit` - Chemical property calculation

## Future Enhancements

Potential additions to the pipeline:
- **Deep learning features** using VAMPnets or autoencoders
- **Advanced clustering** with Gaussian mixture models
- **Network analysis** of state connectivity
- **Kinetic rate calculations** between states
- **GPU acceleration** for large-scale analysis

## Support

For issues or questions:
1. Check the examples in `src/examples.py`
2. Review the troubleshooting section above
3. Examine the detailed docstrings in each module
4. Run with `--mode test` to verify basic functionality
