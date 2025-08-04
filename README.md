# PMARLO: Protein Markov State Model Analysis with Replica Exchange

A Python package for protein simulation and Markov state model chain generation, providing an OpenMM-like interface for molecular dynamics simulations.

## Features

- **Simple API**: OpenMM-inspired interface for easy usage
- **Protein Preparation**: Automated PDB cleanup and preparation
- **Replica Exchange**: Enhanced sampling with temperature replica exchange
- **Metadynamics**: Optional biased sampling for enhanced conformational exploration  
- **Markov State Models**: Advanced MSM analysis with TRAM/dTRAM
- **Pipeline Orchestration**: Complete workflow coordination in just a few lines

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/pmarlo.git
cd pmarlo

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

## Quick Start

### Ultra-Simple One-Liner

```python
from pmarlo import run_pmarlo

# Complete analysis in one line
results = run_pmarlo("protein.pdb", temperatures=[300, 310, 320], steps=1000)
```

### Five-Line Usage (OpenMM-style)

```python
from pmarlo import Protein, ReplicaExchange, Simulation, MarkovStateModel, Pipeline

# Setup components
protein = Protein("protein.pdb", ph=7.0)
replica_exchange = ReplicaExchange("protein.pdb", temperatures=[300, 310, 320])  
simulation = Simulation("protein.pdb", temperature=300, steps=1000)
markov_state_model = MarkovStateModel()

# Run complete pipeline
pipeline = Pipeline("protein.pdb")
results = pipeline.run()
```

### Component-by-Component Control

```python
from pmarlo import Pipeline

# Create pipeline with custom settings
pipeline = Pipeline(
    pdb_file="protein.pdb",
    temperatures=[300.0, 310.0, 320.0],
    steps=10000,
    n_states=100,
    use_replica_exchange=True,
    use_metadynamics=True,
    output_dir="my_analysis"
)

# Run complete analysis
results = pipeline.run()

# Access individual components if needed
components = pipeline.get_components()
protein = components["protein"]
replica_exchange = components["replica_exchange"]
```

## Package Structure

```
pmarlo/
├── src/
│   ├── __init__.py              # Main package exports
│   ├── pipeline.py              # Pipeline orchestration
│   ├── protein/
│   │   ├── __init__.py
│   │   └── protein.py           # Protein class
│   ├── replica_exchange/
│   │   ├── __init__.py
│   │   └── replica_exchange.py  # ReplicaExchange class
│   ├── simulation/
│   │   ├── __init__.py
│   │   └── simulation.py        # Simulation class
│   ├── markov_state_model/
│   │   ├── __init__.py
│   │   └── markov_state_model.py # MarkovStateModel class
│   └── manager/
│       ├── __init__.py
│       └── checkpoint_manager.py # Checkpoint management
├── examples/
│   └── simple_usage.py          # Usage examples
├── setup.py                     # Package setup
└── README.md                    # This file
```

## Examples

### Running the Demos

```bash
# Show the simple API demonstration
python -m src.main --mode simple

# Run a minimal demo (requires test files)
python -m src.main --mode demo

# Run legacy REMD pipeline
python -m src.main --mode remd --steps 1000

# Compare different methods
python -m src.main --mode compare
```

### Advanced Usage

```python
from pmarlo import Pipeline

# Custom pipeline for advanced users
pipeline = Pipeline(
    pdb_file="complex_protein.pdb",
    temperatures=[298.0, 308.0, 318.0, 328.0],  # 4 replicas
    n_replicas=4,
    steps=100000,  # Longer simulation
    n_states=200,  # More MSM states
    use_replica_exchange=True,
    use_metadynamics=True,
    output_dir="advanced_analysis"
)

# Run with checkpointing
results = pipeline.run()

# Access detailed results
print(f"Analysis completed: {results['pipeline']['status']}")
print(f"Output directory: {results['pipeline']['output_dir']}")

if 'replica_exchange' in results:
    print(f"REMD trajectories: {results['replica_exchange']['trajectory_files']}")

if 'msm' in results:
    print(f"MSM analysis: {results['msm']['output_dir']}")
```

## API Reference

### Classes

- **`Protein`**: Handles protein preparation and cleanup
- **`ReplicaExchange`**: Manages replica exchange molecular dynamics
- **`Simulation`**: Single-temperature MD simulations with optional metadynamics
- **`MarkovStateModel`**: Advanced MSM analysis and visualization
- **`Pipeline`**: Orchestrates all components for complete workflow

### Functions

- **`run_pmarlo()`**: One-line function for complete analysis

## Dependencies

- numpy >= 1.18.0
- scipy >= 1.5.0
- matplotlib >= 3.2.0
- pandas >= 1.1.0
- scikit-learn >= 0.23.0
- mdtraj >= 1.9.0
- openmm >= 7.5.0
- pdbfixer >= 1.7
- rdkit >= 2020.09.1

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Citation

If you use PMARLO in your research, please cite:

```bibtex
@software{pmarlo2024,
  title={PMARLO: Protein Markov State Model Analysis with Replica Exchange},
  author={PMARLO Development Team},
  year={2024},
  url={https://github.com/yourusername/pmarlo}
}
```

## Support

- Documentation: [https://pmarlo.readthedocs.io/](https://pmarlo.readthedocs.io/)
- Issues: [https://github.com/yourusername/pmarlo/issues](https://github.com/yourusername/pmarlo/issues)
- Discussions: [https://github.com/yourusername/pmarlo/discussions](https://github.com/yourusername/pmarlo/discussions)