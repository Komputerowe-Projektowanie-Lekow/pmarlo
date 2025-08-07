# PMARLO: Protein Markov State Model Analysis with Replica Exchange

A Python package for protein simulation and Markov state model chain generation, providing an OpenMM-like interface for molecular dynamics simulations.

## Features

- **Protein Preparation**: Automated PDB cleanup and preparation
- **Replica Exchange**: Enhanced sampling with temperature replica exchange
- **Simulation**: Single-temperature MD simulations
- **Markov State Models**: MSM analysis
- **Pipeline Orchestration**: Complete workflow coordination

## Installation

```bash
# Clone the repository
git clone https://github.com/Komputerowe-Projektowanie-Lekow/pmarlo.git
cd pmarlo

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

## Verified Usage Example

The following example demonstrates the verified functionality of PMARLO:

```python
from pmarlo import Protein, ReplicaExchange, Simulation, MarkovStateModel, Pipeline

# Initialize components
protein = Protein("protein.pdb", ph=7.0, auto_prepare=True)
replica_exchange = ReplicaExchange("protein.pdb", temperatures=[300, 310, 320], auto_setup=False)
replica_exchange.setup_replicas()
simulation = Simulation("protein.pdb", temperature=300, steps=1000)
markov_state_model = MarkovStateModel()

# Run complete pipeline
pipeline = Pipeline(
    "protein.pdb",
    temperatures=[300, 310, 320],
    steps=1000,
    auto_continue=True
)
results = pipeline.run()
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
```

## Verification

You can verify your PMARLO installation using the provided verification script:

```bash
python verify_pmarlo.py
```

This will test:
1. Component initialization (Protein, ReplicaExchange, Simulation, MarkovStateModel)
2. Basic pipeline execution with default parameters

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
