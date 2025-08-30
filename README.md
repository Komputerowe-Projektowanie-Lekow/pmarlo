# PMARLO: Protein Markov State Model Analysis with Replica Exchange

[![PyPI Version][pypi-image]][pypi-url]
[![Build Status][build-image]][build-url]
[![Python Versions][versions-image]][versions-url]
[![][stars-image]][stars-url]
[![License][license-image]][license-url]
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/Komputerowe-Projektowanie-Lekow/pmarlo)



A Python package for protein simulation and Markov state model chain generation, providing an OpenMM-like interface for molecular dynamics simulations.

## Features

- **Protein Preparation**: Automated PDB cleanup and preparation
- **Replica Exchange**: Enhanced sampling with temperature replica exchange
- **Simulation**: Single-temperature MD simulations
- **Markov State Models**: MSM analysis
- **Pipeline Orchestration**: Complete workflow coordination

## Installation

```bash
# From PyPI (recommended)
pip install pmarlo

# From source (development)
git clone https://github.com/Komputerowe-Projektowanie-Lekow/pmarlo.git
cd pmarlo
pip install -e .
```

- Python: 3.11–3.12
- Optional: `pip install pmarlo[fixer]` to include `pdbfixer` (only available on Python < 3.12)
- ML CVs (Deep-TICA): `pip install pmarlo[mlcv]` to enable training with
  `mlcolvar` + `torch`. For deployment in PLUMED, ensure PLUMED ≥ 2.9 is built
  with the `pytorch` module so `PYTORCH_MODEL` can load TorchScript models.


## Documentation
Documentation was made using cognition powered by Devin. Here is the link https://deepwiki.com/Komputerowe-Projektowanie-Lekow/pmarlo. It will be updated weekly whenever new features, bug fixes, or other changes are made.


## Quickstart

```python
from pmarlo.pipeline import run_pmarlo

results = run_pmarlo(
    pdb_file="protein.pdb",
    temperatures=[300, 310, 320],
    steps=1000,
    n_states=50,
)
```

### Clean API example

```python
from pmarlo import Protein, ReplicaExchange, RemdConfig, Simulation, Pipeline

# Prepare protein
protein = Protein("protein.pdb", ph=7.0)

# Replica Exchange (auto-setup plans reporter stride automatically)
remd = ReplicaExchange.from_config(
    RemdConfig(
        pdb_file="protein.pdb",
        temperatures=[300.0, 310.0, 320.0],
        auto_setup=True,
        dcd_stride=10,
    )
)

# Single-temperature simulation (optional)
simulation = Simulation("protein.pdb", temperature=300.0, steps=1000)

# Full pipeline
pipeline = Pipeline(
    pdb_file="protein.pdb",
    temperatures=[300.0, 310.0, 320.0],
    steps=1000,
    auto_continue=True,
)
results = pipeline.run()
```

## Complexity

Currently based on the pygount, the amount of lines of code is ~ 6000 lines, which is quite big number, where we can make package less bloated in the next updates.

### v0.0.23

- utilities - Files: 1 | Code: 102 | Comment: 15
- tests - Files: 8 | Code: 679 | Comment: 229
- src - Files: 24 | Code: 5483 | Comment: 1176
- example_programs - Files: 3 | Code: 365 | Comment: 92

### v0.0.33

- pmarlo\utilities - Files: 1 | Code: 105 | Comment: 15
- pmarlo\tests - Files: 52 | Code: 2012 | Comment: 210
- pmarlo\src - Files: 72 | Code: 10237 | Comment: 1936
- pmarlo\example_programs - Files: 4 | Code: 188 | Comment: 27


### v0.0.35

- pmarlo\utilities - Files: 1 | Code: 105 | Comment: 15
- pmarlo\tests - Files: 65 | Code: 2575 | Comment: 245
- pmarlo\src - Files: 87 | Code: 11804 | Comment: 2229
- pmarlo\example_programs - Files: 6 | Code: 361 | Comment: 50


## Package Structure

```
pmarlo/
├── src/pmarlo/
│   ├── __init__.py                  # Public API (Protein, ReplicaExchange, RemdConfig, ...)
│   ├── pipeline.py                  # High-level Pipeline + run_pmarlo helper
│   ├── main.py                      # CLI entry-point (installed as `pmarlo`)
│   ├── protein/
│   │   └── protein.py               # Protein preparation
│   ├── replica_exchange/
│   │   ├── config.py                # RemdConfig (immutable settings)
│   │   └── replica_exchange.py      # ReplicaExchange implementation
│   ├── simulation/
│   │   └── simulation.py            # Single-temperature MD
│   ├── markov_state_model/
│   │   ├── enhanced_msm.py          # Enhanced MSM orchestrator
│   │   └── _*.py                    # Modular MSM mixins
│   ├── features/ | reduce/ | cluster/ | fes/ | states/
│   │   └── ...                      # Analysis and utilities
│   ├── reporting/                   # Plots and exports
│   └── manager/checkpoint_manager.py# Checkpoint management
```

## Verification and CLI

```bash
# Show CLI options
pmarlo --help

# Run a minimal example
pmarlo --mode simple
```

Smoke test in Python:

```bash
python - <<'PY'
import pmarlo
print("PMARLO", pmarlo.__version__)
PY
```

## Dependencies

- numpy >= 1.24, < 2.3
- scipy >= 1.10, < 2.0
- matplotlib >= 3.6, < 4.0
- pandas >= 1.5, < 3.0
- scikit-learn >= 1.2, < 2.0
- mdtraj >= 1.9, < 2.0
- openmm >= 8.0, < 9.0
- rdkit >= 2024.03.1, < 2025.0
- psutil >= 5.9, < 6.1
- pygount >= 2.6, < 3.2
- deeptime >= 0.4.5, < 0.5

Optional on Python < 3.12:
- pdbfixer (install via extra: `pmarlo[fixer]`)

## Progress Events

PMARLO can emit unified progress events via a callback argument to selected APIs. The callback signature is `callback(event: str, info: Mapping[str, Any]) -> None`.

Accepted kwarg aliases: `progress_callback`, `callback`, `on_event`, `progress`, `reporter`.

Events overview:

- setup: elapsed_s; message
- equilibrate: elapsed_s, current_step, total_steps; eta_s
- simulate: elapsed_s, current_step, total_steps; eta_s
- exchange: elapsed_s; sweep_index, n_replicas, acceptance_mean, acceptance_per_pair, temperatures
- write_output: elapsed_s; artifacts
- finished: elapsed_s, status
- aggregate_begin: elapsed_s, total_steps, plan_text
- aggregate_step_start: elapsed_s, index, total_steps, step_name
- aggregate_step_end: elapsed_s, index, total_steps, step_name, duration_s, current_step, total_steps
- aggregate_end: elapsed_s, status

<!-- Badges: -->

[pypi-image]: https://img.shields.io/pypi/v/pmarlo
[pypi-url]: https://pypi.org/project/pmarlo/
[build-image]: https://github.com/Komputerowe-Projektowanie-Lekow/pmarlo/actions/workflows/publish.yml/badge.svg
[build-url]: https://github.com/Komputerowe-Projektowanie-Lekow/pmarlo/actions/workflows/publish.yml
[versions-image]: https://img.shields.io/pypi/pyversions/pmarlo
[versions-url]: https://pypi.org/project/pmarlo/
[stars-image]: https://img.shields.io/github/stars/Komputerowe-Projektowanie-Lekow/pmarlo
[stars-url]: https://github.com/Komputerowe-Projektowanie-Lekow/pmarlo
[license-image]: https://img.shields.io/pypi/l/pmarlo
[license-url]: https://github.com/Komputerowe-Projektowanie-Lekow/pmarlo/blob/main/LICENSE
