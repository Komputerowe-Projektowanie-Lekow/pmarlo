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

## Free Energy Surface/Transition Matrix

This is the animation of the FES/TM generated with specific amount of shards(dataset units that could be combined to make the models better or produce the analysis artifact)

Those were generated in this fashion:
- 1 shard
- 2 shards
- 3 shards
- 4 shards + model creation
- 4 shards + 1 meta_shard guided by the metadynamcis of the model

FES
![Free Energy Surface animation](figs/fes.gif)

TM
![Transition Matrix animation](figs/transition.gif)

## Installation

```bash
# From PyPI (recommended)
pip install pmarlo

# From source (development)
git clone https://github.com/Komputerowe-Projektowanie-Lekow/pmarlo.git
cd pmarlo
pip install -e .
```

- Python: 3.10–3.13
- Optional: pip install pmarlo[fixer] to include code formatting tools (black, isort, ruff) and pdbfixer (pdbfixer only available on Python < 3.12)
- ML CVs (Deep-TICA): pip install pmarlo[mlcv] to enable training with mlcolvar + torch. For deployment in PLUMED, ensure PLUMED ≥ 2.9 is built with the pytorch module so PYTORCH_MODEL can load TorchScript models.


## Testing

Testing
The test layout mirrors `src/pmarlo`, so unit tests live under `tests/unit/<domain>` and integration flows under `tests/integration/**`. Pytest discovers unit tests by default and the `pytest-testmon` plugin keeps reruns focused on files touched in the current branch.

Default quick check: `poetry run pytest --testmon -n auto`.

Suggested commands:

- `poetry run pytest --testmon -n auto` - default fast loop (unit, change-aware)
- `poetry run pytest --testmon --focus data,io -n auto` - run only the selected domains
- `poetry run pytest -m "unit and data" -n auto` - use marker syntax when you prefer classic selection
- `poetry run pytest -m "integration" tests/integration` - integration-only sweep
- `poetry run pytest -m "unit or integration or perf" -n auto` - full suite on demand
- `poetry run pytest --lf -q` - rerun only the most recent failures during triage

Combine `--focus` with `--testmon` whenever you want to zero in on a subset of packages while letting pytest skip unrelated tests automatically.

### Performance Benchmarking

For performance testing and regression detection, see **[README_BENCHMARKS.md](README_BENCHMARKS.md)**. Quick start:

```bash
export PMARLO_RUN_PERF=1  # Enable performance tests
poetry run pytest -m benchmark --benchmark-save=baseline
# Make your changes...
poetry run pytest -m benchmark --benchmark-compare=baseline
```

## Dependency policy

PMARLO now enforces a single canonical implementation for every feature. All runtime fallbacks and legacy code paths have been removed, and missing dependencies raise clear ImportError exceptions during import or first use. Install the relevant extras (for example, `pip install 'pmarlo[analysis]'`) to enable advanced analyses.


## Quickstart

```python
from pmarlo.transform.pipeline import run_pmarlo

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

- numpy >= 1.24, < 2.4
- scipy >= 1.10, < 2.0
- pandas >= 1.5, < 3.0
- mdtraj >= 1.9, < 2.0
- openmm >= 8.1, < 9.0
- rdkit >= 2024.03.1, < 2025.0
- psutil >= 5.9, < 6.1
- pygount >= 2.6, < 3.2
- mlcolvar >= 1.2
- scikit-learn >= 1.2, < 2.0
- deeptime >= 0.4.5, < 0.5
- tomli >= 2.0, < 3.0
- typing-extensions >= 4.8
- pyyaml >= 6.0, < 7.0

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
- demux_begin: elapsed_s, segments
- demux_segment: elapsed_s, current, total, index; eta_s
- demux_end: elapsed_s, frames, file
- emit_begin: elapsed_s, n_inputs, out_dir
- emit_one_begin: elapsed_s, current, total, traj; eta_s
- emit_one_end: elapsed_s, current, total, traj, shard, frames; eta_s
- emit_end: elapsed_s, n_shards
- aggregate_begin: elapsed_s, total_steps, plan_text
- aggregate_step_start: elapsed_s, index, total_steps, step_name
- aggregate_step_end: elapsed_s, index, total_steps, step_name, duration_s
- aggregate_end: elapsed_s, status
- finished: elapsed_s, status

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
