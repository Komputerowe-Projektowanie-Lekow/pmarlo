# PMARLO: Protein Markov State Model Analysis with Replica Exchange

[![PyPI Version][pypi-image]][pypi-url]
[![Build Status][build-image]][build-url]
[![Python Versions][versions-image]][versions-url]
[![][stars-image]][stars-url]
[![License][license-image]][license-url]
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/Komputerowe-Projektowanie-Lekow/pmarlo)



A Python package for Conformation finding, protein simulation and Markov state model chain generation, providing an OpenMM-like interface for molecular dynamics simulations.

## Features

- **Protein Preparation**: Automated PDB cleanup and preparation
- **Replica Exchange**: Enhanced sampling with temperature replica exchange
- **Simulation**: Single-temperature MD simulations
- **Different shards**: Cosutomizable shard modes for specific purposes.
- **Markov State Models**: MSM analysis
- **DeepTICA model training**: Creates model to metabias the simulations with specific force on the CVs.
- **ITS analysis**: Helps to find the slow modes of the protein to get the most of the data from the smalles computation time.
- **Conformations finding**: Enables to get the conformations from currently discovered potential FES.
- **Pipeline Orchestration**: Complete workflow coordination


## Free Energy Surface/Transition Matrix

This is the animation of the FES/TM generated with specific amount of shards(dataset units that could be combined to make the models better or produce the analysis artifact)

Those were generated in this fashion:
- 1 shard
- 2 shards
- 3 shards
- 4 shards + model creation
- 4 shards + 1 meta_shard guided by the metadynamcis of the model

FES (PMARLO analysis workflow)
![Free Energy Surface animation](figs/fes.gif)

TM (PMARLO analysis workflow)
![Transition Matrix animation](figs/transition.gif)

Idea of sampling the protein (PMARLO analysis workflow)
![Idea of sampling the protein](https://i.imgur.com/4zQpIU6.png)

FES with different shard (PMARLO analysis workflow)
![FES with different shard](https://i.imgur.com/zpSrVgP.png)

Stationary distirbution of the protein (PMARLO analysis workflow)
![Stationary distirbution of the protein](https://i.imgur.com/qzAxfQY.png)

PCCA with conformations on top of FES (PMARLO analysis workflow)
![PCCA with conformations on top of FES](https://i.imgur.com/vkrPB9k.png)

Frames per shard (PMARLO analysis workflow)
![Frames per shard](https://i.imgur.com/iSGxPgy.png)

Free Energy Validation (PMARLO analysis workflow)
![Free Energy Validation](https://i.imgur.com/77H0jUu.png)

DeepTICA model preview (PMARLO analysis workflow)
![DeepTICA model preview](https://i.imgur.com/oFtHYn7.png)

OpenMM-Torch Bias Benchmark(Example programs)
![OpenMM-Torch Bias Benchmark](https://i.imgur.com/ATvwglC.png)

ITS (PMARLO analysis workflow)
![ITS](https://i.imgur.com/q9Q21wn.png)

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
The test layout mirrors `src/pmarlo`, so unit tests live under `tests/unit/<domain>` and integration flows under `tests/integration/**`. Pytest discovers unit tests by default and the `pytest-testmon` plugin keeps reruns focused on files touched in the current branch. There is a great experiment suite implemented in the package itself and the example programs to for example benchmark openmm-torch script simulation bias.

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

Quick start:

```bash
export PMARLO_RUN_PERF=1  # Enable performance tests
poetry run pytest -m benchmark --benchmark-save=baseline
# Make your changes...
poetry run pytest -m benchmark --benchmark-compare=baseline
```

## Dependency policy

PMARLO now enforces a single canonical implementation for every feature. All runtime fallbacks and legacy code paths have been removed, and missing dependencies raise clear ImportError exceptions during import or first use. Install the relevant extras (for example, `pip install 'pmarlo[analysis]'`) to enable advanced analyses.

The most suitable usage is to create the micromamba environment with openmm-torch script and the PDBFixer. Rest of the dependencies are suitable for the pip and could be downloaded there. At the moment the current usage is with micromamba environment with , poetry pip .e intalled

Create data through the Python APIs and example programs, then train and analyze
models from notebooks or scripts. For conformations, the workflow has been tested
with 35 shards and approximately 13K frames.

Example bias benchmark (the model and protein should be provided by the user):
```commandline
micromamba activate ommtorch
poetry run python example_programs\12_openmm_bias_benchmark.py --with-bias=yes --model=C:\path\to\deeptica_cv_model.pt --steps 5000 --platform CPU --pdb=C:\path\to\3gd8-fixed.pdb
```

## Quickstart

The repository ships with numbered runnable programs under `example_programs`.
They are intended to be read in order and run directly from the project root.
Each script writes to a matching directory under `example_programs/programs_outputs`.

| Script | Output directory | Purpose |
| --- | --- | --- |
| `01_verify_pmarlo.py` | `01_verify_pmarlo` | Instantiate the core objects and run a short pipeline. |
| `02_pipeline_api.py` | `02_pipeline_api` | Show the high-level `Pipeline` API and setup phases. |
| `03_remd_sampling.py` | `03_remd_sampling` | Run a short REMD sampling job. |
| `04_single_temperature_sampling.py` | `04_single_temperature_sampling` | Collect independent single-temperature trajectories and emit shards. |
| `05_shards_to_build.py` | `05_shards_to_build` | Emit shards, aggregate them, and build MSM/FES artifacts. |
| `06_reproducible_build.py` | `06_reproducible_build` | Build a reproducible provenance bundle. |
| `07_free_energy_landscape.py` | `07_free_energy_landscape` | Run sampling and generate FES/MSM outputs. |
| `08_conformations_msm.py` | `08_conformations_msm` | Find representative conformations from MSM macrostates. |
| `09_conformations_tpt.py` | `09_conformations_tpt` | Run protein conformation discovery with TPT. |
| `10_tpt_basic.py` | `10_tpt_basic` | Demonstrate TPT on a small explicit transition matrix. |
| `11_tpt_drunkards_walk.py` | `11_tpt_drunkards_walk` | Compare PMARLO TPT against deeptime's drunkard's walk example. |
| `12_openmm_bias_benchmark.py` | `12_openmm_bias_benchmark` | Benchmark OpenMM with an exported CV bias model. |

Start with:

```bash
poetry run python example_programs/01_verify_pmarlo.py
poetry run python example_programs/02_pipeline_api.py --mode demo
poetry run python example_programs/05_shards_to_build.py
```

For single-temperature runs or MSM-only workflows, point the examples at your
own PDB and adjust their parameters. The examples use production APIs directly;
there is no web application layer or hidden compatibility path.

## Verification and CLI

- `poetry run pmarlo --help` lists every CLI entry point.
- `poetry run pmarlo --mode simple` runs the minimal built-in demo.
- `poetry run python example_programs/05_shards_to_build.py` documents the shard
  orchestration workflow used in end-to-end examples.

Smoke test inside the virtual environment:

```bash
poetry run python - <<'PY'
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
