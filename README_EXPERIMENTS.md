## PMARLO Experiments (Algorithm Testing Framework)

Lightweight framework to iterate on algorithms without unit tests. Three stages:
- Simulation: protein prep, equilibration, and single-T production
- Replica exchange: multi-T REMD with exchange stats
- MSM: clustering + MSM/TRAM on trajectories

### Quick start (local)

Run with the bundled test data in `tests/data`:

```bash
python -m pmarlo.experiments.cli simulation --steps 500
python -m pmarlo.experiments.cli remd --steps 800 --equil 200
python -m pmarlo.experiments.cli msm --traj tests/data/traj.dcd --top tests/data/3gd8-fixed.pdb
```

Each command creates a timestamped directory under `experiments_output/<stage>/` with `config.json` and result JSON.

### What gets produced
- Simulation: `traj.dcd`, `final.xml`, `metrics.json` (frames, states), prepared PDB
- REMD: `replica_*.dcd`, `stats.json` (acceptance, visits), demux if available
- MSM: analysis artifacts in `msm/` (matrices, tables, plots) and `summary.json`

### Programmatic use

```python
from pmarlo.experiments import (
    run_simulation_experiment,
    run_replica_exchange_experiment,
    run_msm_experiment,
)
from pmarlo.experiments.simulation import SimulationConfig
from pmarlo.experiments.replica_exchange import ReplicaExchangeConfig
from pmarlo.experiments.msm import MSMConfig

# 1) Simulation
sim_res = run_simulation_experiment(SimulationConfig(pdb_file="tests/data/3gd8-fixed.pdb"))

# 2) REMD (use prepared PDB)
remd_res = run_replica_exchange_experiment(ReplicaExchangeConfig(pdb_file="tests/data/3gd8-fixed.pdb"))

# 3) MSM on trajectory
msm_res = run_msm_experiment(
    MSMConfig(
        trajectory_files=["tests/data/traj.dcd"],
        topology_file="tests/data/3gd8-fixed.pdb",
    )
)
```

### Docker (optional)

```bash
docker build -t pmarlo-exp .
docker run --rm -it -v "$PWD":/app pmarlo-exp python -m pmarlo.experiments.cli --help

# Run examples
docker run --rm -it -v "$PWD":/app pmarlo-exp \
  python -m pmarlo.experiments.cli simulation --steps 500 --pdb tests/data/3gd8-fixed.pdb
docker run --rm -it -v "$PWD":/app pmarlo-exp \
  python -m pmarlo.experiments.cli remd --steps 800 --equil 200 --pdb tests/data/3gd8-fixed.pdb
docker run --rm -it -v "$PWD":/app pmarlo-exp \
  python -m pmarlo.experiments.cli msm --traj tests/data/traj.dcd --top tests/data/3gd8-fixed.pdb
```

### Notes
- The experiment runners use existing package APIs (`Pipeline`, `ReplicaExchange`, `EnhancedMSM`)
- Outputs are small and tuned for quick iterations; adjust steps as needed.
