"""
Experiment framework for algorithm testing in PMARLO.

Provides lightweight runners to test:
- Simulation and equilibration
- Replica exchange and exchange statistics
- MSM construction and analysis

Usage (CLI):
  python -m pmarlo.experiments.cli --help
"""

from .msm import run_msm_experiment
from .replica_exchange import run_replica_exchange_experiment
from .simulation import run_simulation_experiment

__all__ = [
    "run_simulation_experiment",
    "run_replica_exchange_experiment",
    "run_msm_experiment",
]
