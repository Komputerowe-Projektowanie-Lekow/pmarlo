## removed

Removed 10 example programs that became non-functional after deletion of `replica_exchange`, `workflow`, `experiments`, `shards`, `transform`, `reporting`, `demultiplexing`, and related modules:

- `01_verify_pmarlo.py` — imported `ReplicaExchange`, `Simulation`, `Pipeline`, `RemdConfig`
- `02_pipeline_api.py` — imported `Pipeline`, `run_pmarlo`, `ReplicaExchange`
- `03_remd_sampling.py` — imported `api.replica_exchange.run_replica_exchange`
- `04_single_temperature_sampling.py` — imported `run_single_temperature_md`, `emit_shards_rg_rmsd_windowed`
- `05_shards_to_build.py` — imported `pmarlo.transform`, `aggregate_and_build`, `emit_shards_from_trajectories`
- `06_reproducible_build.py` — imported `pmarlo.transform.build`
- `07_free_energy_landscape.py` — imported `replica_exchange`, `power_of_two_temperature_ladder`
- `08_conformations_msm.py` — imported `pmarlo.reporting`, `pmarlo.transform`
- `09_conformations_tpt.py` — called `api.build_simple_msm` which is not exported from `pmarlo.api`
- `12_openmm_bias_benchmark.py` — imported `pmarlo.replica_exchange.system_builder`

Retained `10_tpt_basic.py`, `11_tpt_drunkards_walk.py`, and `_example_support.py`; these depend only on active `pmarlo.markov_state_model` code.

## added

Added `example_programs/13_adaptive_retraining_colab.ipynb`, a Google Colab notebook for the mdshare-based offline replay experiment comparing retraining triggers, training-data policies, and training-budget policies for adaptive CV workflows.

Added `example_programs/14_muller_brown_active_bias.py` and `example_programs/14_muller_brown_active_bias_colab.ipynb` for an active-feedback Muller-Brown retraining experiment where the bias force enters the Langevin integrator.

## fixed

Updated the adaptive retraining Colab setup to install only notebook runtime dependencies with `pip`, avoiding Poetry-driven downgrades of preinstalled Colab packages.
