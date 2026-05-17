## removed

- `src/pmarlo/utils/integrator.py` — `create_langevin_integrator` had no importers after removal of `single_temp_md`, `replica_exchange`, and `workflow` modules.
- `src/pmarlo/utils/logging_utils.py` — `emit_banner`, `StageTimer`, and friends had no importers after removal of `experiments` and `workflow` modules; test `tests/unit/utils/test_logging_utils.py` removed alongside.
- `src/pmarlo/utils/temperature.py` — `collect_temperature_values` / `primary_temperature` extracted temperatures from shard metadata for demultiplexing; had no importers after removal of `demultiplexing` and `shards` modules; test `tests/unit/utils/test_temperature.py` removed alongside.
- `tests/unit/utils/test_cleanup.py` — tested `pmarlo.utils.cleanup.prune_workspace` which no longer exists (deleted in a prior cleanup pass).
