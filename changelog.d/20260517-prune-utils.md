## added

- Added `find_conformations_from_msm` to the public API facade for MSM-driven TPT conformations discovery.
- Added `true_medoid` representative selection for conformations.

## fixed

- Fixed diverse representative selection to avoid duplicate frames for identical features and to validate non-finite or negative weights.

## removed

- `src/pmarlo/utils/integrator.py` — `create_langevin_integrator` had no importers after removal of `single_temp_md`, `replica_exchange`, and `workflow` modules.
- `src/pmarlo/utils/logging_utils.py` — `emit_banner`, `StageTimer`, and friends had no importers after removal of `experiments` and `workflow` modules; test `tests/unit/utils/test_logging_utils.py` removed alongside.
- `src/pmarlo/utils/temperature.py` — `collect_temperature_values` / `primary_temperature` extracted temperatures from shard metadata for demultiplexing; had no importers after removal of `demultiplexing` and `shards` modules; test `tests/unit/utils/test_temperature.py` removed alongside.
- `tests/unit/utils/test_cleanup.py` — tested `pmarlo.utils.cleanup.prune_workspace` which no longer exists (deleted in a prior cleanup pass).

## changed

- Conformations analysis now reuses the shared `kT_kJ_per_mol` thermodynamics helper instead of a local duplicate.
- Renamed representative selection from `medoid` to `closest_to_centroid` and updated conformations finder defaults.
