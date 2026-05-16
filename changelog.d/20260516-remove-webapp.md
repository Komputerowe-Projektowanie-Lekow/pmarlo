### added

- Added API feature profiles and configurable molecular-feature shard extraction for notebook and script workflows.

### changed

- Updated documentation, CI filters, and CV-bias guidance to point at package APIs instead of the removed web application.
- Reorganized `example_programs` into numbered API examples with matching numbered output directories under `example_programs/programs_outputs`.
- Replaced obsolete DeepTICA trainer wrapper imports in benchmarks with direct `pmarlo.ml.deeptica.trainer` imports.
- Tightened analysis pair counting, FES component selection, and discretization schema handling to fail on invalid inputs instead of silently repairing them.

### changed

- `generate_2d_fes`, `generate_free_energy_surface`, and `generate_fes_and_pick_minima` no longer accept deprecated `smooth` and `inpaint` boolean arguments; use `fes_smoothing_mode='always'` or `fes_smoothing_mode='auto'` instead. `generate_2d_fes` now accepts `fes_smoothing_mode` as a direct keyword argument.
- `_load_or_train_model` in `transform.build` no longer accepts unused `model_dir` and `model_prefix` compatibility shim parameters.

### removed

- Removed the `pmarlo_webapp` Streamlit application and its app-specific tests.
- Removed stale debug and developer utility scripts from `example_programs`.
- Removed tracked build, distribution, temporary shard, and benchmark output artifacts from the repository.
- Removed deprecated demultiplexing compatibility facades and their wrapper-only tests.
- Removed unused shard metadata fallback helpers and the private `_trig_expand_periodic` API alias.
- Removed analysis dead code including the local Tarjan SCC implementation, stale tau constants, stale diagnostics tests, and the FES private variance-selection alias.
- Removed `find_conformations_with_msm` public API alias (was a thin wrapper around `find_conformations`).
- Removed `PairBuilder.update_tau` backward-compatibility alias (use `set_tau` directly).
- Removed `ShardId.segment_id` backward-compatibility property (use `local_index` directly).
- Removed `_collect_demux_temperatures` from `transform.build` (was an unused delegation stub).
