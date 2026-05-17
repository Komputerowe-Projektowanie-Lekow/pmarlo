### added

- Added API feature profiles and configurable molecular-feature shard extraction for notebook and script workflows.

### fixed

- Fixed joint workflow CK configuration typing, import-time dependency loading, CV bootstrap state handling, weighted transition counting, and ITS eigenvalue calculation.
- Fixed conformation state-flux scoring to use the same incoming/outgoing average as bottleneck ranking.
- Fixed CK guardrail configuration validation, missing-lag handling, per-lag pass typing, absolute-mode reporting, and vectorized multinomial noise estimation.
- Fixed `TemperatureConsistencyError` to participate in the demux exception hierarchy.

### changed

- Updated documentation, CI filters, and CV-bias guidance to point at package APIs instead of the removed web application.
- Reorganized `example_programs` into numbered API examples with matching numbered output directories under `example_programs/programs_outputs`.
- Replaced obsolete DeepTICA trainer wrapper imports in benchmarks with direct `pmarlo.ml.deeptica.trainer` imports.
- Tightened analysis pair counting, FES component selection, and discretization schema handling to fail on invalid inputs instead of silently repairing them.
- Joint workflow frame weights now preserve shard-level totals until the global MSM normalization step, and placeholder iteration metrics now explicitly report missing CV trainer integration.
- Corrected workflow finalization to derive MSM stationary distributions from transition matrices, pass reweighting weights into FES estimation, and validate debug/config inputs before analysis work starts.
- Updated workflow validation to report non-canonical shard identifiers as validation errors, include embedded FES quality diagnostics, and keep shard report columns aligned.
- Moved analysis-specific FES and REMD validation thresholds into `pmarlo.analysis.constants`.
- Simplified visualization diagnostics wrappers to rely on reporting plot validation and avoid redundant figure handling.
- Moved DeepTICA configuration parsing into the DeepTICA feature package and made cleanup report malformed runs instead of aborting the whole workspace prune.
- Consolidated input parsing strictness behind explicit parser options and removed domain-specific default bins from the parsing helper.
- Moved DeepTICA payload sanitization out of generic JSON utilities and made row normalization reuse recursive JSON sanitization.

### changed

- `generate_2d_fes`, `generate_free_energy_surface`, and `generate_fes_and_pick_minima` no longer accept deprecated `smooth` and `inpaint` boolean arguments; use `fes_smoothing_mode='always'` or `fes_smoothing_mode='auto'` instead. `generate_2d_fes` now accepts `fes_smoothing_mode` as a direct keyword argument.
- `_load_or_train_model` in `transform.build` no longer accepts unused `model_dir` and `model_prefix` compatibility shim parameters.
- `data.aggregate.aggregate_and_build` now fails loudly on duplicate shard IDs or missing build metadata instead of silently continuing with incomplete artifacts.
- `pmarlo.conformations` now exposes its public API through direct imports instead of a lazy compatibility export table.
- Conformation state detection now raises clear errors for invalid explicit FES/timescale/population inputs instead of returning placeholder source and sink states.

### removed

- Removed the `pmarlo_webapp` Streamlit application and its app-specific tests.
- Removed stale debug and developer utility scripts from `example_programs`.
- Removed tracked build, distribution, temporary shard, and benchmark output artifacts from the repository.
- Removed deprecated demultiplexing compatibility facades and their wrapper-only tests.
- Removed unused shard metadata fallback helpers and the private `_trig_expand_periodic` API alias.
- Removed analysis dead code including the local Tarjan SCC implementation, stale tau constants, stale diagnostics tests, and the FES private variance-selection alias.
- Removed the `analysis.debug_export.total_pairs_from_shards` wrapper and empty annotated-plot export hook.
- Removed `find_conformations_with_msm` public API alias (was a thin wrapper around `find_conformations`).
- Removed the `find_pathway_intermediates` argument and `get_pathway_intermediates` result alias from the conformations API; use transition and transition-state-ensemble accessors directly.
- Removed `PairBuilder.update_tau` backward-compatibility alias (use `set_tau` directly).
- Removed `ShardId.segment_id` backward-compatibility property (use `local_index` directly).
- Removed `_collect_demux_temperatures` from `transform.build` (was an unused delegation stub).
- Removed private helper exports and compatibility aliases from the `pmarlo.api` facade.
- Removed the duplicate progress-callback alias helper and lazy transform-build import wrapper from `data.aggregate`.
- Removed unused conformation frame-assignment code and a trivial shard ID wrapper from aggregation.
- Removed the lenient `coerce_tau_schedule` API; use `parse_tau_schedule(..., strict=False)` for explicit invalid-token skipping.
- Removed the `demultiplexing`, `replica_exchange`, `reweight`, `samplers`, `experiments`, `workflow`, `shards`, `reporting`, `transform`, and top-level `pairs` packages because they serve multi-simulation, multi-temperature, progress-reporting, or orchestration workflows outside the single-trajectory core.
- Removed shard/demux/transform facade modules from `api`, `data`, `io`, `utils`, and `markov_state_model` along with tests that imported those removed surfaces.
- Removed remaining scaled-time reweight pair construction, `__shards__` analysis compatibility, shard schema constants, and shard frame histogram APIs from the retained single-trajectory modules.

### changed

- Moved DeepTICA lagged pair construction into `pmarlo.features.deeptica.core.pairs` so the single-trajectory ML path no longer depends on the removed top-level `pairs` package.
- Replaced reporting-package plot/export imports in the retained API and visualization modules with local single-trajectory plotting/export helpers.
- Narrowed public package exports to protein, API, visualization, MSM, FES/PMF, and single-trajectory analysis utilities.
- Renamed remaining DeepTICA/MSM diagnostics from shard-oriented wording to trajectory or segment wording, and kept lagged-pair diagnostics limited to uniform single-trajectory timing.
- RDKit protein descriptor calculation now uses in-memory PDB blocks instead of temporary PDB files.
- Protein metric residue sets and pKa values now load from the runtime configuration file.
