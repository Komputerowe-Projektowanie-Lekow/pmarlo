<a id='changelog-0.113.0'></a>
# 0.113.0 — 2025-10-16


### Changed
- **OpenMM System Creation**: `create_system()` in `src/pmarlo/replica_exchange/system_builder.py` now accepts optional CV model parameters and integrates `TorchForce` from openmm-torch when provided
- **API Expansion**: `run_replica_exchange()` in `src/pmarlo/api.py` now accepts `cv_model_path`, `cv_scaler_mean`, and `cv_scaler_scale` parameters for CV-biased sampling
- **Enhanced Training Diagnostics**: Logs now show both pre-clip and post-clip gradient norms (`grad_preclip` and `grad_postclip`), clipping threshold, expected pairs, and tau schedule for curriculum training
- Added uniform stage banners and per-step console echoes across the PMARLO pipeline, REMD sub-phases, and demultiplexing, plus milestone progress logging and end-of-phase summaries so logs clearly track execution (`src/pmarlo/transform/pipeline.py`, `src/pmarlo/replica_exchange/replica_exchange.py`, `src/pmarlo/demultiplexing/demux.py`, `src/pmarlo/utils/logging_utils.py`).
- Added perf-counter driven timing with throughput and peak-memory summaries: pipeline stages now record wall-clock durations (with aggregate timing table and optional tracemalloc peak), REMD equilibration/production phases announce elapsed time, and demultiplexing reports streaming duration alongside frame counts (`src/pmarlo/transform/pipeline.py`, `src/pmarlo/utils/logging_utils.py`, `src/pmarlo/replica_exchange/replica_exchange.py`, `src/pmarlo/demultiplexing/demux.py`).
- **Implemented CV-to-force transformation via CVBiasPotential wrapper**: Created `src/pmarlo/features/deeptica/cv_bias_potential.py` that wraps Deep-TICA models with harmonic expansion bias (E = k * Σ(cv²)). The wrapper transforms collective variable outputs into energy values, allowing OpenMM to compute biasing forces via automatic differentiation (F = -∇E). This enables proper physics-based conformational exploration
- **Re-enabled CV-informed sampling in app**: Updated `example_programs/app_usecase/app/backend.py` and `app.py` to use the new CVBiasPotential. Models are now exported with bias potential included, and simulations can run with CV-based biasing forces (when openmm-torch is installed)
- **Simplified OpenMM integration**: Updated `src/pmarlo/replica_exchange/system_builder.py` to directly add TorchForce without manual scaler handling, since scaling is now embedded in the CVBiasPotential model
- Export pipeline now composes the TorchScript feature extractor, embeds a feature-spec SHA-256, runs `torch.jit.optimize_for_inference`, and persists metadata for runtime validation (`src/pmarlo/features/deeptica/export.py` with new unit tests under `tests/features/deeptica/`).
- `create_system()` enforces `PMARLO_TORCH_THREADS`, pins TorchForce precision to single, and validates the scripted model's metadata (feature hash, PBC flag) before attaching the bias (`src/pmarlo/replica_exchange/system_builder.py`).
- Standardized the example app shard metadata to match the canonical PMARLO shard schema (explicit periodic flags, canonical provenance keys, and float32-aligned `dt_ps`).
- Added `lightning` (pytorch-lightning) to `mlcv` optional dependencies in `pyproject.toml` as required by `mlcolvar`.
- DeepTICACurriculumTrainer now writes real-time progress updates after each epoch and tracks pre-clip gradient norms
- TrainingResult dataclass now includes `checkpoint_dir` and `cv_model_bundle` fields
- SimulationConfig dataclass now includes `cv_model_bundle` field for CV-informed sampling
- `AppBackend.train_model()` now automatically exports models to TorchScript with scaler and metadata
- `AppBackend.run_sampling()` loads CV model info and propagates to replica exchange when selected
- Training completion is now marked in progress file with final summary statistics
- Simulation metadata now includes CV model reference when CV-informed sampling is used

### Fixed
- Ensured DeepTICA models that rely on a backing `.nn`/`.net` module expose a callable `forward()` so the full training workflow and FES shape checks run without raising `NotImplementedError`.
- **Fixed coverage >100% bug in `src/pmarlo/pairs/core.py`**: When using curriculum training with multiple taus, coverage was calculated incorrectly by counting usable pairs for all taus but expected pairs for only the maximum tau, resulting in coverage values like 101.5%. Now correctly counts expected pairs for all taus in the schedule.
- **Fixed gradient norm logging in `src/pmarlo/ml/deeptica/trainer.py`**: Gradient norms were logged AFTER clipping, showing constant values (typically 1.0). Now logs both pre-clip and post-clip gradient norms to reveal true optimization dynamics.
- **Enhanced training diagnostics**: Logs now show `grad_preclip=(mean/max)`, `grad_postclip=(mean/max)`, `clip_at=threshold`, `expected_pairs`, and `tau_schedule` for better debugging.
- Fixed critical bug in `example_programs/app_usecase/app/backend.py` where analysis guardrails were checked against pre-clustering debug data, causing all analyses to fail with "total_pairs_lt_5000" error before the MSM build could complete.
- Modified guardrail validation to check post-clustering MSM build results instead of pre-clustering debug data, allowing proper MSM construction with transition matrix and free energy surface generation.
- Resolved analysis guardrail failures in the example app by ensuring shard manifests declare full provenance required by `load_shard_meta`.
- Fixed dependency management issue where `mlcolvar` required `lightning` (pytorch-lightning) but it was not specified in optional dependencies, causing "No module named 'mlcolvar'" errors.
- Training progress is now visible during and after model training instead of silent execution
- Prevented DeepTICA TorchScript export from recursing indefinitely by tracking visited modules and re-exposing the `plumed_snippet` helper on `DeepTICAModel`.
- Normalized DeepTICA pair diagnostics to cap usable pair counts by actual frame counts, keeping coverage ≤100% in trainer telemetry.
- Added explicit `tomli` dependency and switched tooling imports away from `tomllib` so Python 3.10 environments can execute parity checks.
- Hardened LEARN_CV failure handling to detect missing `lightning`/`pytorch_lightning` installations and emit structured `missing_dependency:*` skip records instead of aborting the build.
- **CRITICAL: Fixed multiple simultaneous simulation bug in Streamlit app**: The "Run replica exchange" button was triggering multiple parallel simulations due to Streamlit rerun behavior. Moved simulation execution inside the button click handler to ensure only ONE simulation runs per click, not one per rerun. This was causing 5+ hour runtimes as multiple simulations competed for resources.
- **Fixed lazy loading import error for CV functions**: Updated `__getattr__` in `src/pmarlo/features/deeptica/__init__.py` to check standalone exports (like `load_cv_model_info`) before trying to load from `_full` module, preventing "cannot import name" errors
- **Enhanced CV integration error handling**: Added comprehensive logging and warnings in `backend.py` for missing openmm-torch, CPU-only PyTorch performance warnings, and graceful degradation when CV models can't be loaded
- **Fixed GitHub Actions test collection failure**: Added `tests/integration` to `testpaths` in `pyproject.toml` and added `@pytest.mark.integration` markers to all integration test files to enable proper test discovery.
- **Fixed GitHub Actions pytest exit code 5 issue**: Updated all GitHub Actions workflow test commands to explicitly specify test directories (`tests/unit tests/devtools tests/integration`) instead of relying on marker filters alone, preventing pytest from collecting unmarked tests incorrectly.
- **Fixed tomli import error in Python 3.11+**: Updated `scripts/check_extras_parity.py` to use `tomllib` from the standard library for Python 3.11+ instead of the external `tomli` package, fixing test collection errors in devtools tests.
- Restored Poetry package metadata (`name`, `description`, `authors`) in `[tool.poetry]` to satisfy package-mode CI checks while keeping it synchronized with the canonical `[project]` metadata.
- **CRITICAL PERFORMANCE FIX: Disabled CV-biased sampling** due to catastrophic 10-100x slowdown caused by incomplete implementation. Root causes: (1) TorchForce passes raw atomic positions but CV model expects molecular features (distances/angles/dihedrals) - feature extraction not implemented, (2) PyTorch runs on CPU when random_seed is set (Reference platform for determinism), (3) PyTorch model called at every MD integration step. With 50k steps × 3 replicas = 150k PyTorch CPU calls → hours instead of minutes. Temporarily disabled CV biasing in `app.py` until proper OpenMM feature extraction is implemented. See `mdfiles/cv_biasing_performance_issue.md` for details.
- Updated integration and analysis test fixtures to emit canonical shard metadata so new schema validations pass and cache invalidation remains covered without depending on legacy shard IDs.


### Improved
- **Comprehensive console output for Streamlit debugging**: Added detailed console output using `print(..., flush=True)` statements throughout the replica exchange simulation pipeline to ensure visibility in Streamlit console
  - **Why print() instead of logger?** Python's `logger` output may not appear in Streamlit console, so we use direct print statements with flush=True for guaranteed visibility
  - Added startup banner in `src/pmarlo/api.py` showing simulation configuration (replicas, temperatures, steps, output directory, seed)
  - Added phase banners showing PHASE 1/2 (MD simulation) and PHASE 2/2 (demultiplexing) with clear stage boundaries
  - Added cancellation awareness messaging warning users that demultiplexing cannot be cancelled with Ctrl+C
  - Added detailed stage output in `src/pmarlo/replica_exchange/replica_exchange.py` showing equilibration and production phases with replica counts and exchange frequencies
  - Added demux stage output in `src/pmarlo/demultiplexing/demux.py` showing exchange history analysis, frame extraction progress, and completion statistics
  - All console output uses consistent 80-character separator lines for visual clarity
  - Completion messages show trajectory counts, frame counts, and output file paths for verification
  - Both `print()` (for console) and `logger` (for file-based logging) are used to ensure traceability
- **Explained demultiplexing lag after Ctrl+C**: The two progress bars users see are (1) the MD simulation which CAN be cancelled, and (2) the demultiplexing phase which CANNOT be cancelled and runs to completion. This is now clearly documented in the console output with warning symbols (⚠️).

### Testing
- Loaded every example shard with `pmarlo.data.shard_io.load_shard_meta`.
- Verified MSM builds successfully with valid transition matrices (50x50, row-stochastic) and stationary distributions from 2000-frame shard datasets.
- Tested complete workflow including normal analysis, DeepTICA integration, and all additional features.
- Added TorchScript parity and periodic-invariance regression tests plus TorchForce finite-difference validation under `tests/force/`.


### Added
- **CV-Informed OpenMM Sampling**: Full integration of trained Deep-TICA models into OpenMM simulations for enhanced sampling
  - `export_cv_model()` in `src/pmarlo/features/deeptica/export.py` exports trained models as TorchScript compatible with openmm-torch
  - `CVBiasForce` and `add_cv_bias_to_system()` in `src/pmarlo/features/deeptica/openmm_integration.py` integrate CV models as biasing forces
  - `check_openmm_torch_available()` validates openmm-torch installation for safe integration
  - Automatic model export after training with scaler parameters, metadata, and usage instructions
  - CV model selection in Sampling tab with automatic propagation to replica exchange simulations
  - Iterative training workflow support: unbiased sampling → CV learning → biased sampling → repeat to map FES
  - Comprehensive documentation in `CV_INTEGRATION_README.md` with workflow examples and troubleshooting
- Real-time training metrics logging to JSON file during DeepTICA model training
- Training progress visualization in Model Training tab with live epoch updates and metrics curves
- Model Preview tab in the app for inspecting trained model architecture, parameters, and training history
- CV-informed sampling option to associate trained CV models with simulation runs
- Checkpoint directory tracking for trained models to access training progress and checkpoints
- `training_progress.json` file written during training with epoch-by-epoch metrics
- `get_training_progress()` method in backend to read real-time training status
- `cv_model_bundle` field in `TrainingResult` and `SimulationConfig` for tracking exported models
- TorchScript feature extractor (`src/pmarlo/features/deeptica/ts_feature_extractor.py`) computes distance, angle, and dihedral features with minimum-image PBC entirely inside TorchScript so OpenMM can evaluate CV inputs on CPU every step.

<a id='changelog-0.107.0'></a>
# 0.107.0 — 2025-10-16


### Removed
- Remove fallbacks; fail-fast policy.
- Removed fallback implementations in the Markov state model toolkit and enforce fail-fast behavior when dependencies or data are missing.
- Remove fallbacks; fail-fast policy for DeepTICA whitening and device selection, enforcing explicit metadata and dependencies.


## Added
- Documented the new fail-fast policy for MSM utilities.

## Testing
- Updated analysis and DeepTICA tests to rely on real modules and strict metadata validation.

<a id='changelog-0.106.0'></a>
# 0.106.0 — 2025-10-16

### Removed

- Dropped the `statsmodels` dependency by replacing toroidal KDE and autocorrelation
  routines with internal NumPy/SciPy implementations so the example Streamlit app
  runs without extra site-packages. Removed `statsmodels` from requirements.txt and
  updated poetry.lock to reflect the changes.
- Remove fallbacks across utility modules and enforce fail-fast handling for missing dependencies and workspace layouts.
- Remove fallbacks; fail-fast policy for analysis metadata and whitening handling.
- Remove fallbacks; fail-fast policy.
- Removed alternative shard code paths in the data pipeline and enforced fail-fast dependency handling.
- Removed experiment fallbacks, enforcing canonical dependencies and raising immediately when requirements are missing.
- Remove fallbacks; fail-fast policy.
- Remove fallbacks; fail-fast policy across replica-exchange utilities.
- Removed reporting module fallbacks and soft-degradation paths; plotting now fails fast when dependencies or data are missing.


### Added
- Document w_frame as the sole supported weight key during reweighting.
- Document demultiplexing as a single streaming implementation with fail-fast behaviour.

### Changed
- Enforced hard dependency on matplotlib and numpy within reporting exports and plots to follow the fail-fast policy.
- Remove fallbacks; fail-fast policy for demultiplexing subsystems.
- Require explicit dependency availability in seeding, MDTraj, and path utilities.
- Remove fallbacks; fail-fast policy for analysis reweighting outputs.


### Technical Details

- **No-redef fixes**: Renamed stub classes (e.g., `_EnhancedMSMStubClass`) and removed duplicate function definitions to avoid mypy redefinition errors
- **Valid-type fixes**: Used `TYPE_CHECKING` blocks to create type-only aliases (e.g., `EnhancedMSMType`) for conditional class definitions
- **Protocol instantiation fixes**: Applied constructor casting with `cast(Any, Constructor)` pattern when protocols masked real `__init__` signatures
- **Union-attr fixes**: Added explicit None checks and type guards before calling methods on potentially None values
- **Arg-type fixes**: Normalized `str | Path | None` to `str` using assertions and `str()` conversion in ReplicaExchange.from_config
- **Return-type fixes**: Fixed missing return values in run_simulation method by returning appropriate `List[str]` values
- **Transform/apply fixes**: Added missing imports, fixed function signatures, and corrected argument passing for MSM analysis functions
- **Type annotation improvements**: Added proper type annotations to dictionaries and variables to resolve object type inference issues
- **Deeptime dependency enforcement**:
  - Removed `_manual_tica()` and `_manual_vamp()` fallback implementations from reduction.py
  - Removed `_fit_msm_fallback()` custom MSM fitting from bridge.py and _msm_utils.py
  - All TICA, VAMP, and MSM estimation now exclusively use deeptime library implementations
  - Updated `build_simple_msm()` to directly use deeptime without fallback try/except logic
  - Updated tests to remove monkeypatching and fallback-related test cases
- **Monkeypatching elimination**: Removed all monkeypatching from tests throughout the project:
  - Removed sys.modules mocking for openmm, statsmodels, mdtraj, deeptime in test_experiments_kpi.py
  - Removed scipy.constants fallback mocking in test_thermodynamics.py and enforced scipy as required dependency
  - Replaced implementation detail tests (numpy.cov, sklearn.CCA spying) with behavioral tests in test_diagnostics.py
  - Replaced monkeypatch.chdir with explicit os.chdir/cleanup pattern in test_paths.py
  - Removed deeptime module mocking in test_ck.py
  - Tests now use pytest.importorskip for truly optional dependencies and real implementations for required ones
- Fixed comprehensive mypy type errors across multiple modules to achieve zero mypy errors for the type tox environment
- Resolved no-redef issues by using distinct names for fallback implementations (EnhancedMSM, prepare_structure)
- Fixed valid-type issues by implementing proper TYPE_CHECKING patterns for conditional type definitions
- Eliminated protocol instantiation errors by using proper type casting and constructor patterns
- Fixed union-attr issues by adding proper None checks before calling methods like `.lower()`
- Resolved arg-type and return-type mismatches in replica_exchange.py by normalizing Path/str types and fixing return statements
- Fixed mixed-type issues in transform/apply.py by adding proper imports, type annotations, and argument handling
- **Removed fallback implementations for TICA and VAMP** to enforce use of standard `deeptime` library implementations only
- **Removed custom MSM fitting fallback (_fit_msm_fallback)** from bridge.py and _msm_utils.py to enforce deeptime-only MSM estimation
- **Eliminated monkeypatching across the entire test suite** to ensure tests use real dependencies and standard library implementations


<a id='changelog-0.102.0'></a>
# 0.102.0 — 2025-10-16

### Fixed
- Switched implied timescale convergence regression to use `scipy.stats.linregress` for improved numerical stability and richer diagnostics.
- Replaced the bespoke temperature ladder retuning heuristic with a SciPy-based optimisation that fits pairwise acceptance targets while respecting the overall temperature span.
- Improve spectral gap calculation by relying on ``numpy.partition`` to avoid full sorting when identifying the dominant eigenvalues.
- Use `scipy.linalg.eigh` to solve the manual TICA generalized eigenproblem, improving numerical stability when deeptime and PyEMMA are unavailable.
- Replaced the handcrafted autocorrelation routine with `statsmodels`' robust
  `acf` implementation and documented dependency updates to keep diagnostics aligned with standard statistical tooling.
- Replaced the custom eigenvector solver in `markov_state_model._msm_utils._stationary_from_T` with SciPy's dense and sparse eigensolvers for a more robust stationary distribution calculation.


<a id='changelog-0.101.0'></a>
# 0.101.0 — 2025-10-16

## Fixed
- Replaced custom weighted mean and variance calculations in `analysis.fes` with NumPy's `average` implementation to leverage built-in numerical stability.
- Reused `numpy.cov` in canonical correlation diagnostics and MSM reduction fallbacks to avoid maintaining manual covariance code.
- Replaced the custom NaN-safe centering in `markov_state_model.reduction._preprocess` with a scikit-learn pipeline for consistent feature scaling.
- Replaced the bespoke DeeptiCA whitening covariance logic with scikit-learn's `ShrunkCovariance` and SciPy-based transforms to rely on well-tested numerical routines.
- Replaced the DeepTICA StandardScaler fallback with scikit-learn's implementation to leverage the maintained feature set.


## Testing
- Added unit tests that confirm the TICA, VAMP, and diagnostics fallbacks call into `numpy.cov`.

<a id='changelog-0.100.0'></a>
# 0.100.0 — 2025-10-16


## Fixed
- Deduplicated repeated array concatenation fallback logic by introducing the shared ``concatenate_or_empty`` utility and reusing it within the API and pair-building helpers.
- Centralized exchange history validation into ``normalize_exchange_mapping`` and reused it across demux, compatibility helpers, and the legacy API to ensure consistent error handling.
- Reused the shared `_row_normalize` helper throughout MSM estimators to eliminate divergent implementations across `bridge.py` and `ck_runner.py`.
- Centralized directory creation logic via the shared `ensure_directory` helper and refactored call sites to rely on it, making future permission or error-handling adjustments consistent.
- Replaced scattered ``np.zeros((0, ...))`` fallbacks with ``np.empty`` helpers to ensure consistent zero-length array shapes and dtypes across workflow, MSM, feature, and utility modules.
- Consolidated repeated topology loading and atom selection logic by introducing shared mdtraj utilities and reusing them across trajectory streaming and writing helpers.
- Removed duplicate MSM helper implementations by delegating candidate lag generation, count regularisation, and transition validation in the markov_state_model package to the shared ``pmarlo.utils.msm_utils`` module.
- Reused the shared residue-label helper for chi1 feature labels to keep error handling and label formatting consistent with other torsion features.
- Reused the shared ``_stationary_from_T`` helper from ``_msm_utils`` inside the MSM bridge to eliminate duplicate stationary-distribution logic.




<a id='changelog-0.99.0'></a>
# 0.99.0 — 2025-10-16

## Added
- Introduced unit coverage for the shared `kT` helper, including SciPy and fallback execution paths.
- Shared `pmarlo.utils.temperature` helpers with focused tests to extract shard temperatures from nested metadata consistently.
- Verification tests for the shared path resolution helper covering current working directory, repository root, and custom search roots.
- Regression tests covering mask/inpaint semantics of the shared free-energy converter.

## Changed
- Consolidated trajectory and MSM loading path resolution through `pmarlo.utils.path_utils.resolve_project_path`, eliminating divergent fallback behaviour.
- Demux shard filtering, shard JSON parsing, and shard readers now reuse the shared helper to normalise temperature provenance across subsystems.
- Centralised shard index discovery in `pmarlo.shards.indexing.initialise_shard_indices` and updated the API/data emitters to reuse the helper so shard numbering and seed allocation stay consistent without duplicating glob logic.
- DeepTICA curriculum config now leaves checkpoint_dir unset unless supplied, which prevents stray tmp_models/ and checkpoint folders; the trainer module also exposes the legacy batching helpers directly so downstream imports stay stable without touching TensorBoard extras.
- cluster_microstates now warns when scikit-learn collapses multiple clusters into one and trims any excess centers instead of raising, keeping MSM builds resilient to degenerate inputs.
- Developer utilities live under the scripts package so CLI helpers share a single home across the repository.
- Consolidated shard validation helper into pmarlo.utils.validation so schema modules share one implementation.
- Centralized shard JSON loading through `pmarlo.utils.json_io.load_json_file`, consolidating metadata parsing with clearer decode errors across shard readers.
- Centralised the Boltzmann density→free-energy conversion into `free_energy_from_density` so Ramachandran and MSM workflows share identical numerics.
- Unified shard read/write logic onto the canonical schema via pmarlo.data.shard wrappers so example and core code consume a single format.
- Experiment inputs regenerated in-place to include canonical metadata (feature_spec, dt_ps, temperature) with filenames aligned to canonical shard IDs.

## Removed
- Cleared legacy DeepTICA artefacts (checkpoints/, tmp_models/, runs/, and bias/) from the workspace so
new runs start clean.

## Fixed
- Consolidated thermal energy calculation into a shared helper to eliminate drift between free-energy and plotting code paths.
- Runner no longer trips legacy shard validation errors when loading example datasets.

<a id='changelog-0.98.0'></a>
# 0.98.0 — 2025-10-16

## Added
- STEPS.MD distills the experiment plan into phased implementation steps for the app_usecase workflow.
- Experiment input/output scaffolding under `app_intputs/experiments/` and `app/experiment_outputs/` with guidance files describing required shards, configs, manifests, and artefacts (including NPZ/JSON templates).
- Deterministic config templates (`transform_plan.yaml`, `discretize.yaml`, `reweighter.yaml`, `msm.yaml`) encoding DeepTICA, TRAM, and MSM guardrails for the E0/E1/E2 workflows.
- Experiment helper module (`example_programs/app_usecase/app/experiment/common.py`) resolving manifests, shard metadata, and config bundles for scripted runs.
- CLI runner framework (`example_programs/app_usecase/app/experiment/runner.py`) plus dedicated scripts for E0/E1/E2 that execute analysis, compute weight diagnostics, and emit acceptance reports under `experiment_outputs/`.
- Guardrail evaluation hooks that calculate ESS, SCC coverage, diagonal mass, and empty-state fractions against experiment-specific thresholds, reporting violations directly in `weights_summary.json`, `msm_summary.json`, and acceptance reports.
- DeepTICA orchestration now propagates lag fallbacks, records training diagnostics, enforces pair-count thresholds, and surfaces skip/failure reasons in analysis summaries and acceptance reports.
- Acceptance report generator consolidates weight/MSM/DeepTICA status and lists emitted artefacts (bundle, debug summaries, optional FES plots) to speed manual review.
- Unit tests added for experiment runner helpers covering weight guardrails, DeepTICA summarisation, and acceptance reporting without requiring full datasets.
- Headless analysis runner (`example_programs/app_usecase/app/headless.py`) to mirror Streamlit workflows from the terminal and print/save debug summaries.
- Dataset debug exporter (`pmarlo.analysis.debug_export`) that writes transition counts, MSM arrays, diagnostics, and configuration summaries for each build.
- Feature validation stage (`pmarlo.analysis.validate_features`) capturing column statistics and emitting `feature_stats.json` artefacts for debug bundles.
- Discretizer fingerprint now records the fitted feature schema so downstream bundles surface exact CV column metadata.
- State assignment validation raises `NoAssignmentsError` when clustering yields no labels and persists per-split `state_ids.npy` / `valid_mask.npy` in debug bundles.
- Expected MSM transition pairs now recorded via `expected_pairs(...)` with per-shard stride metadata in debug summaries.
- Introduced `pmarlo.constants` as the single source of project-wide physical, numeric, and domain defaults for reuse across subsystems.
- Pre-clustering diagnostics that log CV statistics, flag non-finite frames, and emit a `cv_distribution.png` histogram for quick visual inspection.
- Guard in the MSM clustering pipeline that verifies the requested number of microstates matches the unique labels returned by K-means/MiniBatchKMeans.
- Default analysis configuration now requests 20 microstates (down from 150) across headless/UI flows to better match modest datasets.
- Analyses now propagate `kmeans_kwargs={'n_init': 50}` so every K-means invocation retries with 50 restarts for more stable clustering.
- Headless analysis runner now aborts with a clear error message when `summary.json` reports `analysis_healthy=false`, echoing the recorded guardrail violations.
- Standalone `scripts/diagnose_cvs.py` utility to load trajectories, compute Rg/RMSD_ref, and dump detailed pandas/matplotlib diagnostics without triggering MSM/FES builds.
- Core DeepTICA trainer package modules (`config`, `loops`, `sampler`, `schedulers`, `trainer`) and focused unit tests under `tests/unit/features/deeptica/core/` covering each helper.
- Lightweight README for `src/pmarlo/features/deeptica/core/` documenting module responsibilities.
- Expanded reweighter unit coverage for TRAM aliasing, input immutability, and bias validation regressions.


## Changed
- Streamlit backend now captures per-build analysis summaries, persists them under `analysis_debug/`, and surfaces warning counts in stored artifacts.
- Workspace layout prepares a dedicated `analysis_debug` directory so raw analysis data lands in a predictable location alongside bundles.
- Captured discretizer fingerprints plus tau metadata per analysis build; UI and CLI now surface stride-derived effective tau and alert when overrides drift.
- Added synthetic MSM/CLI/Hypothesis tests plus compare_debug_bundles utility to guard against regression in fingerprint and transition statistics.
- MSM discretisation now aborts when post-whitening CVs contain non-finite values or zero variance columns, logging column statistics for every split.
- KMeans/Grid discretizers persist the training feature schema and refuse to transform splits whose names or order diverge, raising `FeatureMismatchError` with detailed differences.
- GitHub Actions test workflow now runs on pushes to both `main` and `development`, ensuring consistent CI coverage across active branches.
- Tau derivation, segment resolution, and shard metadata assembly gained helper-based refactors so the lint complexity caps are met without altering runtime behaviour or validations.
- Smoke suite now executes on direct pushes to `development`, mirroring pull-request coverage for the branch.
- Refactored reweighting, MSM, DeeptiCA, replica-exchange, and reporting modules to import shared constants instead of hard-coded literals.
- Standardised epsilon/tolerance guards, display scales, and energy thresholds by referencing the shared constants module.
- Diagnostics now reuse `features.deeptica.core.pairs.build_pair_info` for uniform pair statistics while keeping bias-aware reporting intact.
- Public trainer API continues to live at `pmarlo.features.deeptica_trainer` via the new package re-export.
- Optimised the reweighting kernel to reuse in-place NumPy buffers, reducing temporary allocations while validating bias lengths.


## Fixed
- Reordered `MSMDiscretizationResult` dataclass fields so mandatory analysis outputs (counts, transition matrices, etc.) register correctly when the module import runs under Python 3.12.
- Restored Poetry metadata (`name`, `description`, `authors`) so `poetry install --with dev,tests` operates in package mode without validation errors.
- CI now installs Poetry 2.1.3 to stay compatible with the checked-in `poetry.lock`, unblocking `poetry install --with dev,tests` on GitHub runners.
- GitHub Actions now clears `.testmondata` before running pytest, preventing stale testmon caches from causing readonly SQLite failures or xdist worker crashes.
- Aligned MSM discretizer `fit`/`transform` signatures with feature-schema-aware callers so datasets can pass metadata without triggering `TypeError`, while maintaining schema validation during transformations.
- Corrected `expected_pairs(...)` to keep per-shard stride alignment when zero-length segments are present, matching the simulated transition counts in integration tests.
- Tightened typing across reweighting, CV validation, and discretiser helpers so mypy passes cleanly while keeping runtime behaviour unchanged.
- Added `deeptime` to the dedicated tests dependency group so deeptime-backed unit suites stay active on CI instead of skipping.
- DeepTICA training tests now import helpers from `pmarlo.ml.deeptica.trainer`, bypassing the deprecated compatibility shim that previously raised during CI when extras were missing.
- Tests dependency group now installs the CPU `torch` build so DeepTICA smoke tests have the ML stack available on GitHub runners.
- Trajectory ingestion now drops frames with NaN/Inf CV values before shard emission, preventing invalid samples from reaching clustering.


<a id='changelog-0.97.0'></a>
# 0.97.0 — 2025-10-16

## Added

- STEPS.MD distills the experiment plan into phased implementation steps for the app_usecase workflow.
- Experiment input/output scaffolding under `app_intputs/experiments/` and `app/experiment_outputs/` with guidance files describing required shards, configs, manifests, and artefacts (including NPZ/JSON templates).
- Deterministic config templates (`transform_plan.yaml`, `discretize.yaml`, `reweighter.yaml`, `msm.yaml`) encoding DeepTICA, TRAM, and MSM guardrails for the E0/E1/E2 workflows.
- Experiment helper module (`example_programs/app_usecase/app/experiment/common.py`) resolving manifests, shard metadata, and config bundles for scripted runs.
- CLI runner framework (`example_programs/app_usecase/app/experiment/runner.py`) plus dedicated scripts for E0/E1/E2 that execute analysis, compute weight diagnostics, and emit acceptance reports under `experiment_outputs/`.
- Guardrail evaluation hooks that calculate ESS, SCC coverage, diagonal mass, and empty-state fractions against experiment-specific thresholds, reporting violations directly in `weights_summary.json`, `msm_summary.json`, and acceptance reports.
- DeepTICA orchestration now propagates lag fallbacks, records training diagnostics, enforces pair-count thresholds, and surfaces skip/failure reasons in analysis summaries and acceptance reports.
- Acceptance report generator consolidates weight/MSM/DeepTICA status and lists emitted artefacts (bundle, debug summaries, optional FES plots) to speed manual review.
- Unit tests added for experiment runner helpers covering weight guardrails, DeepTICA summarisation, and acceptance reporting without requiring full datasets.
- Headless analysis runner (`example_programs/app_usecase/app/headless.py`) to mirror Streamlit workflows from the terminal and print/save debug summaries.
- Dataset debug exporter (`pmarlo.analysis.debug_export`) that writes transition counts, MSM arrays, diagnostics, and configuration summaries for each build.
- Feature validation stage (`pmarlo.analysis.validate_features`) capturing column statistics and emitting `feature_stats.json` artefacts for debug bundles.
- Discretizer fingerprint now records the fitted feature schema so downstream bundles surface exact CV column metadata.
- State assignment validation raises `NoAssignmentsError` when clustering yields no labels and persists per-split `state_ids.npy` / `valid_mask.npy` in debug bundles.
- Expected MSM transition pairs now recorded via `expected_pairs(...)` with per-shard stride metadata in debug summaries.

- Introduced `pmarlo.constants` as the single source of project-wide physical, numeric, and domain defaults for reuse across subsystems.

- Pre-clustering diagnostics that log CV statistics, flag non-finite frames, and emit a `cv_distribution.png` histogram for quick visual inspection.
- Guard in the MSM clustering pipeline that verifies the requested number of microstates matches the unique labels returned by K-means/MiniBatchKMeans.
- Default analysis configuration now requests 20 microstates (down from 150) across headless/UI flows to better match modest datasets.
- Analyses now propagate `kmeans_kwargs={'n_init': 50}` so every K-means invocation retries with 50 restarts for more stable clustering.
- Headless analysis runner now aborts with a clear error message when `summary.json` reports `analysis_healthy=false`, echoing the recorded guardrail violations.
- Standalone `scripts/diagnose_cvs.py` utility to load trajectories, compute Rg/RMSD_ref, and dump detailed pandas/matplotlib diagnostics without triggering MSM/FES builds.

- Core DeepTICA trainer package modules (`config`, `loops`, `sampler`, `schedulers`, `trainer`) and focused unit tests under `tests/unit/features/deeptica/core/` covering each helper.
- Lightweight README for `src/pmarlo/features/deeptica/core/` documenting module responsibilities.

- Expanded reweighter unit coverage for TRAM aliasing, input immutability, and bias validation regressions.

## Changed

- Streamlit backend now captures per-build analysis summaries, persists them under `analysis_debug/`, and surfaces warning counts in stored artifacts.
- Workspace layout prepares a dedicated `analysis_debug` directory so raw analysis data lands in a predictable location alongside bundles.
- Captured discretizer fingerprints plus tau metadata per analysis build; UI and CLI now surface stride-derived effective tau and alert when overrides drift.
- Added synthetic MSM/CLI/Hypothesis tests plus compare_debug_bundles utility to guard against regression in fingerprint and transition statistics.
- MSM discretisation now aborts when post-whitening CVs contain non-finite values or zero variance columns, logging column statistics for every split.
- KMeans/Grid discretizers persist the training feature schema and refuse to transform splits whose names or order diverge, raising `FeatureMismatchError` with detailed differences.
- GitHub Actions test workflow now runs on pushes to both `main` and `development`, ensuring consistent CI coverage across active branches.
- Tau derivation, segment resolution, and shard metadata assembly gained helper-based refactors so the lint complexity caps are met without altering runtime behaviour or validations.
- Smoke suite now executes on direct pushes to `development`, mirroring pull-request coverage for the branch.

- Refactored reweighting, MSM, DeeptiCA, replica-exchange, and reporting modules to import shared constants instead of hard-coded literals.
- Standardised epsilon/tolerance guards, display scales, and energy thresholds by referencing the shared constants module.

- Diagnostics now reuse `features.deeptica.core.pairs.build_pair_info` for uniform pair statistics while keeping bias-aware reporting intact.
- Public trainer API continues to live at `pmarlo.features.deeptica_trainer` via the new package re-export.

- Optimised the reweighting kernel to reuse in-place NumPy buffers, reducing temporary allocations while validating bias lengths.

## Fixed

- Reordered `MSMDiscretizationResult` dataclass fields so mandatory analysis outputs (counts, transition matrices, etc.) register correctly when the module import runs under Python 3.12.
- Restored Poetry metadata (`name`, `description`, `authors`) so `poetry install --with dev,tests` operates in package mode without validation errors.
- CI now installs Poetry 2.1.3 to stay compatible with the checked-in `poetry.lock`, unblocking `poetry install --with dev,tests` on GitHub runners.
- GitHub Actions now clears `.testmondata` before running pytest, preventing stale testmon caches from causing readonly SQLite failures or xdist worker crashes.
- Aligned MSM discretizer `fit`/`transform` signatures with feature-schema-aware callers so datasets can pass metadata without triggering `TypeError`, while maintaining schema validation during transformations.
- Corrected `expected_pairs(...)` to keep per-shard stride alignment when zero-length segments are present, matching the simulated transition counts in integration tests.
- Tightened typing across reweighting, CV validation, and discretiser helpers so mypy passes cleanly while keeping runtime behaviour unchanged.
- Added `deeptime` to the dedicated tests dependency group so deeptime-backed unit suites stay active on CI instead of skipping.
- DeepTICA training tests now import helpers from `pmarlo.ml.deeptica.trainer`, bypassing the deprecated compatibility shim that previously raised during CI when extras were missing.
- Tests dependency group now installs the CPU `torch` build so DeepTICA smoke tests have the ML stack available on GitHub runners.

- Trajectory ingestion now drops frames with NaN/Inf CV values before shard emission, preventing invalid samples from reaching clustering.

## Changelog Entry

## Fixed
- Fixed comprehensive mypy type errors across multiple modules to achieve zero mypy errors for the type tox environment
- Resolved no-redef issues by using distinct names for fallback implementations (EnhancedMSM, prepare_structure)
- Fixed valid-type issues by implementing proper TYPE_CHECKING patterns for conditional type definitions
- Eliminated protocol instantiation errors by using proper type casting and constructor patterns
- Fixed union-attr issues by adding proper None checks before calling methods like `.lower()`
- Resolved arg-type and return-type mismatches in replica_exchange.py by normalizing Path/str types and fixing return statements
- Fixed mixed-type issues in transform/apply.py by adding proper imports, type annotations, and argument handling

## Technical Details
- **No-redef fixes**: Renamed stub classes (e.g., `_EnhancedMSMStubClass`) and removed duplicate function definitions to avoid mypy redefinition errors
- **Valid-type fixes**: Used `TYPE_CHECKING` blocks to create type-only aliases (e.g., `EnhancedMSMType`) for conditional class definitions
- **Protocol instantiation fixes**: Applied constructor casting with `cast(Any, Constructor)` pattern when protocols masked real `__init__` signatures
- **Union-attr fixes**: Added explicit None checks and type guards before calling methods on potentially None values
- **Arg-type fixes**: Normalized `str | Path | None` to `str` using assertions and `str()` conversion in ReplicaExchange.from_config
- **Return-type fixes**: Fixed missing return values in run_simulation method by returning appropriate `List[str]` values
- **Transform/apply fixes**: Added missing imports, fixed function signatures, and corrected argument passing for MSM analysis functions
- **Type annotation improvements**: Added proper type annotations to dictionaries and variables to resolve object type inference issues

<a id='changelog-0.96.0'></a>
# 0.96.0 — 2025-10-16

### Fixed
- Reworked DeepTICA trainer metric helpers so implicit scalars and vectors coerce to floats safely and `mypy` can verify curriculum bookkeeping.
- Normalised EnhancedMSM protocol usage across the MSM pipeline, experiments, and transforms so constructors and factories accept the documented keyword arguments during type checking.
- Tightened demultiplexing metadata helpers and transform step handler registration to return concrete types that satisfy the typing gate.

ï»¿### Fixed
- Declared stable DeepTICA trainer aliases and history helpers so numpy-derived arrays and curriculum settings satisfy the typing gate.
- Tightened demultiplexing metadata, shard ID, and transform utilities to return concrete types and align futures bookkeeping for mypy.
- Cleaned DeepTICA facade, demux stubs, and tests so flake8 passes without suppressing project defaults.

ï»¿## Fixed
- Avoided post-init mutation in TrainerConfig so frozen CurriculumConfig instances accept normalised fields and keep caller-defined tau order without raising.

## Added

- `pmarlo.features.deeptica.core.trainer_api.train_deeptica_pipeline` orchestrates feature prep, pair building, training, and whitening while returning `TrainingArtifacts` for callers.
- Unit test `tests/unit/features/deeptica/core/test_trainer_api.py` exercises the new pipeline with stubbed curriculum trainer dependencies.

- Shared sampler package at pmarlo.samplers exposing BalancedTempSampler for both feature and trainer layers.
- Central pair-construction helpers under pmarlo.pairs.core with unit coverage ensuring diagnostics and weights logic stay stable.

- STEPS.MD distills the experiment plan into phased implementation steps for the app_usecase workflow.
- Experiment input/output scaffolding under `app_intputs/experiments/` and `app/experiment_outputs/` with guidance files describing required shards, configs, manifests, and artefacts (including NPZ/JSON templates).
- Deterministic config templates (`transform_plan.yaml`, `discretize.yaml`, `reweighter.yaml`, `msm.yaml`) encoding DeepTICA, TRAM, and MSM guardrails for the E0/E1/E2 workflows.
- Experiment helper module (`example_programs/app_usecase/app/experiment/common.py`) resolving manifests, shard metadata, and config bundles for scripted runs.
- CLI runner framework (`example_programs/app_usecase/app/experiment/runner.py`) plus dedicated scripts for E0/E1/E2 that execute analysis, compute weight diagnostics, and emit acceptance reports under `experiment_outputs/`.
- Guardrail evaluation hooks that calculate ESS, SCC coverage, diagonal mass, and empty-state fractions against experiment-specific thresholds, reporting violations directly in `weights_summary.json`, `msm_summary.json`, and acceptance reports.
- DeepTICA orchestration now propagates lag fallbacks, records training diagnostics, enforces pair-count thresholds, and surfaces skip/failure reasons in analysis summaries and acceptance reports.
- Acceptance report generator consolidates weight/MSM/DeepTICA status and lists emitted artefacts (bundle, debug summaries, optional FES plots) to speed manual review.
- Unit tests added for experiment runner helpers covering weight guardrails, DeepTICA summarisation, and acceptance reporting without requiring full datasets.
- Headless analysis runner (`example_programs/app_usecase/app/headless.py`) to mirror Streamlit workflows from the terminal and print/save debug summaries.
- Dataset debug exporter (`pmarlo.analysis.debug_export`) that writes transition counts, MSM arrays, diagnostics, and configuration summaries for each build.
- Feature validation stage (`pmarlo.analysis.validate_features`) capturing column statistics and emitting `feature_stats.json` artefacts for debug bundles.
- Discretizer fingerprint now records the fitted feature schema so downstream bundles surface exact CV column metadata.
- State assignment validation raises `NoAssignmentsError` when clustering yields no labels and persists per-split `state_ids.npy` / `valid_mask.npy` in debug bundles.
- Expected MSM transition pairs now recorded via `expected_pairs(...)` with per-shard stride metadata in debug summaries.

- Introduced `pmarlo.constants` as the single source of project-wide physical, numeric, and domain defaults for reuse across subsystems.

- Pre-clustering diagnostics that log CV statistics, flag non-finite frames, and emit a `cv_distribution.png` histogram for quick visual inspection.
- Guard in the MSM clustering pipeline that verifies the requested number of microstates matches the unique labels returned by K-means/MiniBatchKMeans.
- Default analysis configuration now requests 20 microstates (down from 150) across headless/UI flows to better match modest datasets.
- Analyses now propagate `kmeans_kwargs={'n_init': 50}` so every K-means invocation retries with 50 restarts for more stable clustering.
- Headless analysis runner now aborts with a clear error message when `summary.json` reports `analysis_healthy=false`, echoing the recorded guardrail violations.
- Standalone `scripts/diagnose_cvs.py` utility to load trajectories, compute Rg/RMSD_ref, and dump detailed pandas/matplotlib diagnostics without triggering MSM/FES builds.

- Core DeepTICA trainer package modules (`config`, `loops`, `sampler`, `schedulers`, `trainer`) and focused unit tests under `tests/unit/features/deeptica/core/` covering each helper.
- Lightweight README for `src/pmarlo/features/deeptica/core/` documenting module responsibilities.

- Expanded reweighter unit coverage for TRAM aliasing, input immutability, and bias validation regressions.

## Changed

- `pmarlo.features.deeptica._full.train_deeptica` now delegates to the modular pipeline, reducing duplication and keeping optional dependency handling centralized.
- `pmarlo.features.deeptica.core.dataset.split_sequences` is exported for reuse and the DeepTICA core README documents the trainer API boundaries.

- Legacy DeepTICA facade now delegates pair building, training, and dataset wiring to the core modules, trimming bespoke helpers in _full.py.
- `train_deeptica` now routes training through `DeepTICACurriculumTrainer`, replacing the legacy Lightning fallback while keeping telemetry and whitening metadata.
- pmarlo.features.deeptica_trainer re-exports the canonical ml trainer, with lightweight wrappers plus iter_pair_batches for pairwise batching.
- Feature and shard samplers now hydrate through the shared implementation; PairBuilder.make_pairs resolves indices via build_pair_info for consistent diagnostics.

- Refactored the DeepTICACurriculumTrainer training workflow into modular helpers for dataset preparation, curriculum staging, and epoch aggregation to satisfy complexity limits without simplifying recorded metrics.
- Broke down the lint-hotspot helpers for diagnostics, FES, shard aggregation, demux dataset building, and the demultiplexing orchestrators so complexity checks pass without altering runtime behaviour.

- Promoted mlcolvar and scikit-learn to default dependencies and pruned extras that previously re-declared them.

- Locked tooling envs to Python 3.12 in tox and tightened project requires-python to hold back 3.13 auto-detection.
- Centralised flake8 configuration with targeted per-file ignores to unblock test imports while planning deeper refactors.

- Streamlit backend now captures per-build analysis summaries, persists them under `analysis_debug/`, and surfaces warning counts in stored artifacts.
- Workspace layout prepares a dedicated `analysis_debug` directory so raw analysis data lands in a predictable location alongside bundles.
- Captured discretizer fingerprints plus tau metadata per analysis build; UI and CLI now surface stride-derived effective tau and alert when overrides drift.
- Added synthetic MSM/CLI/Hypothesis tests plus compare_debug_bundles utility to guard against regression in fingerprint and transition statistics.
- MSM discretisation now aborts when post-whitening CVs contain non-finite values or zero variance columns, logging column statistics for every split.
- KMeans/Grid discretizers persist the training feature schema and refuse to transform splits whose names or order diverge, raising `FeatureMismatchError` with detailed differences.
- GitHub Actions test workflow now runs on pushes to both `main` and `development`, ensuring consistent CI coverage across active branches.
- Tau derivation, segment resolution, and shard metadata assembly gained helper-based refactors so the lint complexity caps are met without altering runtime behaviour or validations.
- Smoke suite now executes on direct pushes to `development`, mirroring pull-request coverage for the branch.

- Refactored reweighting, MSM, DeeptiCA, replica-exchange, and reporting modules to import shared constants instead of hard-coded literals.
- Standardised epsilon/tolerance guards, display scales, and energy thresholds by referencing the shared constants module.

- Diagnostics now reuse `features.deeptica.core.pairs.build_pair_info` for uniform pair statistics while keeping bias-aware reporting intact.
- Public trainer API continues to live at `pmarlo.features.deeptica_trainer` via the new package re-export.

- Optimised the reweighting kernel to reuse in-place NumPy buffers, reducing temporary allocations while validating bias lengths.

## Deprecated

- Importing pmarlo.features.samplers or the old trainer utilities now emits deprecation warnings guiding callers to the consolidated modules.

## Fixed

- Fixed mypy type errors in `replica_exchange/simulation.py`: resolved duplicate Simulation class definition and inconsistent function signatures for `prepare_system`, `production_run`, `build_transition_model`, `relative_energies`, and `plot_DG` to match their full implementation counterparts.
- Fixed multiple mypy type errors across the codebase:
  - Added proper type annotation for `__all__` in `ml/__init__.py`
  - Fixed `_require` argument type error in `shards/schema.py`
  - Fixed union-attr error in `reweight/reweighter.py`
  - Fixed no-any-return errors in `io/shard_id.py`, `data/shard.py`, and `analysis/discretize.py`
  - Added missing type annotations for variables in `replica_exchange/_simulation_full.py`, `features/deeptica_trainer.py`, and `features/deeptica/_full.py`
- Achieved significant progress in mypy type error resolution: reduced from 185+ errors to 121 errors (35% reduction)
- Fixed critical blocking issues: resolved module name conflicts and missing imports that were preventing comprehensive type checking
- Harmonized EnhancedMSM protocols, pipeline helpers, and demux fill-policy handling to match runtime defaults while satisfying the remaining mypy signature checks.
- Reduced cyclomatic complexity in replica-exchange feature extraction, exchange log parsing, and fallback loader initialization to satisfy flake8 C901 without altering runtime behavior.
- Refactored transform pipeline helpers to lower cyclomatic complexity while preserving Deep-TICA training behavior.
- Simplified shard selection, MSM/FES builders, and runner orchestration into reusable helpers for maintainability.
- Fixed the Streamlit example application startup by importing the missing typing helpers in `transform/apply` and exposing the demultiplexing writer type for static analysis.
- Corrected the demultiplexing stack: restored the `DemuxIntegrityError` re-export, fixed streaming state bookkeeping and parallel segment reads, and now record trajectory digests from frame data so repeated demux runs produce identical manifests.

- Prevented false mixed-kinds rejections in `pmarlo.data.aggregate` by using metadata-first shard kind inference and safe demux fallbacks when filenames lack demux hints.
- Added a regression test that exercises demux shards emitted from neutral filenames to ensure the MSM/FES builder accepts single-kind datasets.

- Resolved deeptica trainer type regressions by aliasing optional imports, normalising tuple defaults, and formalising dataset fallbacks for mypy.
- Removed unused typing artifacts and ensured StandardScaler fallbacks conform to the runtime interface expected by feature preparation.

- Reordered `MSMDiscretizationResult` dataclass fields so mandatory analysis outputs (counts, transition matrices, etc.) register correctly when the module import runs under Python 3.12.
- Restored Poetry metadata (`name`, `description`, `authors`) so `poetry install --with dev,tests` operates in package mode without validation errors.
- CI now installs Poetry 2.1.3 to stay compatible with the checked-in `poetry.lock`, unblocking `poetry install --with dev,tests` on GitHub runners.
- GitHub Actions now clears `.testmondata` before running pytest, preventing stale testmon caches from causing readonly SQLite failures or xdist worker crashes.
- Aligned MSM discretizer `fit`/`transform` signatures with feature-schema-aware callers so datasets can pass metadata without triggering `TypeError`, while maintaining schema validation during transformations.
- Corrected `expected_pairs(...)` to keep per-shard stride alignment when zero-length segments are present, matching the simulated transition counts in integration tests.
- Tightened typing across reweighting, CV validation, and discretiser helpers so mypy passes cleanly while keeping runtime behaviour unchanged.
- Added `deeptime` to the dedicated tests dependency group so deeptime-backed unit suites stay active on CI instead of skipping.
- DeepTICA training tests now import helpers from `pmarlo.ml.deeptica.trainer`, bypassing the deprecated compatibility shim that previously raised during CI when extras were missing.
- Tests dependency group now installs the CPU `torch` build so DeepTICA smoke tests have the ML stack available on GitHub runners.

- Trajectory ingestion now drops frames with NaN/Inf CV values before shard emission, preventing invalid samples from reaching clustering.

## Changelog Entry

## Fixed
- Fixed comprehensive mypy type errors across multiple modules to achieve zero mypy errors for the type tox environment
- Resolved no-redef issues by using distinct names for fallback implementations (EnhancedMSM, prepare_structure)
- Fixed valid-type issues by implementing proper TYPE_CHECKING patterns for conditional type definitions
- Eliminated protocol instantiation errors by using proper type casting and constructor patterns
- Fixed union-attr issues by adding proper None checks before calling methods like `.lower()`
- Resolved arg-type and return-type mismatches in replica_exchange.py by normalizing Path/str types and fixing return statements
- Fixed mixed-type issues in transform/apply.py by adding proper imports, type annotations, and argument handling

## Technical Details
- **No-redef fixes**: Renamed stub classes (e.g., `_EnhancedMSMStubClass`) and removed duplicate function definitions to avoid mypy redefinition errors
- **Valid-type fixes**: Used `TYPE_CHECKING` blocks to create type-only aliases (e.g., `EnhancedMSMType`) for conditional class definitions
- **Protocol instantiation fixes**: Applied constructor casting with `cast(Any, Constructor)` pattern when protocols masked real `__init__` signatures
- **Union-attr fixes**: Added explicit None checks and type guards before calling methods on potentially None values
- **Arg-type fixes**: Normalized `str | Path | None` to `str` using assertions and `str()` conversion in ReplicaExchange.from_config
- **Return-type fixes**: Fixed missing return values in run_simulation method by returning appropriate `List[str]` values
- **Transform/apply fixes**: Added missing imports, fixed function signatures, and corrected argument passing for MSM analysis functions
- **Type annotation improvements**: Added proper type annotations to dictionaries and variables to resolve object type inference issues

<a id='changelog-0.0.69'></a>
# 0.0.69 — 2025-09-22

REMD: wire end-to-end seeding; app auto-seed per shard; record in provenance.

- API `run_replica_exchange` accepts `random_seed`/`random_state` and forwards to `RemdConfig`.
- App: Simulation Seed mode (fixed | auto | none). Auto generates a unique 32â€‘bit seed per run and logs it.
- Run directories now include the seed (e.g., `run-YYYYMMDD-HHMMSS-seed123`).
- Shard provenance `source` includes `sim_seed` and `seed_mode`.
- Added unit/integration tests for seed propagation and determinism.

Also:
- Add robust resume/chaining: optional checkpoint/PDB restarts, jittered restarts, and safer checkpoint frequency.
- Add diversified starting conditions in the app: Initial PDB, Last frame of run, Random highâ€‘T frame; optional velocity reseed.
- Replica-exchange diagnostics (ladder suggestion, acceptance, diffusion); UI controls for exchange frequency; diagnostics panel with sparkline.
- REMD: honor explicit temperature vectors; validate increasing >0; persist ladder in provenance.json and temps.txt; record schedule mode.
- App: temperature schedule selector (auto-linear, auto-geometric, custom) with Apply toggle; ladder preview and validation; applied to run config.
- Utility: stable geometric ladder generator with tests.

## Added

- created possibility with batch algorithm testing with kubernetes docker desktop kubeadm administrator kernel
- done k8s for the kubernetes with local server

- Optional solvation step that adds an explicit water box when none is present.

## Changed

- made another file for the kubernetes suite with experiment possibility
- changed docker file
- moved md files to the separated directory

- Water molecules are now preserved during protein preparation by default.
<a id='changelog-0.0.36'></a>
# 0.0.36 — 2025-08-30

## Added

- Unified progress callback/reporting (`pmarlo.progress.ProgressReporter`) with ETA and rate limiting.
- Callback kwarg aliases normalized via `coerce_progress_callback`.
- Transform plan serialization helpers: `to_json`, `from_json`, `to_text`.
- Aggregate/build progress events from `pmarlo.transform.runner.apply_plan`.
- Example usage in `example_programs/all_capabilities_demo.py` printing progress.
- Tests for progress reporting, plan serialization, and transform runner events.

## Changed

- `api.run_replica_exchange` accepts `**kwargs` and passes `progress_callback` to the simulation.
- `ReplicaExchange.run_simulation` emits stage events (`setup`, `equilibrate`, `simulate`, `exchange`, `finished`).
- `transform.build.build_result` optionally accepts `progress_callback` to surface aggregate events during transforms.

<a id='changelog-0.14.0'></a>
# 0.14.0 — 2025-08-08

## Added

- psutils for the memory management.

## Changed

- changes in the pyproject.toml and experiments.
- KPIs for the methods and algorithm testing suite upgrades.
- docker now has a lock generations and not just distribution usage.
- made deduplication effort in the probability calculation and logging info from all the modules

<a id='changelog-0.13.0'></a>
# 0.13.0 — 2025-08-08

## Added

- Whole suite for the experimenting with the algoritms(simulation, replica exchange, markov state model) in the docker containers to make them separately run.

<a id='changelog-0.12.0'></a>
# 0.12.0 — 2025-08-08

## Added

- Added the **\[tool.scriv]** section to `pyproject.toml`, setting the format to `md`, the output file to `CHANGELOG.md`, and the fragments directory to `changelog.d`.
