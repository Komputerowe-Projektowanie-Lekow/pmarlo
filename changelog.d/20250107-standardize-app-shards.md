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

### Changed
- **OpenMM System Creation**: `create_system()` in `src/pmarlo/replica_exchange/system_builder.py` now accepts optional CV model parameters and integrates `TorchForce` from openmm-torch when provided
- **API Expansion**: `run_replica_exchange()` in `src/pmarlo/api.py` now accepts `cv_model_path`, `cv_scaler_mean`, and `cv_scaler_scale` parameters for CV-biased sampling
- **Enhanced Training Diagnostics**: Logs now show both pre-clip and post-clip gradient norms (`grad_preclip` and `grad_postclip`), clipping threshold, expected pairs, and tau schedule for curriculum training
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
- **Fixed GitHub Actions test collection failure**: Added `tests/integration` to `testpaths` in `pyproject.toml` and added `@pytest.mark.integration` markers to all integration test files to enable proper test discovery.
- **Fixed GitHub Actions pytest exit code 5 issue**: Updated all GitHub Actions workflow test commands to explicitly specify test directories (`tests/unit tests/devtools tests/integration`) instead of relying on marker filters alone, preventing pytest from collecting unmarked tests incorrectly.
- **Fixed tomli import error in Python 3.11+**: Updated `scripts/check_extras_parity.py` to use `tomllib` from the standard library for Python 3.11+ instead of the external `tomli` package, fixing test collection errors in devtools tests.

### Testing
- Loaded every example shard with `pmarlo.data.shard_io.load_shard_meta`.
- Verified MSM builds successfully with valid transition matrices (50x50, row-stochastic) and stationary distributions from 2000-frame shard datasets.
- Tested complete workflow including normal analysis, DeepTICA integration, and all additional features.
