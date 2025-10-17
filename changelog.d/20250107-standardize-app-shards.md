### Added
- Real-time training metrics logging to JSON file during DeepTICA model training
- Training progress visualization in Model Training tab with live epoch updates and metrics curves
- Model Preview tab in the app for inspecting trained model architecture, parameters, and training history
- CV-informed sampling option to associate trained CV models with simulation runs
- Checkpoint directory tracking for trained models to access training progress and checkpoints
- `training_progress.json` file written during training with epoch-by-epoch metrics
- `get_training_progress()` method in backend to read real-time training status

### Changed
- Standardized the example app shard metadata to match the canonical PMARLO shard schema (explicit periodic flags, canonical provenance keys, and float32-aligned `dt_ps`).
- Added `lightning` (pytorch-lightning) to `mlcv` optional dependencies in `pyproject.toml` as required by `mlcolvar`.
- DeepTICACurriculumTrainer now writes real-time progress updates after each epoch
- TrainingResult dataclass now includes `checkpoint_dir` field
- SimulationConfig dataclass now includes `cv_model_bundle` field for CV-informed sampling
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

### Testing
- Loaded every example shard with `pmarlo.data.shard_io.load_shard_meta`.
- Verified MSM builds successfully with valid transition matrices (50x50, row-stochastic) and stationary distributions from 2000-frame shard datasets.
- Tested complete workflow including normal analysis, DeepTICA integration, and all additional features.
