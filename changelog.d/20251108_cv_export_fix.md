## fixed

- Fixed critical FES component selection bug in `src/pmarlo/analysis/fes.py` where free energy surfaces were always computed using the first two CV columns instead of selecting components by variance, causing incorrect FES plots when using mlcolvar/DeepTICA outputs where component importance varies
- Conformations analysis now tolerates shard metadata that omits `source.range` by reusing the recorded global spans when mapping frames back to trajectories, preventing the downstream error described in the webapp logs
- Ensured the default FES builder in `src/pmarlo/transform/build.py` now reuses the variance-based `select_highest_variance_components` helper so recorded MSM/FES runs also choose the most informative CV pair instead of the original hardcoded first two axes
- Fixed `_export_cv_model` method in `pmarlo_webapp/app/backend/training.py` that was trying to pickle.load a JSON file, causing silent CV bundle export failures during training
- Added `weights_only=False` parameter to `torch.load()` calls in `src/pmarlo/features/deeptica/_full.py` for PyTorch 2.6+ compatibility
- Improved error handling in CV model export to raise informative errors instead of silently failing
- Added strict feature-count validation to both the webapp backend and `export_cv_bundle.py`, preventing CV bundle exports when the trained model does not match `feature_spec.yaml`
- Streamlit backend now auto-discovers DeepTICA bundles in `app_output/models`, ensuring newly trained models immediately appear in the Model Preview picker even if the state file was out of sync
- Sampling tab now auto-generates the CV bias bundle when you choose a model: the backend recreates missing `deeptica_cv_model.*` files from the stored bundle metadata, surfaces errors if the shards are incompatible, and persists the bundle path for future runs
- Fixed cross-platform path conversion so Windows paths captured in `state.json` resolve correctly when the app runs either natively on Windows or inside WSL; this previously caused auto-export to fail with "Model bundle is missing on disk" even when the file existed
- Improved `check_openmm_torch_available()` to load the TorchForce plugin via OpenMM when the standalone `openmmtorch` Python module is not importable, and to search additional directories (including those provided via `PMARLO_OPENMM_PLUGIN_DIR` / `OPENMM_PLUGIN_DIR`), eliminating spurious "openmm-torch is not installed" errors when the plugin lives in a sibling Conda/Mamba environment
- Fixed adaptive FES grid sizing so the bin count can shrink below the previous 40-bin floor and loop longer when needed, which keeps the empty-bin fraction below the target even when the data occupy a narrow range
- Fixed test expectation in `tests/analysis/test_fes.py` where `ensure_fes_inputs_whitened` was expected to return True for already-whitened data, but correctly returns False to indicate no new whitening was applied
- Fixed `tests/unit/replica_exchange/test_single_temp_production.py` so its configuration stub now mirrors `RemdConfig` (including `forcefield_files` and `target_accept`) and the step-count assertion tolerates the extra production churn triggered by the single-temperature path

## added

- Added `select_highest_variance_components` function to `src/pmarlo/analysis/fes.py` that intelligently selects CV components based on variance for FES computation, ensuring mlcolvar/DeepTICA outputs are properly utilized by choosing the most informative components
- Created `pmarlo_webapp/export_cv_bundle.py` utility script to manually export CV bias bundles from existing trained DeepTICA models
- Added comprehensive `pmarlo_webapp/CV_BIAS_REQUIREMENTS.md` documentation explaining the difference between CV-based and molecular feature-based models and their use cases
- Enhanced error messages in sampling tab to guide users when CV bias bundles are missing, including specific causes and solutions
- Sampling tab now exposes a feature profile selector with live compatibility messaging so users can request molecular shards directly from the UI
- Training tab surfaces feature profile metadata and compatibility badges for the selected shard batches

## changed

- Updated CV bundle export logic to properly load DeepTICAModel from separate `.pt`, `.scaler.pt`, and `.json` files instead of attempting to unpickle the bundle
- Modified model loading to handle architecture evolution with `strict=False` parameter and automatic key prefix remapping for backward compatibility
- Analysis tabs (MSM/FES, conformations, ITS/CK diagnostics, validation) now warn when mixed CV and molecular feature sets are selected, reducing accidental misuse of shards
- Updated Streamlit renderers across the webapp (validation, MSM/FES, run discovery, CK+ITS tabs and shared helpers) to replace the deprecated `use_container_width` flag with the new `width` parameter so the UI keeps stretching layouts reliably without relying on removed options
- Reorganized the Sampling and Training tabs so their recorded-run/model loaders sit inside collapsible expanders at the top of each page, matching the existing pattern for other analysis tabs and making load controls easy to find.
- Added a configurable FES grid strategy option to the MSM/FES build (UI, CLI, and analysis backend) so adaptive histogram grids can be requested and persist in recorded build metadata
