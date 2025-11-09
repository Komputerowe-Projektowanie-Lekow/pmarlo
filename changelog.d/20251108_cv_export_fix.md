## fixed

- Fixed `_export_cv_model` method in `pmarlo_webapp/app/backend/training.py` that was trying to pickle.load a JSON file, causing silent CV bundle export failures during training
- Added `weights_only=False` parameter to `torch.load()` calls in `src/pmarlo/features/deeptica/_full.py` for PyTorch 2.6+ compatibility
- Improved error handling in CV model export to raise informative errors instead of silently failing
- Added strict feature-count validation to both the webapp backend and `export_cv_bundle.py`, preventing CV bundle exports when the trained model does not match `feature_spec.yaml`
- Streamlit backend now auto-discovers DeepTICA bundles in `app_output/models`, ensuring newly trained models immediately appear in the Model Preview picker even if the state file was out of sync
- Sampling tab now auto-generates the CV bias bundle when you choose a model: the backend recreates missing `deeptica_cv_model.*` files from the stored bundle metadata, surfaces errors if the shards are incompatible, and persists the bundle path for future runs
- Fixed cross-platform path conversion so Windows paths captured in `state.json` resolve correctly when the app runs either natively on Windows or inside WSL; this previously caused auto-export to fail with “Model bundle is missing on disk” even when the file existed
- Improved `check_openmm_torch_available()` to load the TorchForce plugin via OpenMM when the standalone `openmmtorch` Python module is not importable, and to search additional directories (including those provided via `PMARLO_OPENMM_PLUGIN_DIR` / `OPENMM_PLUGIN_DIR`), eliminating spurious “openmm-torch is not installed” errors when the plugin lives in a sibling Conda/Mamba environment

## added

- Created `pmarlo_webapp/export_cv_bundle.py` utility script to manually export CV bias bundles from existing trained DeepTICA models
- Added comprehensive `pmarlo_webapp/CV_BIAS_REQUIREMENTS.md` documentation explaining the difference between CV-based and molecular feature-based models and their use cases
- Enhanced error messages in sampling tab to guide users when CV bias bundles are missing, including specific causes and solutions
- Sampling tab now exposes a feature profile selector with live compatibility messaging so users can request molecular shards directly from the UI
- Training tab surfaces feature profile metadata and compatibility badges for the selected shard batches

## changed

- Updated CV bundle export logic to properly load DeepTICAModel from separate `.pt`, `.scaler.pt`, and `.json` files instead of attempting to unpickle the bundle
- Modified model loading to handle architecture evolution with `strict=False` parameter and automatic key prefix remapping for backward compatibility
- Analysis tabs (MSM/FES, conformations, ITS/CK diagnostics, validation) now warn when mixed CV and molecular feature sets are selected, reducing accidental misuse of shards

