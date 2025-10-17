### Changed
- Standardized the example app shard metadata to match the canonical PMARLO shard schema (explicit periodic flags, canonical provenance keys, and float32-aligned `dt_ps`).
- Added `lightning` (pytorch-lightning) to `mlcv` optional dependencies in `pyproject.toml` as required by `mlcolvar`.

### Fixed
- Fixed critical bug in `example_programs/app_usecase/app/backend.py` where analysis guardrails were checked against pre-clustering debug data, causing all analyses to fail with "total_pairs_lt_5000" error before the MSM build could complete.
- Modified guardrail validation to check post-clustering MSM build results instead of pre-clustering debug data, allowing proper MSM construction with transition matrix and free energy surface generation.
- Resolved analysis guardrail failures in the example app by ensuring shard manifests declare full provenance required by `load_shard_meta`.
- Fixed dependency management issue where `mlcolvar` required `lightning` (pytorch-lightning) but it was not specified in optional dependencies, causing "No module named 'mlcolvar'" errors.

### Testing
- Loaded every example shard with `pmarlo.data.shard_io.load_shard_meta`.
- Verified MSM builds successfully with valid transition matrices (50x50, row-stochastic) and stationary distributions from 2000-frame shard datasets.
- Tested complete workflow including normal analysis, DeepTICA integration, and all additional features.
