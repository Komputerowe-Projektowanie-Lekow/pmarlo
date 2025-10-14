## Added
- Headless analysis runner (`example_programs/app_usecase/app/headless.py`) to mirror Streamlit workflows from the terminal and print/save debug summaries.
- Dataset debug exporter (`pmarlo.analysis.debug_export`) that writes transition counts, MSM arrays, diagnostics, and configuration summaries for each build.
- Feature validation stage (`pmarlo.analysis.validate_features`) capturing column statistics and emitting `feature_stats.json` artefacts for debug bundles.
- Discretizer fingerprint now records the fitted feature schema so downstream bundles surface exact CV column metadata.
- State assignment validation raises `NoAssignmentsError` when clustering yields no labels and persists per-split `state_ids.npy` / `valid_mask.npy` in debug bundles.
- Expected MSM transition pairs now recorded via `expected_pairs(...)` with per-shard stride metadata in debug summaries.

## Changed
- Streamlit backend now captures per-build analysis summaries, persists them under `analysis_debug/`, and surfaces warning counts in stored artifacts.
- Workspace layout prepares a dedicated `analysis_debug` directory so raw analysis data lands in a predictable location alongside bundles.
- Captured discretizer fingerprints plus tau metadata per analysis build; UI and CLI now surface stride-derived effective tau and alert when overrides drift.
- Added synthetic MSM/CLI/Hypothesis tests plus compare_debug_bundles utility to guard against regression in fingerprint and transition statistics.
- MSM discretisation now aborts when post-whitening CVs contain non-finite values or zero variance columns, logging column statistics for every split.
- KMeans/Grid discretizers persist the training feature schema and refuse to transform splits whose names or order diverge, raising `FeatureMismatchError` with detailed differences.

## Fixed
- Reordered `MSMDiscretizationResult` dataclass fields so mandatory analysis outputs (counts, transition matrices, etc.) register correctly when the module import runs under Python 3.12.
