### Changed
- Relocated the Streamlit application from `example_programs/app_usecase` into the dedicated `pmarlo_webapp/app` package and refreshed user-facing run instructions.
- Added path migration helpers so saved state, analysis bundles, and shard metadata rebased cleanly onto the new workspace layout.
- Updated tests, utilities, and docs to import the `pmarlo_webapp` modules and to source example inputs from the new directory.
