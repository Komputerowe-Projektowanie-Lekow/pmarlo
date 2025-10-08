## Added
- Headless analysis runner (`example_programs/app_usecase/app/headless.py`) to mirror Streamlit workflows from the terminal and print/save debug summaries.
- Dataset debug exporter (`pmarlo.analysis.debug_export`) that writes transition counts, MSM arrays, diagnostics, and configuration summaries for each build.

## Changed
- Streamlit backend now captures per-build analysis summaries, persists them under `analysis_debug/`, and surfaces warning counts in stored artifacts.
- Workspace layout prepares a dedicated `analysis_debug` directory so raw analysis data lands in a predictable location alongside bundles.
- Captured discretizer fingerprints plus tau metadata per analysis build; UI and CLI now surface stride-derived effective tau and alert when overrides drift.
