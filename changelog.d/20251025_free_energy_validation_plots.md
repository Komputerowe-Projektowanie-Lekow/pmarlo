### Added
- Two new validation plotting functions in `pmarlo.reporting.plots`:
  - `plot_sampling_validation`: Shows 1D trajectory traces over histogram to validate sampling connectivity between metastable states
  - `plot_free_energy_2d`: Renders 2D FES contour plot on collective variable components
- New self-contained "Free Energy Validation" tab in the example app with:
  - Independent shard selection interface
  - On-demand TICA projection computation
  - Real-time FES calculation from selected shards
  - Side-by-side display of sampling connectivity and FES plots
- Wrapper functions in app diagnostics module (`create_sampling_validation_plot`, `create_fes_validation_plot`) to integrate library plots with Streamlit UI

### Changed
- Updated `example_programs/app_usecase/app/plots/diagnostics.py` to export new validation plot wrappers
- Refactored Free Energy Validation tab to compute validation metrics independently rather than relying on pre-computed MSM/FES analysis results
- Enhanced app workflow with standalone validation tool that allows users to verify sampling quality on-demand for any shard combination

### Fixed
- `plot_sampling_validation` now fails fast on missing trajectories, empty shards, or unknown colormaps instead of silently
  falling back, aligning reporting behavior with deterministic output requirements.
