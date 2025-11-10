## fixed

- Require `run_labels` to match the length of `projection_data` inside `create_sampling_validation_plot`
  so malformed metadata now raises a `ValueError` before the visualization is drawn.

## changed

- Clarified the sampling validation legend in `src/pmarlo/reporting/plots.py` by documenting
  that standard traces use solid lines and metabiased traces use dashed lines, ensuring the
  legend title explicitly references the metabiased style.
