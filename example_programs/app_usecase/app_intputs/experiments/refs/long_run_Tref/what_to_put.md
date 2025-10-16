# Reference Ensemble at `T_ref`

Store reference data used for truth checks against the experiment outputs.

- Required files:
  - `summary.npz`: aggregated statistics from a long unbiased simulation at `T_ref` (stationary distribution, FES grid, energy stats).
- Optional helpers:
  - `metadata.json`: provenance for the reference run (simulation engine, RNG seed, trajectory length).
  - `README.md`: brief description of how the reference dataset was generated.

Ensure the data schema matches what `common.py` will consume for comparison (e.g., keys for stationary probabilities, FES grid axes). Keep files deterministic so tests can assert acceptance thresholds.
