### Fixed
- Record the built MSM state count and raise a `state_count_mismatch` guardrail when the transition matrix size disagrees with the declared discretizer fingerprint.
- Surface the MSM state count and run identifier in the app plot so mismatches are visible in the UI.
- Added regression coverage that exercises the guardrail path and verifies the analysis debug summary captures the mismatch metadata.
