### Fixed
- Preserve complex MSM eigenvalues when computing implied timescales so `safe_timescales` no longer drops the imaginary
  component and incorrectly reports stable negative eigenmodes.
- Added regression coverage for complex-valued eigenvalues in the implied timescale math tests.
