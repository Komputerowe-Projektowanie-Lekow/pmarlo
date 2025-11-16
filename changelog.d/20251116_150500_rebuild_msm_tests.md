### added
- Added conceptual MSM rebuild tests to lock in lag handling, inactive-state treatment, and deterministic-cycle behavior for `KineticImportanceScore._rebuild_msm`.

### fixed
- `_rebuild_msm` now explicitly restricts MSM fitting to the largest connected component so disconnected states remain neutral with identity dynamics and zero stationary weight.
