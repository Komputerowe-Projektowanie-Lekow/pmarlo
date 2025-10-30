## Fixed

- Raised explicit errors when trajectory alignment fails instead of silently
  returning unaligned data, ensuring data issues are detected early in the
  workflow by `_align_trajectory`.
- Normalized MDAnalysis I/O dependency checks to raise the dedicated
  `TrajectoryIOError`/`TrajectoryWriteError` instead of leaking a
  `ModuleNotFoundError`, maintaining fail-fast behaviour when optional
  backends are unavailable.
