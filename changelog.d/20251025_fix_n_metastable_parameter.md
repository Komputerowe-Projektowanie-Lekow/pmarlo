### Fixed

- Fixed missing `n_metastable` parameter in conformations CLI and app backend that caused PCCA+ to always use 2 clusters instead of the user-specified value.
- The `conformations_cli.py` now properly passes `--n-metastable` argument to `find_conformations()`.
- The `backend.py` now properly passes `ConformationsConfig.n_metastable` to `find_conformations()`.

### Added

- Made `n_metastable` an explicit parameter in `find_conformations()` function signature (previously hidden in `**kwargs`).
- Added comprehensive documentation for `n_metastable` parameter explaining it controls the number of PCCA+ clusters.
- Added example usage in docstring showing how to increase the number of metastable states.

### Improved

- Enhanced API clarity by promoting `n_metastable` from hidden kwargs to explicit parameter with default value of 2.
- Users can now easily control PCCA+ clustering granularity without needing to know about the hidden kwargs interface.
