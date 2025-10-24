### Added
- Enabled multi-start clustering in `cluster_microstates` by honouring the `n_init` restart count and selecting the run with the lowest inertia.

### Changed
- Exposed a dedicated `kmeans_n_init` control for the conformations Streamlit app and CLI so users can configure clustering restarts without tripping backend validation.

### Fixed
- Prevented unsupported `n_init` kwargs from reaching deeptime estimators, eliminating the TypeError raised during conformations analysis.
