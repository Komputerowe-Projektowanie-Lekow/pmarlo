### Added
- Conformations analysis now accepts precomputed DeepTICA projections via the new `cv_method="deeptica"` option, including optional whitening metadata reuse.

### Changed
- Updated the Streamlit conformations panel to let users choose between TICA and DeepTICA CVs and provide DeepTICA projection/metadata paths.

### Fixed
- DeepTICA-driven conformations runs now raise explicit errors when required projection assets are missing instead of silently reverting to TICA.
- Workflow backend raises descriptive errors when shard metadata contains missing or non-numeric values rather than silently substituting defaults.
