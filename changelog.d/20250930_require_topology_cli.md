### Added
- Command-line conformations analysis now requires an explicit `--topology` PDB path and validates shard metadata before loading trajectories.

### Fixed
- The CLI aborts immediately when trajectory metadata is missing or files are absent instead of silently skipping structure exports.
