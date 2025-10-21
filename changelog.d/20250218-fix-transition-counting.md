### Fixed
- Corrected `total_pairs_from_shards` to reuse `expected_pairs` so strided counting tallies every valid transition pair in debug summaries.
- Added regression coverage verifying strided pair predictions align with the counting algorithm.
- Taught `compute_analysis_debug` to accept discrete trajectories supplied as mapping values so lagged pair detection no longer drops split-labelled datasets.
