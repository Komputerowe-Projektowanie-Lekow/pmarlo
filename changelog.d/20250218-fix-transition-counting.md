### Fixed
- Corrected `total_pairs_from_shards` to reuse `expected_pairs` so strided counting tallies every valid transition pair in debug summaries.
- Added regression coverage verifying strided pair predictions align with the counting algorithm.
