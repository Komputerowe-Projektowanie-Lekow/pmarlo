## fixed

- Streamlit conformations workflow now reconstructs missing shard `source.range`
  metadata by tracking shard ordering per run/trajectory, so legacy shard sets
  that never recorded frame offsets can still extract representative structures
  instead of failing with “frame range” errors.
- Added regression tests around the range-derivation helper to ensure future
  changes keep the inferred spans stable across runs and replicas.
