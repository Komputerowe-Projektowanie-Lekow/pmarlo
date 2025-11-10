## fixed

- `emit_shards_from_trajectories` now derives shard IDs with the provenance
  `run_id` for both demux and replica outputs, matching the canonical ID rules
  enforced by `pmarlo.shards`. This prevents shard writers from failing with
  “shard_id ... does not match canonical ...” when run metadata is present.
- Added regression coverage to ensure future emitters continue to include the
  run-aware suffixes in their shard filenames.
