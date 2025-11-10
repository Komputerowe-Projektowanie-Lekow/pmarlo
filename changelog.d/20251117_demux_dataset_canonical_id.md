## fixed

- Reworked `tests/unit/data/test_demux_dataset.py` so the shard helper now derives IDs with the run-aware suffix enforced by `pmarlo.shards.canonical_shard_id`, preventing helper-generated shards from conflicting with the canonical validation in recent test cases.
