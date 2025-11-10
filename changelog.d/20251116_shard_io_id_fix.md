## fixed

- Ensured `tests/unit/io/test_shard_io.py` now derives the canonical shard ID from the provided run identifier before calling `write_shard`, so the helper-generated IDs match what `canonical_shard_id` expects and the shard IO regression no longer fails.
