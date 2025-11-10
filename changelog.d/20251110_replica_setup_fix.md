## fixed

- Make the shard summary demo helper generate the canonical `T…_run-…` identifiers so different-run shards no longer raise the canonicalization `ValueError`.
- Extend the OpenMM unit stub used by the tests with all frequently used units (e.g., `nanometer`, `kilojoule_per_mole`, `bar`, `amu`, `femtoseconds`, `picosecond`, `Unit`, …) so replica-setup and force-related tests no longer fail with missing attribute errors when OpenMM itself is unavailable.
