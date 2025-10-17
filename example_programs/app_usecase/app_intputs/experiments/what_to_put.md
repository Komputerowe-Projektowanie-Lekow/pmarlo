# Experiments Input Root

This directory hosts all inputs needed to reproduce the E0/E1/E2 workflows described in `plan.md`.

Subdirectories created:

- `same_temp_shards/`: homogeneous-temperature shards for the E0 baseline.
- `mixed_ladders_shards/`: overlapping-ladder shards for the E1 positive reweight scenario.
- `disjoint_ladders_shards/`: optional negative-control dataset to exercise failure guardrails.
- `configs/`: YAML files configuring transforms, discretization, reweighting, and MSM steps.
- `refs/long_run_Tref/`: reference ensemble used for acceptance comparisons.
- `shard_meta.template.json`: canonical JSON metadata schema for individual shards (paired with `.npz` files).
- `npz_schema.md`: required/optional array layout for shard `.npz` archives.

Populate each subdirectory following its own `what_to_put.md` so the orchestration code can locate data deterministically.
