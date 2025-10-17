# Same-Temperature Shards (E0)

Generate five `.npz` shards that were all simulated at the identical temperature ladder (baseline, no reweight).

- Required files:
  - `shard_000.npz` … `shard_004.npz`: consistent feature arrays, deterministic RNG seeds recorded in metadata.
  - `manifest.yaml`: start from `manifest.template.yaml` and list shard filenames, shared temperature value, total frames per shard, and RNG seeds.
  - `shard_XXX.json`: per-shard metadata following `shard_meta.template.json`.
- Dataset expectations:
  - Each shard length ≥ lag used for DeepTICA (target 4k–10k frames after any stride reductions).
  - Include per-shard provenance (simulation ID, temperature tag) inside the manifest to support diagnostics.
  - Ensure discretization-ready columns (CVs) align with the plan’s analysis pipeline schema.
  - `.npz` files must follow `../npz_schema.md` (keys: `X`, `t_index`, `dt_ps`, optional energy/bias).
