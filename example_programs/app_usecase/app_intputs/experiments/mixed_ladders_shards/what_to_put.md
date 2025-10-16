# Overlapping Ladder Shards (E1)

Prepare five `.npz` shards drawn from slightly shifted temperature ladders with overlap around a common reference temperature `T_ref`.

- Required files:
  - `shard_000.npz` … `shard_004.npz`: feature matrices plus any auxiliary metadata needed for TRAM/MBAR.
  - `manifest.yaml`: derive from `manifest.template.yaml`, documenting each shard’s ladder (β/T list), frame counts, stride, and RNG seed.
  - `shard_XXX.json`: shard metadata following `shard_meta.template.json`, with ladder info recorded under `provenance.temperature_ladder_K`.
- Dataset expectations:
  - Adjacent ladders must overlap at least one temperature near `T_ref` so reweighting can converge (record `T_ref` explicitly).
  - Provide per-shard energy/time series fields needed by the reweighter (e.g., reduced energies at all ladder temperatures).
  - Ensure each shard length ≥ DeepTICA lag to keep `pairs_total > 0`; note lag choice inside the manifest comments.
  - `.npz` files should respect `../npz_schema.md`; include additional arrays (e.g., `energy_ladder`) if TRAM requires them and document keys in the manifest.
