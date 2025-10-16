# Disjoint Ladder Shards (E2 – Negative Control)

Optionally generate five `.npz` shards whose temperature ladders have **no overlap** near `T_ref`. This dataset should trigger guardrail failures during reweighting.

- Required files:
  - `shard_000.npz` … `shard_004.npz`: same schema as E1 shards so the pipeline can attempt to merge them.
  - `manifest.yaml`: adapt `manifest.template.yaml`, clearly labeling each ladder’s temperature range and noting that overlap with `T_ref` is absent by design.
  - `shard_XXX.json`: metadata files based on `shard_meta.template.json` with `provenance.expected_failure` describing the intended guardrail outcome.
- Dataset expectations:
  - Make the lack of overlap unambiguous (e.g., ladders offset by ≥ 20–30 K) so ESS collapses during TRAM/MBAR.
  - Keep shard lengths comparable to E1 so any failure is attributable to thermodynamic incompatibility, not data sparsity.
  - Add a comment describing the expected failure mode (TRAM error or ESS ≈ 0) to help interpret acceptance_report.md.
  - `.npz` content must follow `../npz_schema.md`; if optional arrays are skipped, leave them empty rather than omitting keys to simplify downstream checks.
