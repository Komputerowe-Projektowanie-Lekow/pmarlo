Deep‑TICA Learning Path: Instrumentation and Usage

Overview
- PMARLO can optionally learn data‑driven collective variables (CVs) via Deep‑TICA.
- The transform builder now records detailed diagnostics about the learning step under the artifact key `mlcv_deeptica` inside build bundles.
- The example app shows two tabs: Original CVs and Deep‑TICA CVs. Deep‑TICA plots only render if learning actually applied.

Artifact Schema (`artifacts["mlcv_deeptica"]`)
- applied: bool — True if Deep‑TICA applied and learned CVs were used.
- skipped: bool — True if Deep‑TICA training was skipped.
- reason: one of "no_records", "no_pairs", "exception", "ok".
- lag_requested: int — the initially requested lag.
- lag_used: int — the actual lag used for pair construction (may differ with fallback).
- lag_fallback: list[int] | null — optional fallback ladder if configured.
- attempts: list[{lag, pairs_total}] — summary of attempts (requested lag and fallbacks).
- n_shards: int — number of input shards considered.
- frames_total: int — total frames across shards.
- pairs_total: int — total time‑lagged pairs across all shards.
- per_shard: list of {id, frames, pairs} for each shard.
- wall_time_s: float — wall time for the learning step.
- torch_device: str|null — e.g. "cpu" or "cuda" if available.
- seed: int — Deep‑TICA seed.
- If applied: additional fields may be present (model_dir, files, torchscript_sha256, cache_key).

Skip Reasons
- no_records: No shard records/X matrix available for pair construction.
- no_pairs: Pair construction produced zero pairs (e.g., shards too short for selected lag).
- exception: An internal error occurred; learning was skipped (dataset left unchanged). Error text is recorded under `error`.
- ok: Learning completed and was applied.

Ensuring Data Sufficiency
- For uniform time (no bias), each shard must satisfy `len(shard) > lag` to yield any pairs.
- For scaled time (with per‑frame bias V(s)), pairs are computed via scaled time t′. While less strict than uniform time, very short shards still may not yield pairs.
- Recommended:
  - Start with small lags (e.g., 2–5 frames) to validate the pipeline.
  - Inspect `per_shard` diagnostics to confirm non‑zero pairs per shard.
  - Aggregate more shards or reduce lag if pairs_total == 0.

Picking Lag
- Choose lag relative to the correlation time of your fastest meaningful process.
- When in doubt, sweep a small set (e.g., 5, 10, 15) and examine training stability and resulting FES/MSM consistency.

Optional Lag Fallback
- You can enable a fallback ladder via `LEARN_CV(method="deeptica", lag=5, lag_fallback=[5,4,3,2,1], ...)`.
- If no pairs are found at the requested lag, the transform builder tries smaller lags in order until pairs are available or the list is exhausted.
- Artifacts record `lag_fallback`, `attempts`, and the `lag_used` that succeeded.

Cross‑Shard Pairing (optional)
- Enable with `LEARN_CV(method="deeptica", lag=…, cross_shard_pairing=True)`.
- When shards are contiguous slices from the same trajectory (same `source.traj` and identical temperature), the transform pipeline merges adjacent shards for pair generation so that time‑lagged pairs can cross shard boundaries.
- Replicas or different thermodynamic states are never merged (different temperatures).

Interpreting the App UI
- Original CVs tab: Always shows baseline MSM/FES from the original CVs.
- Deep‑TICA CVs tab: Only shows learned‑CV MSM/FES if `applied=True` and `reason=ok`.
  - If skipped: a clear message shows the reason; no plots are shown.
  - If a plot cannot be built, the UI shows "picture failed to create itself".

Logging
- During Deep‑TICA preparation, the transform builder prints per‑shard diagnostics:
  - Summary: `n_shards=... frames_total=... lag_requested=... pairs_total=...`
  - Per‑shard: `shard <id> frames=<n> pairs=<m>`

Notes
- Public APIs remain stable; instrumentation is attached as build artifacts.
- Artifacts are serialized within the bundle JSON (`BuildResult.to_json`).
