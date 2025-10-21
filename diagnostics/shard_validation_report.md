# Shard Validation Summary
This report summarises metadata for the bundled Streamlit example shards.
## same_temp_shards
| shard | n_frames | range_start | range_stop | first_index | last_index | stride | contiguous_length |
|---|---|---|---|---|---|---|---|
| T300K_seg0000_rep000 | 1000 | 0 | 1000 | 0 | 999 | 1 | 999 |
| T300K_seg0001_rep000 | 1000 | 0 | 1000 | 0 | 999 | 1 | 999 |
| T300K_seg0002_rep000 | 1000 | 0 | 1000 | 0 | 999 | 1 | 999 |
| T300K_seg0003_rep000 | 1000 | 0 | 1000 | 0 | 999 | 1 | 999 |
| T300K_seg0004_rep000 | 1000 | 0 | 1000 | 0 | 999 | 1 | 999 |
*Detected stride*: 1; *Required separation* (lag/stride) = 10
*Minimum frames in set*: 1000

## disjoint_ladders_shards
| shard | n_frames | range_start | range_stop | first_index | last_index | stride | contiguous_length |
|---|---|---|---|---|---|---|---|
| T268K_seg0000_rep000 | 1000 | 0 | 1000 | 0 | 999 | 1 | 999 |
| T268K_seg0001_rep000 | 1000 | 0 | 1000 | 0 | 999 | 1 | 999 |
| T284K_seg0000_rep000 | 1000 | 0 | 1000 | 0 | 999 | 1 | 999 |
| T320K_seg0000_rep000 | 1000 | 0 | 1000 | 0 | 999 | 1 | 999 |
| T336K_seg0000_rep000 | 1000 | 0 | 1000 | 0 | 999 | 1 | 999 |
*Detected stride*: 1; *Required separation* (lag/stride) = 10
*Minimum frames in set*: 1000

## Analysis parameters
- Default lag from `BuildConfig`: 10 frames.

## Conclusions
- All shards contain 1000 frames with contiguous indices 0–999 and stride 1, yielding 999-frame segments.
- With lag 10 and stride 1, 10-frame separations are required; each shard has far more frames, so (t, t+lag) pairs exist within single shards.
- Pair construction (`_build_uniform_pairs`) only links frames within a shard, so cross-shard timestamps are not used to bridge gaps; per-shard coverage suffices.
