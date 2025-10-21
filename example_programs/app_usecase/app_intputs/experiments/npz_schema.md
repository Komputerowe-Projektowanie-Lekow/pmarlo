# Canonical `.npz` schema for PMARLO shards

Each shard must provide arrays compatible with `pmarlo.shards.format.read_shard_npz_json`.

Required keys:

- `X`: Shape `(n_frames, n_features)` float32 matrix of transformed features / CVs.
- `t_index`: Shape `(n_frames,)` int64 array of absolute frame indices (monotonic).
- `dt_ps`: Scalar float32 storing the timestep spacing for the shard (seconds × 1e12).

Optional keys (recommended for experiments):

- `energy`: Shape `(n_frames,)` float32 reduced potential at the shard’s native temperature.
- `bias`: Shape `(n_frames,)` float32 bias potential applied during the run (0 for unbiased runs).
- `w_frame`: Shape `(n_frames,)` float32 precomputed weights (leave empty for reweighter to fill).

For TRAM/MBAR inputs supply per-ladder energies via additional arrays (e.g., `energy_ladder`) or encode them alongside `energy` using per-temperature columns; document the choice in the shard metadata `provenance`.

All arrays should be deterministic given the manifest seeds so integration tests can compare results across runs.
