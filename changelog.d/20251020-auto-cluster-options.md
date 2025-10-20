### Added
- Optional `silhouette_sample_size` argument for MSM microstate clustering to evaluate silhouette scores on a random subset of frames, reducing auto-selection overhead for large datasets.
- `auto_n_states_override` parameter to bypass the silhouette optimization loop while keeping `n_states="auto"` semantics, plus regression tests covering both new code paths.
