### Changed

- Optimized `lump_micro_to_macro_T` in `bridge.py` to use vectorized operations instead of nested loops, significantly improving performance for MSM macro-state lumping.
- Optimized `compute_macro_populations` in `bridge.py` to use `np.bincount` instead of loop with repeated `np.where` calls.
- Optimized `_canonicalize_macro_labels` in `bridge.py` to use vectorized array indexing instead of list comprehension.
- Removed redundant array conversion in `_bayesian_transition_samples` in `_its.py`.
- Optimized `_detect_reducibility` in `msm_utils.py` to reduce per-state iteration overhead.
- Optimized `serialize_macro_mapping` in `bridge.py` to avoid explicit list comprehension during array to list conversion.
- Cached array shape queries in `finalize.py` and `_features.py` to avoid redundant attribute access.
