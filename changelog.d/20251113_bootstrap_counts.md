## fixed

- Ensured `tests/unit/markov_state_model/test_bootstrap_counts.py` now supplies an `output_dir` to `EnhancedMSM` so its bootstrap-counts self-check can run without `MSMBase` raising a `TypeError`.
