## fixed

- Fixed `tests/unit/app_usecase/test_backend_analysis_pairs.py::test_analysis_total_pairs_matches_summary` by importing `BuildConfig` from `pmarlo_webapp.app.backend` and ensuring the test monkeypatches `pmarlo_webapp.app.backend.analysis.load_shards_as_dataset` / `build_from_shards`, keeping the fake dataset/build logic active instead of triggering actual shard reads.
