### Added
- Comprehensive `pytest-benchmark` performance benchmarking suite covering critical operations:
  - REMD (Replica Exchange MD): system creation, MD throughput, exchange overhead, platform selection
  - Demultiplexing: facade and streaming engine performance
  - MSM Clustering: KMeans vs MiniBatchKMeans, automatic state selection, silhouette scoring
  - DeepTICA Training: feature preparation, network operations, VAMP-2 loss computation, full epoch training
  - FES Computation: weighted histograms, Gaussian smoothing, probability-to-free-energy conversion
  - Sharded Data Pipeline: shard I/O, aggregation, validation, memory-efficient streaming
- Added `@pytest.mark.benchmark` marker for targeted performance testing
- Benchmark tests can be run with `pytest -m benchmark` for focused performance comparisons
- All benchmark tests respect `PMARLO_RUN_PERF` environment variable for CI/CD control
- Added `pytest-benchmark ^4.0` to test dependencies in pyproject.toml

### Changed
- Enhanced existing `tests/perf/test_demux_perf.py` with benchmark marker
- Extended pytest markers configuration to include dedicated `benchmark` marker
- Added `tests/perf` to pytest discovery paths so performance benchmarks are actually collected when requested
- Reworked performance suites to favor lightweight algorithm-focused micro-benchmarks, drastically reducing data volumes for DeepTICA, MSM clustering, shard aggregation, and REMD diagnostics workloads

### Documentation
- Each benchmark file includes detailed docstrings explaining what is being measured
- Benchmark tests are organized by subsystem for easy navigation and selective execution

### Fixed
- Streaming demux benchmark reinitializes its trajectory writer for every run, preventing \"Writer is not open\" errors during repeated iterations
- Shard aggregation performance helper now emits canonical NPZ/JSON shards via `pmarlo.data.shard.write_shard`, restoring compatibility with the current shard metadata API
- DeepTICA, MSM, and REMD performance benchmarks updated for the current dataset/loader, clustering, and exchange statistics APIs, eliminating AttributeError and outdated assertion failures

