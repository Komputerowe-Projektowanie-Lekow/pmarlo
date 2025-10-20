### Added
- TorchScript-based feature extraction module (`src/pmarlo/features/deeptica/ts_feature_extractor.py`) that computes molecular features (distances, angles, dihedrals) directly from atomic positions and box vectors at MD step time, eliminating Python callbacks and enabling CPU-viable CV biasing.
- Comprehensive validation infrastructure in `system_builder.py` and `export.py`: hash-locked feature specifications, strict dtype checking (float32 enforcement), dimension validation, scaler sanity checks, and required method verification. All validations raise hard errors with no fallbacks.
- Configuration module (`src/pmarlo/settings/`) with explicit required keys (`enable_cv_bias`, `bias_mode`, `torch_threads`, `precision`) and strict validation; missing or invalid keys cause immediate `ConfigurationError`.
- Periodic CV and bias energy logging every 1000 steps during replica exchange simulations, reporting mean and standard deviation of collective variables and bias energy.
- Benchmark harness (`scripts/bench_openmm.py`) to measure OpenMM performance with and without CV biasing, reporting steps/second, bias energy statistics, and CV statistics for performance assessment on CPU and GPU platforms.
- Test suite for TorchScript feature extraction: parity tests (eager vs scripted), finite-difference force validation, and PBC wrap invariance tests in `tests/force/`.
- Feature specification validation with SHA-256 hashing to prevent model/spec mismatches at runtime.
- Explicit unit coverage ensuring the Streamlit backend hits the real REMD runner by default and only engages the synthetic stub when tests request it.

### Fixed
- **CRITICAL: Platform selection bug causing 6x slowdown** - `platform_selector.py` was incorrectly selecting Reference platform (10-100x slower than CPU) whenever `random_seed` was set. Fixed to auto-select fastest available platform (CUDA > CPU > Reference) while maintaining determinism via platform-specific flags. This restores REMD performance from ~20 minutes for 5K steps to ~3 minutes.
- Restore the replica-exchange `Simulation` helper to accept modern configuration options and create its output directory, allowing deterministic integration tests to pass again.
- Initialize `_bias_log_interval` attribute in `ReplicaExchange.__init__()` to prevent `AttributeError` during CV monitoring setup.
- Optimize default exchange frequency from 50 to 100 steps based on benchmark showing 100 steps provides best balance of exchange statistics and throughput.
- Streamlit "quick preset" no longer routes through the synthetic sampling stub; REMD now executes normally unless `SimulationConfig.stub_result` is explicitly set, fixing instant-complete runs in the demo app.

### Changed
- Surface pipeline configuration, stage timings, runtime summaries, and failure notifications directly in the console for easier headless monitoring (`src/pmarlo/transform/pipeline.py`, `src/pmarlo/utils/logging_utils.py`).
- Add timing and quality diagnostics to replica exchange runs, including elapsed durations for equilibration/production and console warnings when acceptance or diffusion fall outside recommended ranges (`src/pmarlo/replica_exchange/replica_exchange.py`).
- Report demultiplexing runtime, output metadata, and duration summaries at completion so downstream workflows can validate performance (`src/pmarlo/demultiplexing/demux.py`).
- Update `CV_INTEGRATION_GUIDE.md` and `CV_REQUIREMENTS.md` to reflect TorchScript implementation, remove "feature extraction not implemented" warnings, and correct physics terminology: the bias is a **harmonic restraint in CV space** (E = k·Σ(cv²)), not an "exploration" bias. Documentation now includes CPU performance benchmarks, configuration requirements, and explicit usage examples.
- Export workflow in `features/deeptica/export.py` now produces a single TorchScript module with embedded feature extraction, scaler, CV model, and bias potential, optimized via `torch.jit.optimize_for_inference()`.
- `system_builder.py` loads TorchScript CV bias models with comprehensive validation, sets PyTorch thread count from configuration, and enforces single-precision computation for CPU performance.

### Removed
- Silent fallback behaviors in CV bias loading and configuration; all errors now terminate early with clear exceptions.
