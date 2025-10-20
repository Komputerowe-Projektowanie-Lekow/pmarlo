# pmarlo Performance Benchmarking Suite

This document describes the comprehensive performance benchmarking suite for pmarlo, designed to track and compare performance across code changes.

## Table of Contents
- [Overview](#overview)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Running Benchmarks](#running-benchmarks)
- [Comparing Results](#comparing-results)
- [Benchmark Coverage](#benchmark-coverage)
- [Best Practices](#best-practices)
- [Interpreting Results](#interpreting-results)

## Overview

The pmarlo benchmarking suite uses `pytest-benchmark` to measure performance of the most critical operations in the package:

- **REMD (Replica Exchange MD)**: Core molecular dynamics simulation performance
- **Demultiplexing**: Trajectory demultiplexing and streaming operations
- **MSM Clustering**: Markov state model microstate clustering
- **DeepTICA Training**: Deep learning-based time-lagged independent component analysis
- **FES Computation**: Free energy surface calculation and smoothing
- **Sharded Data Pipeline**: Data I/O, aggregation, and validation

These benchmarks allow you to:
1. ✅ Measure baseline performance before making changes
2. ✅ Identify performance regressions or improvements
3. ✅ Compare different implementation strategies
4. ✅ Track performance trends over time

## Quick Start

### 1. Install pytest-benchmark

```bash
poetry install  # pytest-benchmark is included in test dependencies
```

Alternatively, if not using Poetry:

```bash
pip install pytest-benchmark
```

### 2. Enable Performance Tests

Performance tests are gated by an environment variable to avoid running them accidentally:

**Windows (PowerShell):**
```powershell
$env:PMARLO_RUN_PERF=1
```

**Windows (CMD):**
```cmd
set PMARLO_RUN_PERF=1
```

**Linux/Mac:**
```bash
export PMARLO_RUN_PERF=1
```

### 3. Run Benchmarks

Run all benchmarks:
```bash
poetry run pytest -m benchmark
```

Run specific benchmark file:
```bash
poetry run pytest -m benchmark tests/perf/test_remd_performance.py
```

### 4. Save Baseline

Before making changes, save a performance baseline:
```bash
poetry run pytest -m benchmark --benchmark-save=baseline
```

### 5. Make Your Changes

Implement your code changes, algorithmic improvements, or optimizations.

### 6. Compare Performance

After changes, compare against the baseline:
```bash
poetry run pytest -m benchmark --benchmark-compare=baseline
```

This will show you exactly which tests got faster or slower, and by how much.

## Installation

### Using Poetry (Recommended)

The benchmark dependencies are already included in the project:

```bash
cd pmarlo
poetry install
```

### Using pip

If you're not using Poetry:

```bash
pip install pytest-benchmark
```

## Running Benchmarks

### Run All Benchmarks

```bash
# Set environment variable
export PMARLO_RUN_PERF=1  # Linux/Mac
# OR
$env:PMARLO_RUN_PERF=1    # Windows PowerShell

# Run benchmarks
poetry run pytest -m benchmark
```

### Run Specific Subsystem

Run only REMD benchmarks:
```bash
poetry run pytest -m benchmark tests/perf/test_remd_performance.py
```

Run only MSM clustering benchmarks:
```bash
poetry run pytest -m benchmark tests/perf/test_msm_clustering_perf.py
```

Run only DeepTICA benchmarks:
```bash
poetry run pytest -m benchmark tests/perf/test_deeptica_training_perf.py
```

### Run Specific Test

```bash
poetry run pytest -m benchmark tests/perf/test_remd_performance.py::test_md_throughput_cpu
```

### Benchmark Options

Control number of iterations:
```bash
pytest -m benchmark --benchmark-min-rounds=5
```

Disable garbage collection during benchmarks (more stable results):
```bash
pytest -m benchmark --benchmark-disable-gc
```

Save results with custom name:
```bash
pytest -m benchmark --benchmark-save=my_experiment
```

Generate histogram:
```bash
pytest -m benchmark --benchmark-histogram
```

## Comparing Results

### Basic Comparison

1. **Save baseline** before changes:
   ```bash
   poetry run pytest -m benchmark --benchmark-save=before_optimization
   ```

2. **Make your changes** to the code

3. **Compare** against baseline:
   ```bash
   poetry run pytest -m benchmark --benchmark-compare=before_optimization
   ```

### Example Output

```
----------------------- benchmark comparison: 0001_before vs 0002_after -----------------------
Name (time in ms)                          Before      After       Change
-------------------------------------------------------------------------------------------------
test_md_throughput_cpu                    245.32     198.45     -19.1% (faster)
test_exchange_overhead                    892.15     901.23      +1.0% (slower, not significant)
test_system_creation_no_bias              123.45     120.12      -2.7% (faster)
-------------------------------------------------------------------------------------------------
```

### Interpreting Comparison

- **Negative change (-)**: Performance improved (faster) ✅
- **Positive change (+)**: Performance degraded (slower) ⚠️
- **"not significant"**: Difference is within noise margin

### Advanced Comparison

Compare specific saved results:
```bash
pytest -m benchmark --benchmark-compare=0001_baseline --benchmark-compare-fail=mean:5%
```

This will fail if any benchmark is more than 5% slower than baseline.

## Benchmark Coverage

### 1. REMD Performance (`test_remd_performance.py`)

Measures the most critical REMD operations that were optimized in the performance fix:

- `test_system_creation_no_bias`: System creation time (~123ms baseline)
- `test_md_throughput_cpu`: Raw MD step throughput (~245ms for 500 steps)
- `test_replica_setup_time`: Replica initialization overhead
- `test_exchange_overhead`: Exchange attempt performance
- `test_short_remd_full_workflow`: Complete short REMD workflow
- `test_platform_selection_with_seed`: Regression test for platform selection bug

**Why it matters:** REMD performance was previously degraded by 6x due to incorrect platform selection. These benchmarks ensure the fix stays in place.

### 2. Demultiplexing (`test_demux_perf.py`)

Measures trajectory demultiplexing performance:

- `test_perf_demux_facade`: High-level demux API performance
- `test_perf_streaming_demux`: Streaming demux engine (memory efficient)

**Why it matters:** Demultiplexing can process thousands of trajectory frames and needs to be memory-efficient.

### 3. MSM Clustering (`test_msm_clustering_perf.py`)

Measures clustering performance across different dataset sizes:

- `test_kmeans_small_dataset`: KMeans on 1K samples (baseline)
- `test_kmeans_medium_dataset`: KMeans on 10K samples
- `test_minibatch_kmeans_large_dataset`: MiniBatchKMeans on 50K samples
- `test_auto_clustering_small`: Automatic state selection (silhouette scoring)
- `test_high_dimensional_clustering`: High-dimensional data (DeepTICA output)
- `test_many_states_clustering`: Clustering with 50 states (stress test)
- `test_silhouette_scoring_overhead`: Silhouette score computation overhead

**Why it matters:** Clustering is O(n*k) and can dominate MSM construction time on large datasets.

### 4. DeepTICA Training (`test_deeptica_training_perf.py`)

Measures deep learning training performance:

- `test_feature_preparation`: Feature scaling and normalization
- `test_network_construction`: Network initialization
- `test_pair_building`: Time-lagged pair construction
- `test_dataset_creation`: PyTorch dataset creation
- `test_vamp2_loss_computation`: VAMP-2 loss computation
- `test_forward_pass`: Network forward pass (inference)
- `test_backward_pass`: Network backward pass (training)
- `test_full_training_epoch_small`: Complete training epoch
- `test_output_whitening`: Output whitening application

**Why it matters:** DeepTICA training can take minutes to hours on large datasets; optimization is critical.

### 5. FES Computation (`test_fes_computation_perf.py`)

Measures free energy surface calculation performance:

- `test_2d_histogram_unweighted`: Basic 2D histogram
- `test_2d_histogram_weighted`: Weighted histogram (MSM reweighting)
- `test_gaussian_smoothing_small`: Smoothing on 50x50 grid
- `test_gaussian_smoothing_large`: Smoothing on 200x200 grid
- `test_probability_to_free_energy`: Probability to free energy conversion
- `test_full_fes_pipeline_small`: Complete FES computation (1K frames)
- `test_full_fes_pipeline_medium`: Complete FES computation (10K frames)
- `test_full_fes_pipeline_large`: Complete FES computation (100K frames)
- `test_periodic_boundary_handling`: Periodic boundaries (phi/psi angles)
- `test_high_resolution_fes`: High resolution (200x200 bins)

**Why it matters:** FES computation is called frequently during analysis and needs to be fast for interactive use.

### 6. Sharded Data Pipeline (`test_shards_aggregation_perf.py`)

Measures data I/O and aggregation performance:

- `test_single_shard_write`: Shard writing performance
- `test_single_shard_read`: Shard reading performance
- `test_multiple_shards_read`: Sequential reading
- `test_shard_validation`: Metadata validation
- `test_data_concatenation_small`: Data concatenation (3 shards)
- `test_data_concatenation_large`: Data concatenation (10 shards)
- `test_shard_aggregation_with_hashing`: Aggregation with determinism check
- `test_memory_efficient_aggregation`: Streaming aggregation
- `test_large_single_shard_io`: Large shard I/O (stress test)

**Why it matters:** Efficient data I/O is critical for large-scale workflows processing many trajectories.

## Best Practices

### 1. Always Use Environment Variable

Performance tests are disabled by default. Always set `PMARLO_RUN_PERF=1`:

```bash
export PMARLO_RUN_PERF=1  # Add to your shell profile for convenience
```

### 2. Save Baseline Before Changes

```bash
poetry run pytest -m benchmark --benchmark-save=baseline
```

### 3. Run Benchmarks Multiple Times

For more stable results, increase the number of rounds:

```bash
poetry run pytest -m benchmark --benchmark-min-rounds=10
```

### 4. Minimize System Load

- Close unnecessary applications
- Disable background services when possible
- Don't run benchmarks while other intensive tasks are running

### 5. Use Specific Benchmarks

Don't run all benchmarks if you only changed specific code:

```bash
# Only benchmark REMD if you changed REMD code
poetry run pytest -m benchmark tests/perf/test_remd_performance.py
```

### 6. Check for Statistical Significance

Small differences (< 5%) might be noise. Look for:
- Consistent trends across multiple runs
- Large changes (> 10%)
- Changes in multiple related tests

### 7. Benchmark on Target Hardware

If optimizing for specific hardware (e.g., GPU), run benchmarks on that hardware:

```bash
# REMD with CUDA
poetry run pytest -m benchmark tests/perf/test_remd_performance.py -k cuda
```

## Interpreting Results

### Understanding Benchmark Output

```
tests/perf/test_remd_performance.py::test_md_throughput_cpu
Mean: 245.32 ms
StdDev: 12.45 ms
Min: 231.12 ms
Max: 267.89 ms
```

- **Mean**: Average time across all iterations
- **StdDev**: Standard deviation (lower is more consistent)
- **Min/Max**: Range of measurements

### What to Look For

✅ **Good performance improvement:**
- Mean decreases by > 10%
- StdDev remains low
- Change is consistent across related tests

⚠️ **Potential issue:**
- Mean increases by > 5%
- StdDev increases significantly (less stable)
- Only some tests improve while related tests degrade

### Example Analysis

**Scenario 1: Algorithm Optimization**
```
test_kmeans_medium_dataset
Before: 456.78 ms ± 23.45 ms
After:  312.34 ms ± 18.92 ms
Change: -31.6% (faster) ✅
```
**Analysis:** Clear win! Algorithm is faster and more stable (lower StdDev).

**Scenario 2: Trade-off**
```
test_system_creation_no_bias
Before: 123.45 ms ± 5.67 ms
After:  145.23 ms ± 4.89 ms
Change: +17.6% (slower) ⚠️

test_md_throughput_cpu
Before: 245.32 ms ± 12.45 ms
After:  198.12 ms ± 8.34 ms
Change: -19.2% (faster) ✅
```
**Analysis:** Setup is slower but runtime is faster. Overall win if MD steps dominate total time.

## Continuous Integration

### Running in CI/CD

Add to your CI pipeline:

```yaml
# GitHub Actions example
- name: Run Performance Benchmarks
  env:
    PMARLO_RUN_PERF: 1
  run: |
    poetry run pytest -m benchmark --benchmark-save=ci_${{ github.sha }}
```

### Regression Detection

Fail build on performance regression:

```bash
poetry run pytest -m benchmark \
  --benchmark-compare=baseline \
  --benchmark-compare-fail=mean:10%
```

This fails if any benchmark is > 10% slower than baseline.

## Troubleshooting

### Benchmarks Not Running

**Problem:** Tests are skipped with "perf tests disabled"

**Solution:** Set environment variable:
```bash
export PMARLO_RUN_PERF=1
```

### Import Errors

**Problem:** `pytest_benchmark` not found

**Solution:** Install benchmark dependencies:
```bash
poetry install
# OR
pip install pytest-benchmark
```

### High Variance in Results

**Problem:** StdDev is high, results inconsistent

**Solution:**
1. Close background applications
2. Increase rounds: `--benchmark-min-rounds=20`
3. Disable GC: `--benchmark-disable-gc`

### OpenMM Not Found (REMD tests)

**Problem:** OpenMM import errors

**Solution:** Install OpenMM:
```bash
poetry install  # Should include OpenMM
# OR
conda install -c conda-forge openmm
```

### PyTorch Not Found (DeepTICA tests)

**Problem:** PyTorch import errors

**Solution:** Install PyTorch:
```bash
poetry install --extras mlcv
```

## Advanced Usage

### Comparing Multiple Baselines

```bash
# Compare against multiple saved benchmarks
pytest -m benchmark \
  --benchmark-compare=0001_baseline \
  --benchmark-compare=0002_optimization_v1
```

### Exporting Results

```bash
# Export to JSON
pytest -m benchmark --benchmark-json=results.json

# Export to CSV
pytest -m benchmark --benchmark-json=results.json
python -m pytest_benchmark compare results.json --csv=results.csv
```

### Profiling Individual Tests

For deeper analysis of a specific test:

```bash
python -m cProfile -o profile.stats tests/perf/test_remd_performance.py::test_md_throughput_cpu
python -m pstats profile.stats
```

## Summary

The pmarlo benchmark suite provides comprehensive performance testing across all critical subsystems. By following the workflow:

1. **Baseline** → 2. **Change** → 3. **Compare**

You can confidently optimize performance while preventing regressions.

**Key Commands:**
```bash
# Enable performance tests
export PMARLO_RUN_PERF=1

# Save baseline
poetry run pytest -m benchmark --benchmark-save=baseline

# Make your changes...

# Compare
poetry run pytest -m benchmark --benchmark-compare=baseline
```

For questions or issues, see the [main README](README.md) or open an issue on GitHub.

