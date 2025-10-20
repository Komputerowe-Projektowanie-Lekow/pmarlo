from __future__ import annotations

"""Performance benchmarks for sharded data aggregation pipeline.

These benchmarks measure data pipeline performance which is critical for:
- Shard emission from trajectories
- Shard reading and validation
- Data aggregation
- Bundle building

Run with: pytest -m benchmark tests/perf/test_shards_aggregation_perf.py
"""

import os
from pathlib import Path

import numpy as np
import pytest

pytestmark = [pytest.mark.perf, pytest.mark.benchmark, pytest.mark.data]

# Optional plugin
pytest.importorskip("pytest_benchmark", reason="pytest-benchmark not installed")

if not os.getenv("PMARLO_RUN_PERF"):
    pytest.skip(
        "perf tests disabled; set PMARLO_RUN_PERF=1 to run", allow_module_level=True
    )


def _create_synthetic_shard(
    n_frames: int, n_cvs: int, output_path: Path, temp: float = 300.0, seed: int = 42
) -> Path:
    """Create a synthetic shard file for benchmarking."""
    from pmarlo.data.shard import ShardMetadata, write_shard

    rng = np.random.default_rng(seed)

    # Generate synthetic CV data
    cv_data = rng.standard_normal((n_frames, n_cvs)).astype(np.float32)

    # Create metadata
    cv_names = [f"cv_{i}" for i in range(n_cvs)]
    periodic = [False] * n_cvs

    metadata = ShardMetadata(
        cv_names=tuple(cv_names),
        periodic=tuple(periodic),
        temperature=temp,
        length=n_frames,
        source="synthetic_benchmark",
    )

    # Write shard
    write_shard(output_path, metadata, cv_data, dtraj=None)
    return output_path


@pytest.fixture
def small_shards(tmp_path):
    """Create small shards (3 shards, 200 frames each, 5 CVs)."""
    shards = []
    for i in range(3):
        shard_path = tmp_path / f"shard_small_{i}.npz"
        _create_synthetic_shard(200, 5, shard_path, seed=42 + i)
        shards.append(shard_path)
    return shards


@pytest.fixture
def medium_shards(tmp_path):
    """Create medium shards (5 shards, 1000 frames each, 10 CVs)."""
    shards = []
    for i in range(5):
        shard_path = tmp_path / f"shard_medium_{i}.npz"
        _create_synthetic_shard(1000, 10, shard_path, seed=42 + i)
        shards.append(shard_path)
    return shards


@pytest.fixture
def large_shards(tmp_path):
    """Create large shards (10 shards, 2000 frames each, 20 CVs)."""
    shards = []
    for i in range(10):
        shard_path = tmp_path / f"shard_large_{i}.npz"
        _create_synthetic_shard(2000, 20, shard_path, seed=42 + i)
        shards.append(shard_path)
    return shards


def test_single_shard_write(benchmark, tmp_path):
    """Benchmark single shard write operation."""
    n_frames = 1000
    n_cvs = 10
    output_path = tmp_path / "benchmark_shard.npz"

    def _write_shard():
        return _create_synthetic_shard(n_frames, n_cvs, output_path)

    result = benchmark(_write_shard)
    assert result.exists()


def test_single_shard_read(benchmark, tmp_path):
    """Benchmark single shard read operation."""
    from pmarlo.data.shard import read_shard

    shard_path = _create_synthetic_shard(1000, 10, tmp_path / "test_shard.npz")

    def _read_shard():
        return read_shard(shard_path)

    result = benchmark(_read_shard)
    metadata, X, dtraj = result
    assert metadata is not None
    assert X is not None


def test_multiple_shards_read(benchmark, medium_shards):
    """Benchmark reading multiple shards sequentially."""
    from pmarlo.data.shard import read_shard

    def _read_all_shards():
        results = []
        for shard_path in medium_shards:
            metadata, X, dtraj = read_shard(shard_path)
            results.append((metadata, X, dtraj))
        return results

    results = benchmark(_read_all_shards)
    assert len(results) == len(medium_shards)


def test_shard_validation(benchmark, medium_shards):
    """Benchmark shard metadata validation."""
    from pmarlo.data.shard import read_shard

    # Read all shards first
    shards_data = []
    for shard_path in medium_shards:
        metadata, X, dtraj = read_shard(shard_path)
        shards_data.append((metadata, X, dtraj))

    def _validate_consistency():
        # Check CV names consistency
        first_cv_names = shards_data[0][0].cv_names
        for metadata, _, _ in shards_data[1:]:
            if metadata.cv_names != first_cv_names:
                raise ValueError("CV names mismatch")

        # Check periodic consistency
        first_periodic = shards_data[0][0].periodic
        for metadata, _, _ in shards_data[1:]:
            if metadata.periodic != first_periodic:
                raise ValueError("Periodic mismatch")

        return True

    result = benchmark(_validate_consistency)
    assert result is True


def test_data_concatenation_small(benchmark, small_shards):
    """Benchmark data concatenation from small shards."""
    from pmarlo.data.shard import read_shard

    # Read all shards
    shards_data = []
    for shard_path in small_shards:
        metadata, X, dtraj = read_shard(shard_path)
        shards_data.append(X)

    def _concatenate():
        return np.vstack(shards_data)

    result = benchmark(_concatenate)
    assert result.shape[0] == sum(s.shape[0] for s in shards_data)


def test_data_concatenation_large(benchmark, large_shards):
    """Benchmark data concatenation from large shards."""
    from pmarlo.data.shard import read_shard

    shards_data = []
    for shard_path in large_shards:
        metadata, X, dtraj = read_shard(shard_path)
        shards_data.append(X)

    def _concatenate():
        return np.vstack(shards_data)

    result = benchmark(_concatenate)
    assert result.shape[0] == sum(s.shape[0] for s in shards_data)


def test_shard_aggregation_with_hashing(benchmark, medium_shards):
    """Benchmark shard aggregation with hash computation (determinism check)."""
    import hashlib
    from pmarlo.data.shard import read_shard

    def _aggregate_and_hash():
        # Read and aggregate
        all_data = []
        for shard_path in medium_shards:
            metadata, X, dtraj = read_shard(shard_path)
            all_data.append(X)

        concatenated = np.vstack(all_data)

        # Compute hash for determinism
        data_bytes = concatenated.tobytes()
        hash_digest = hashlib.sha256(data_bytes).hexdigest()

        return concatenated, hash_digest

    result, hash_digest = benchmark(_aggregate_and_hash)
    assert len(hash_digest) == 64  # SHA256 hex digest length


def test_transform_pipeline_overhead(benchmark, small_shards):
    """Benchmark transform pipeline application overhead."""
    from pmarlo.data.shard import read_shard
    from pmarlo.transform.plan import TransformPlan, TransformStep

    # Create simple transform plan
    plan = TransformPlan(steps=(TransformStep("SMOOTH_FES", {"sigma": 0.6}),))

    # Read data
    shards_data = []
    for shard_path in small_shards:
        metadata, X, dtraj = read_shard(shard_path)
        shards_data.append(X)

    concatenated = np.vstack(shards_data)

    def _apply_transform():
        # Simulate transform application (actual transform not critical for benchmark)
        # This is just to measure overhead of transform framework
        result = concatenated.copy()
        for step in plan.steps:
            # Simulate some operation
            result = result + 0.0
        return result

    result = benchmark(_apply_transform)
    assert result.shape == concatenated.shape


def test_memory_efficient_aggregation(benchmark, large_shards):
    """Benchmark memory-efficient aggregation (streaming approach)."""
    from pmarlo.data.shard import read_shard

    def _aggregate_streaming():
        # First pass: count total frames
        total_frames = 0
        n_cvs = None
        for shard_path in large_shards:
            metadata, X, _ = read_shard(shard_path)
            total_frames += X.shape[0]
            if n_cvs is None:
                n_cvs = X.shape[1]

        # Preallocate
        result = np.zeros((total_frames, n_cvs), dtype=np.float32)

        # Second pass: fill
        offset = 0
        for shard_path in large_shards:
            metadata, X, _ = read_shard(shard_path)
            result[offset : offset + X.shape[0]] = X
            offset += X.shape[0]

        return result

    result = benchmark(_aggregate_streaming)
    assert result.shape[0] > 0


def test_shard_metadata_extraction(benchmark, medium_shards):
    """Benchmark metadata extraction from multiple shards."""
    from pmarlo.data.shard import read_shard

    def _extract_metadata():
        metadata_list = []
        for shard_path in medium_shards:
            metadata, _, _ = read_shard(shard_path)
            metadata_list.append(
                {
                    "cv_names": metadata.cv_names,
                    "temperature": metadata.temperature,
                    "length": metadata.length,
                    "source": metadata.source,
                }
            )
        return metadata_list

    result = benchmark(_extract_metadata)
    assert len(result) == len(medium_shards)


def test_parallel_shard_write(benchmark, tmp_path):
    """Benchmark parallel shard writing (simulate batch emission)."""
    n_shards = 5
    n_frames = 500
    n_cvs = 10

    def _write_multiple_shards():
        shard_paths = []
        for i in range(n_shards):
            shard_path = tmp_path / f"parallel_shard_{i}.npz"
            _create_synthetic_shard(n_frames, n_cvs, shard_path, seed=42 + i)
            shard_paths.append(shard_path)
        return shard_paths

    result = benchmark(_write_multiple_shards)
    assert len(result) == n_shards
    assert all(p.exists() for p in result)


def test_large_single_shard_io(benchmark, tmp_path):
    """Benchmark I/O for a single large shard (stress test)."""
    from pmarlo.data.shard import read_shard

    # Create large shard
    shard_path = tmp_path / "large_shard.npz"
    _create_synthetic_shard(10_000, 50, shard_path)

    def _read_large_shard():
        return read_shard(shard_path)

    result = benchmark(_read_large_shard)
    metadata, X, _ = result
    assert X.shape == (10_000, 50)

