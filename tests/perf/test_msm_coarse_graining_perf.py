from __future__ import annotations

"""PCCA+ coarse-graining performance benchmarks.

These micro-benchmarks focus on the spectral lumping routine used to map
hundreds of microstates into a small macrostate partition. Each benchmark
constructs a synthetic, metastable transition matrix with known block
structure and verifies that the coarse-grained labels recover the dominant
metastable sets. The workload intentionally stays lightweight so it can be
executed frequently while still exercising the full PCCA+ path.

Run with: pytest -m benchmark tests/perf/test_msm_coarse_graining_perf.py
"""

import os

import numpy as np
import pytest

pytestmark = [pytest.mark.perf, pytest.mark.benchmark, pytest.mark.msm]

pytest.importorskip("pytest_benchmark", reason="pytest-benchmark not installed")
pytest.importorskip("deeptime", reason="deeptime is required for PCCA+ benchmarks")

if not os.getenv("PMARLO_RUN_PERF"):
    pytest.skip(
        "perf tests disabled; set PMARLO_RUN_PERF=1 to run", allow_module_level=True
    )


def _metastable_transition_matrix(
    n_blocks: int,
    block_size: int,
    intra_prob: float = 0.94,
    noise: float = 0.0,
    *,
    seed: int = 13,
) -> np.ndarray:
    """Create a block-structured transition matrix with optional noise."""
    rng = np.random.default_rng(seed)
    n_states = n_blocks * block_size
    T = np.zeros((n_states, n_states), dtype=float)

    for block in range(n_blocks):
        start = block * block_size
        stop = start + block_size
        for row in range(start, stop):
            T[row, start:stop] = intra_prob / block_size
            if noise:
                T[row, start:stop] += rng.normal(0.0, noise, size=block_size)
            remaining_prob = max(1.0 - np.sum(T[row, start:stop]), 0.0)
            if n_states - block_size:
                other = np.delete(np.arange(n_states), np.s_[start:stop])
                share = remaining_prob / other.size if other.size else 0.0
                T[row, other] = share
            T[row] = np.clip(T[row], 0.0, None)
            row_sum = T[row].sum()
            if row_sum == 0.0:
                T[row, row] = 1.0
            else:
                T[row] /= row_sum
    return T


def _assert_block_alignment(labels: np.ndarray, block_size: int, n_blocks: int) -> None:
    for block in range(n_blocks):
        start = block * block_size
        stop = start + block_size
        block_labels = labels[start:stop]
        counts = np.bincount(block_labels)
        dominant = counts.argmax()
        assert counts[dominant] >= int(0.6 * block_size)


def test_pcca_macrostate_lumping_block_structure(benchmark: pytest.BenchmarkFixture) -> None:
    """Benchmark PCCA+ coarse graining on a well-separated 4-block system."""
    from pmarlo.markov_state_model.bridge import pcca_like_macrostates

    n_blocks = 4
    block_size = 20
    T = _metastable_transition_matrix(n_blocks, block_size, intra_prob=0.96)

    def _run() -> np.ndarray | None:
        return pcca_like_macrostates(T, n_macrostates=n_blocks)

    labels = benchmark(_run)
    assert labels is not None
    assert labels.shape == (n_blocks * block_size,)
    assert np.unique(labels).size == n_blocks
    _assert_block_alignment(labels, block_size, n_blocks)


def test_pcca_scales_to_hundreds_of_microstates(benchmark: pytest.BenchmarkFixture) -> None:
    """Benchmark PCCA+ when lumping 8x80 microstates."""
    from pmarlo.markov_state_model.bridge import pcca_like_macrostates

    n_blocks = 8
    block_size = 80
    T = _metastable_transition_matrix(n_blocks, block_size, intra_prob=0.93)

    def _run() -> np.ndarray | None:
        return pcca_like_macrostates(T, n_macrostates=n_blocks)

    labels = benchmark(_run)
    assert labels is not None
    assert labels.shape == (n_blocks * block_size,)
    assert np.unique(labels).size == n_blocks
    _assert_block_alignment(labels, block_size, n_blocks)


def test_pcca_robust_against_small_noise(benchmark: pytest.BenchmarkFixture) -> None:
    """Benchmark PCCA+ stability when transition probabilities are noisy."""
    from pmarlo.markov_state_model.bridge import pcca_like_macrostates

    n_blocks = 5
    block_size = 30
    T = _metastable_transition_matrix(
        n_blocks,
        block_size,
        intra_prob=0.92,
        noise=0.02,
        seed=77,
    )

    def _run() -> np.ndarray | None:
        return pcca_like_macrostates(T, n_macrostates=n_blocks)

    labels = benchmark(_run)
    assert labels is not None
    assert labels.shape == (n_blocks * block_size,)
    assert np.unique(labels).size == n_blocks
    _assert_block_alignment(labels, block_size, n_blocks)
