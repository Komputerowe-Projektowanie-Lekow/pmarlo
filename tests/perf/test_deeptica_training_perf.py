from __future__ import annotations

"""Performance benchmarks for DeepTICA training pipeline.

These benchmarks measure training performance which is critical for:
- Feature preparation and scaling
- Network forward/backward passes
- VAMP-2 loss computation
- Curriculum training overhead

Run with: pytest -m benchmark tests/perf/test_deeptica_training_perf.py
"""

import os
from dataclasses import dataclass

import numpy as np
import pytest

pytestmark = [pytest.mark.perf, pytest.mark.benchmark, pytest.mark.features]

# Optional dependencies
pytest.importorskip("pytest_benchmark", reason="pytest-benchmark not installed")
pytest.importorskip("torch", reason="PyTorch required for DeepTICA benchmarks")

if not os.getenv("PMARLO_RUN_PERF"):
    pytest.skip(
        "perf tests disabled; set PMARLO_RUN_PERF=1 to run", allow_module_level=True
    )


@dataclass
class MockConfig:
    """Mock configuration for DeepTICA training."""

    seed: int = 42
    output_dim: int = 2
    hidden_units: tuple = (64, 32)
    activation: str = "elu"
    dropout: float = 0.0
    batch_norm: bool = False
    residual: bool = False
    device: str = "cpu"


def _generate_synthetic_trajectories(
    n_trajs: int, n_frames: int, n_features: int, seed: int = 42
) -> list[np.ndarray]:
    """Generate synthetic trajectory data for benchmarking."""
    rng = np.random.default_rng(seed)
    trajectories = []

    for i in range(n_trajs):
        # Generate trajectory with some temporal correlation
        traj = np.zeros((n_frames, n_features), dtype=np.float32)
        traj[0] = rng.standard_normal(n_features)

        for t in range(1, n_frames):
            # Add temporal correlation
            traj[t] = 0.9 * traj[t - 1] + 0.1 * rng.standard_normal(n_features)

        trajectories.append(traj)

    return trajectories


@pytest.fixture
def small_trajs():
    """Small trajectories (3 trajs, 200 frames, 10 features)."""
    return _generate_synthetic_trajectories(3, 200, 10)


@pytest.fixture
def medium_trajs():
    """Medium trajectories (5 trajs, 1000 frames, 20 features)."""
    return _generate_synthetic_trajectories(5, 1000, 20)


@pytest.fixture
def large_trajs():
    """Large trajectories (10 trajs, 2000 frames, 50 features)."""
    return _generate_synthetic_trajectories(10, 2000, 50)


def test_feature_preparation(benchmark, medium_trajs):
    """Benchmark feature preparation (scaling and normalization)."""
    from pmarlo.features.deeptica.core.inputs import prepare_features

    def _prepare():
        return prepare_features(
            medium_trajs, tau_schedule=(1, 5, 10), seed=42
        )

    prep = benchmark(_prepare)
    assert prep.scaler is not None
    assert prep.X_scaled is not None


def test_network_construction(benchmark, medium_trajs):
    """Benchmark network construction (initialization overhead)."""
    from pmarlo.features.deeptica.core.inputs import prepare_features
    from pmarlo.features.deeptica.core.model import build_network

    prep = prepare_features(medium_trajs, tau_schedule=(1, 5, 10), seed=42)
    cfg = MockConfig()

    def _build_network():
        return build_network(cfg, prep.scaler, seed=42)

    network = benchmark(_build_network)
    assert network is not None


def test_pair_building(benchmark, medium_trajs):
    """Benchmark time-lagged pair construction."""
    from pmarlo.features.deeptica.core.pairs import build_pair_info

    tau_schedule = (1, 5, 10)

    def _build_pairs():
        return build_pair_info(
            medium_trajs, tau_schedule, pairs=None, weights=None
        )

    pair_info = benchmark(_build_pairs)
    assert pair_info.idx_t is not None
    assert pair_info.idx_tau is not None
    assert len(pair_info.idx_t) > 0


def test_dataset_creation(benchmark, medium_trajs):
    """Benchmark PyTorch dataset creation."""
    from pmarlo.features.deeptica.core.dataset import create_dataset
    from pmarlo.features.deeptica.core.inputs import prepare_features

    prep = prepare_features(medium_trajs, tau_schedule=(1, 5, 10), seed=42)

    # Create dummy indices
    n_pairs = min(5000, prep.X_scaled.shape[0] - 10)
    idx_t = np.arange(n_pairs, dtype=np.int64)
    idx_tau = idx_t + 10

    def _create_dataset():
        return create_dataset(
            prep.X_scaled, idx_t, idx_tau, lengths=[len(t) for t in medium_trajs]
        )

    dataset = benchmark(_create_dataset)
    assert len(dataset) > 0


def test_dataloader_creation(benchmark, medium_trajs):
    """Benchmark PyTorch DataLoader creation and iteration overhead."""
    from pmarlo.features.deeptica.core.dataset import create_dataset, create_loaders
    from pmarlo.features.deeptica.core.inputs import prepare_features

    prep = prepare_features(medium_trajs, tau_schedule=(1, 5, 10), seed=42)

    n_pairs = min(5000, prep.X_scaled.shape[0] - 10)
    idx_t = np.arange(n_pairs, dtype=np.int64)
    idx_tau = idx_t + 10
    weights = np.ones(n_pairs, dtype=np.float32)

    sequences = [np.arange(len(t)) for t in medium_trajs]
    lengths = [len(t) for t in medium_trajs]

    def _create_loaders():
        return create_loaders(
            prep.X_scaled,
            idx_t,
            idx_tau,
            weights,
            sequences,
            lengths,
            batch_size=256,
            device="cpu",
        )

    loaders = benchmark(_create_loaders)
    assert loaders is not None


def test_vamp2_loss_computation(benchmark, medium_trajs):
    """Benchmark VAMP-2 loss computation (core training objective)."""
    import torch
    from pmarlo.features.deeptica.core.inputs import prepare_features
    from pmarlo.features.deeptica.core.model import build_network

    prep = prepare_features(medium_trajs, tau_schedule=(1, 5, 10), seed=42)
    cfg = MockConfig()
    network = build_network(cfg, prep.scaler, seed=42)
    network.eval()

    # Create dummy batch
    n_samples = 256
    X = torch.randn(n_samples, medium_trajs[0].shape[1])

    def _compute_loss():
        with torch.no_grad():
            output = network(X)
            # Simulate VAMP-2 loss computation
            loss = torch.mean(output**2)
        return loss

    loss = benchmark(_compute_loss)
    assert loss is not None


def test_forward_pass(benchmark, medium_trajs):
    """Benchmark network forward pass (inference speed)."""
    import torch
    from pmarlo.features.deeptica.core.inputs import prepare_features
    from pmarlo.features.deeptica.core.model import build_network

    prep = prepare_features(medium_trajs, tau_schedule=(1, 5, 10), seed=42)
    cfg = MockConfig()
    network = build_network(cfg, prep.scaler, seed=42)
    network.eval()

    X = torch.randn(256, medium_trajs[0].shape[1])

    def _forward():
        with torch.no_grad():
            return network(X)

    output = benchmark(_forward)
    assert output is not None
    assert output.shape[1] == cfg.output_dim


def test_backward_pass(benchmark, medium_trajs):
    """Benchmark network backward pass (training overhead)."""
    import torch
    from pmarlo.features.deeptica.core.inputs import prepare_features
    from pmarlo.features.deeptica.core.model import build_network

    prep = prepare_features(medium_trajs, tau_schedule=(1, 5, 10), seed=42)
    cfg = MockConfig()
    network = build_network(cfg, prep.scaler, seed=42)
    network.train()

    optimizer = torch.optim.Adam(network.parameters(), lr=0.001)

    X = torch.randn(256, medium_trajs[0].shape[1])

    def _backward():
        optimizer.zero_grad()
        output = network(X)
        loss = torch.mean(output**2)
        loss.backward()
        optimizer.step()
        return loss.item()

    loss = benchmark(_backward)
    assert loss is not None


def test_full_training_epoch_small(benchmark, small_trajs):
    """Benchmark one full training epoch on small dataset."""
    import torch
    from pmarlo.features.deeptica.core.dataset import create_loaders
    from pmarlo.features.deeptica.core.inputs import prepare_features
    from pmarlo.features.deeptica.core.model import build_network
    from pmarlo.features.deeptica.core.pairs import build_pair_info

    prep = prepare_features(small_trajs, tau_schedule=(1, 5, 10), seed=42)
    cfg = MockConfig()
    network = build_network(cfg, prep.scaler, seed=42)
    network.train()

    pair_info = build_pair_info(
        small_trajs, (1, 5, 10), pairs=None, weights=None
    )

    sequences = [np.arange(len(t)) for t in small_trajs]
    lengths = [len(t) for t in small_trajs]

    train_loader, val_loader = create_loaders(
        prep.X_scaled,
        pair_info.idx_t,
        pair_info.idx_tau,
        pair_info.weights,
        sequences,
        lengths,
        batch_size=128,
        device="cpu",
    )

    optimizer = torch.optim.Adam(network.parameters(), lr=0.001)

    def _train_epoch():
        total_loss = 0.0
        for batch in train_loader:
            X_t, X_tau = batch[:2]
            optimizer.zero_grad()
            output_t = network(X_t)
            output_tau = network(X_tau)
            loss = torch.mean((output_t - output_tau) ** 2)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_loader)

    loss = benchmark(_train_epoch)
    assert loss is not None


def test_scaling_overhead(benchmark, large_trajs):
    """Benchmark feature scaling overhead on large dataset."""
    from sklearn.preprocessing import StandardScaler

    # Concatenate all trajectories
    X_concat = np.vstack(large_trajs)

    def _scale():
        scaler = StandardScaler()
        return scaler.fit_transform(X_concat)

    X_scaled = benchmark(_scale)
    assert X_scaled.shape == X_concat.shape


def test_output_whitening(benchmark, medium_trajs):
    """Benchmark output whitening application."""
    import torch
    from pmarlo.features.deeptica.core.inputs import prepare_features
    from pmarlo.features.deeptica.core.model import (
        apply_output_whitening,
        build_network,
    )

    prep = prepare_features(medium_trajs, tau_schedule=(1, 5, 10), seed=42)
    cfg = MockConfig()
    network = build_network(cfg, prep.scaler, seed=42)

    # Get sample output
    X = torch.randn(1000, medium_trajs[0].shape[1])
    with torch.no_grad():
        output = network(X).numpy()

    def _whiten():
        return apply_output_whitening(network, output)

    whitened_net = benchmark(_whiten)
    assert whitened_net is not None

