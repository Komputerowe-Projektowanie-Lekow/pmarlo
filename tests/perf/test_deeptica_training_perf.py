from __future__ import annotations

"""Performance benchmarks for DeepTICA training pipeline.

These benchmarks measure training performance which is critical for:
- Feature preparation and scaling
- Network forward/backward passes
- VAMP-2 loss computation
- Curriculum training overhead

Run with: pytest -m benchmark tests/perf/test_deeptica_training_perf.py
"""

import csv
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
    batch_size: int = 128
    val_frac: float = 0.1
    num_workers: int = 0


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
    """Small trajectories (3 trajs, 128 frames, 12 features)."""
    return _generate_synthetic_trajectories(3, 128, 12)


@pytest.fixture
def medium_trajs():
    """Medium trajectories (4 trajs, 512 frames, 24 features)."""
    return _generate_synthetic_trajectories(4, 512, 24)


@pytest.fixture
def large_trajs():
    """Large trajectories (6 trajs, 1024 frames, 32 features)."""
    return _generate_synthetic_trajectories(6, 1024, 32)


def test_feature_preparation(benchmark, medium_trajs):
    """Benchmark feature preparation (scaling and normalization)."""
    from pmarlo.features.deeptica.core.inputs import prepare_features

    def _prepare():
        return prepare_features(medium_trajs, tau_schedule=(1, 5, 10), seed=42)

    prep = benchmark(_prepare)
    assert prep.scaler is not None
    assert prep.Z is not None


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
        return build_pair_info(medium_trajs, tau_schedule, pairs=None, weights=None)

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
    n_pairs = min(5000, prep.Z.shape[0] - 10)
    idx_t = np.arange(n_pairs, dtype=np.int64)
    idx_tau = idx_t + 10
    weights = np.ones(n_pairs, dtype=np.float32)

    def _create_dataset():
        return create_dataset(prep.Z, idx_t, idx_tau, weights)

    dataset = benchmark(_create_dataset)
    assert len(dataset) > 0


def test_dataloader_creation(benchmark, medium_trajs):
    """Benchmark PyTorch DataLoader creation and iteration overhead."""
    from pmarlo.features.deeptica.core.dataset import create_dataset, create_loaders
    from pmarlo.features.deeptica.core.inputs import prepare_features

    prep = prepare_features(medium_trajs, tau_schedule=(1, 5, 10), seed=42)

    n_pairs = min(5000, prep.Z.shape[0] - 10)
    idx_t = np.arange(n_pairs, dtype=np.int64)
    idx_tau = idx_t + 10
    weights = np.ones(n_pairs, dtype=np.float32)

    dataset = create_dataset(prep.Z, idx_t, idx_tau, weights)
    cfg = MockConfig(batch_size=256, val_frac=0.2, num_workers=0)

    def _create_loaders():
        return create_loaders(dataset, cfg)

    bundle = benchmark(_create_loaders)
    assert bundle.train_loader is not None
    assert bundle.val_loader is not None


def test_dataloader_batch_iteration(benchmark, medium_trajs):
    """Benchmark throughput of the DeepTICA training DataLoader."""

    from pmarlo.features.deeptica.core.dataset import create_dataset, create_loaders
    from pmarlo.features.deeptica.core.inputs import prepare_features

    prep = prepare_features(medium_trajs, tau_schedule=(1, 5, 10), seed=1234)

    n_pairs = min(8000, prep.Z.shape[0] - 5)
    idx_t = np.arange(n_pairs, dtype=np.int64)
    idx_tau = idx_t + 5
    weights = np.full(n_pairs, 1.0, dtype=np.float32)

    dataset = create_dataset(prep.Z, idx_t, idx_tau, weights)
    cfg = MockConfig(batch_size=192, val_frac=0.15, num_workers=0)
    bundle = create_loaders(dataset, cfg)

    assert bundle.train_loader is not None

    def _iterate_loader() -> int:
        total = 0
        for batch in bundle.train_loader:
            data = batch["data"]
            data_lag = batch["data_lag"]
            total += int(data.shape[0])
            # Ensure paired batches remain aligned
            _ = data_lag.shape
        return total

    seen = benchmark(_iterate_loader)
    assert seen >= cfg.batch_size


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


def test_whitening_wrapper_forward(benchmark):
    """Benchmark the whitening layer applied after network outputs."""

    import torch

    from pmarlo.features.deeptica.core.model import WhitenWrapper

    torch.manual_seed(42)

    base = torch.nn.Linear(24, 12)
    mean = np.linspace(-0.5, 0.5, 12, dtype=np.float32)
    transform = np.eye(12, dtype=np.float32)

    wrapper = WhitenWrapper(base, mean, transform)
    wrapper.eval()

    batch = torch.randn(512, 24)

    def _forward_pass():
        with torch.no_grad():
            return wrapper(batch)

    output = benchmark(_forward_pass)
    assert output.shape == (512, 12)


def test_backward_pass_and_optimizer_step(benchmark):
    """Benchmark DeepTICA backward pass and optimizer update."""

    import numpy as np
    import torch
    from torch import nn

    from pmarlo.features.deeptica.losses import VAMP2Loss
    from pmarlo.features.deeptica_trainer import loops

    rng = np.random.default_rng(1234)
    feature_dim = 6
    batch_frames = 128
    lag = 3

    x_t_full = rng.normal(size=(batch_frames, feature_dim)).astype(np.float32)
    noise = rng.normal(scale=0.1, size=(batch_frames, feature_dim)).astype(np.float32)
    x_tau_full = np.roll(x_t_full, -lag, axis=0) + noise

    weights = np.full(batch_frames, 1.0 / float(batch_frames), dtype=np.float32)
    batch = [
        (
            x_t_full[: batch_frames // 2],
            x_tau_full[: batch_frames // 2],
            weights[: batch_frames // 2],
        ),
        (
            x_t_full[batch_frames // 2 :],
            x_tau_full[batch_frames // 2 :],
            weights[batch_frames // 2 :],
        ),
    ]

    x_t, x_tau, w = loops.prepare_batch(batch, use_weights=True)

    model = nn.Sequential(
        nn.Linear(feature_dim, 16),
        nn.ELU(),
        nn.Linear(16, 4),
    )
    model.train()
    loss_module = VAMP2Loss(dtype=torch.float64)
    parameters = tuple(model.parameters())
    baseline_state = {
        name: param.detach().clone() for name, param in model.state_dict().items()
    }

    def _train_step() -> tuple[float, float, float]:
        model.load_state_dict(baseline_state)
        optimizer = torch.optim.SGD(parameters, lr=0.05, momentum=0.0)
        optimizer.zero_grad(set_to_none=True)
        loss, score = loops.compute_loss_and_score(model, loss_module, x_t, x_tau, w)
        loss.backward()
        grad_norm = loops.compute_grad_norm(parameters)
        optimizer.step()
        return (
            float(loss.detach().cpu().item()),
            float(score.detach().cpu().item()),
            float(grad_norm),
        )

    loss_value, score_value, grad_norm_value = benchmark(_train_step)

    assert np.isfinite(loss_value)
    assert np.isfinite(score_value)
    assert grad_norm_value > 0.0

    final_state = {
        name: param.detach().clone() for name, param in model.state_dict().items()
    }
    assert any(
        not torch.allclose(final_state[name], baseline_state[name])
        for name in baseline_state
    )


def test_iter_pair_batches_sampling(benchmark):
    """Benchmark iter_pair_batches to ensure deterministic coverage of pairs."""

    import numpy as np

    from pmarlo.features.deeptica_trainer.sampler import iter_pair_batches
    from pmarlo.pairs.core import PairInfo

    total_pairs = 45
    batch_size = 12
    idx_t = np.arange(total_pairs, dtype=np.int64)
    idx_tau = idx_t + 5
    weights = np.ones(total_pairs, dtype=np.float32)
    pair_info = PairInfo(
        idx_t=idx_t, idx_tau=idx_tau, weights=weights, diagnostics={"lag": 5}
    )

    expected_batches = list(
        iter_pair_batches(pair_info, batch_size=batch_size, shuffle=True, seed=99)
    )

    def _collect_batches():
        return list(
            iter_pair_batches(pair_info, batch_size=batch_size, shuffle=True, seed=99)
        )

    batches = benchmark(_collect_batches)

    assert batches
    assert len(batches) == len(expected_batches)
    for observed, reference in zip(batches, expected_batches):
        assert np.array_equal(observed, reference)

    flattened = np.concatenate(batches)
    assert flattened.size == total_pairs
    assert np.array_equal(np.sort(flattened), np.arange(total_pairs))


def test_iter_pair_batches_data_loading(benchmark):
    """Benchmark mapping time-lagged pair batches onto frame data."""

    import numpy as np

    from pmarlo.features.deeptica_trainer.sampler import iter_pair_batches
    from pmarlo.pairs.core import PairInfo

    total_pairs = 36
    lag = 4
    idx_t = np.arange(total_pairs, dtype=np.int64)
    idx_tau = idx_t + lag
    weights = np.ones(total_pairs, dtype=np.float32)
    pair_info = PairInfo(
        idx_t=idx_t, idx_tau=idx_tau, weights=weights, diagnostics={"lag": lag}
    )

    frames = np.arange(total_pairs + lag, dtype=np.float32).reshape(-1, 1)

    def _load_batches():
        result: list[tuple[np.ndarray, np.ndarray]] = []
        for batch_idx in iter_pair_batches(
            pair_info, batch_size=10, shuffle=False, seed=None
        ):
            frames_t = frames[pair_info.idx_t[batch_idx]]
            frames_tau = frames[pair_info.idx_tau[batch_idx]]
            result.append((frames_t, frames_tau))
        return result

    batches = benchmark(_load_batches)

    assert batches
    total_loaded = 0
    for frames_t, frames_tau in batches:
        assert frames_t.shape == frames_tau.shape
        assert np.allclose(frames_tau - frames_t, float(lag))
        total_loaded += int(frames_t.shape[0])

    assert total_loaded == total_pairs


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

    from pmarlo.features.deeptica.core.dataset import create_dataset, create_loaders
    from pmarlo.features.deeptica.core.inputs import prepare_features
    from pmarlo.features.deeptica.core.model import build_network
    from pmarlo.features.deeptica.core.pairs import build_pair_info

    prep = prepare_features(small_trajs, tau_schedule=(1, 5, 10), seed=42)
    cfg = MockConfig()
    network = build_network(cfg, prep.scaler, seed=42)
    network.train()

    pair_info = build_pair_info(small_trajs, (1, 5, 10), pairs=None, weights=None)

    weights = (
        np.asarray(pair_info.weights, dtype=np.float32).reshape(-1)
        if pair_info.weights is not None
        else np.ones_like(pair_info.idx_t, dtype=np.float32)
    )
    dataset = create_dataset(prep.Z, pair_info.idx_t, pair_info.idx_tau, weights)
    bundle = create_loaders(dataset, MockConfig(batch_size=64, val_frac=0.2))
    train_loader = bundle.train_loader
    assert train_loader is not None

    optimizer = torch.optim.Adam(network.parameters(), lr=0.001)

    def _train_epoch():
        total_loss = 0.0
        for batch in train_loader:
            X_t = batch["data"]
            X_tau = batch["data_lag"]
            optimizer.zero_grad()
            output_t = network(X_t)
            output_tau = network(X_tau)
            loss = torch.mean((output_t - output_tau) ** 2)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / max(1, len(train_loader))

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
    from pmarlo.features.deeptica.core.inputs import prepare_features
    from pmarlo.features.deeptica.core.model import (
        apply_output_whitening,
        build_network,
    )
    from pmarlo.features.deeptica.core.pairs import build_pair_info

    prep = prepare_features(medium_trajs, tau_schedule=(1, 5, 10), seed=42)
    cfg = MockConfig()
    network = build_network(cfg, prep.scaler, seed=42)

    pair_info = build_pair_info(
        medium_trajs, prep.tau_schedule, pairs=None, weights=None
    )
    idx_tau = np.asarray(pair_info.idx_tau, dtype=np.int64)

    def _whiten():
        return apply_output_whitening(network, prep.Z, idx_tau, apply=False)

    whitened_net, info = benchmark(_whiten)
    assert whitened_net is not None
    assert "transform" in info and info["transform"]


def test_model_checkpointing_persists_state(benchmark, tmp_path):
    """Benchmark model checkpoint serialization and validate saved state."""

    import torch

    from pmarlo.ml.deeptica.trainer import checkpoint_if_better

    model = torch.nn.Sequential(
        torch.nn.Linear(8, 16),
        torch.nn.ReLU(),
        torch.nn.Linear(16, 4),
    )

    metrics = {"val_score": 0.85}
    checkpoint_path = tmp_path / "deeptica_best.pt"

    def _checkpoint_once() -> float:
        if checkpoint_path.exists():
            checkpoint_path.unlink()
        return checkpoint_if_better(
            model_net=model,
            checkpoint_path=checkpoint_path,
            metrics=metrics,
            metric_name="val_score",
            best_score=0.5,
        )

    new_best = benchmark(_checkpoint_once)

    assert checkpoint_path.exists()
    assert new_best == pytest.approx(metrics["val_score"])

    saved_state = torch.load(checkpoint_path, map_location="cpu")
    model_state = model.state_dict()
    assert saved_state.keys() == model_state.keys()
    for key, tensor in model_state.items():
        assert torch.allclose(saved_state[key], tensor)


def test_training_history_logging_to_csv(benchmark, tmp_path):
    """Benchmark history serialization by writing per-epoch metrics to CSV."""

    from pmarlo.features.deeptica.core.history import (
        LossHistory,
        collect_history_metrics,
        summarize_history,
    )

    epoch_metrics = [
        {
            "train_loss": 1.0 - 0.07 * idx,
            "val_loss": 1.2 - 0.05 * idx,
            "val_score": 0.3 + 0.04 * idx,
        }
        for idx in range(6)
    ]
    csv_path = tmp_path / "training_history.csv"

    def _write_history():
        history = LossHistory()
        for metrics in epoch_metrics:
            history.record_train(metrics)
            history.record_val(metrics)
        summary = summarize_history(history)
        collected = collect_history_metrics(summary)
        if csv_path.exists():
            csv_path.unlink()
        with csv_path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["epoch", "loss", "val_loss", "val_score"])
            n_epochs = len(collected["loss_curve"])
            assert n_epochs == len(collected["val_loss_curve"])
            assert n_epochs == len(collected["val_score_curve"])
            for idx in range(n_epochs):
                writer.writerow(
                    [
                        idx,
                        collected["loss_curve"][idx],
                        collected["val_loss_curve"][idx],
                        collected["val_score_curve"][idx],
                    ]
                )
        return csv_path, collected

    csv_path, collected = benchmark(_write_history)

    assert csv_path.exists()
    with csv_path.open("r", newline="") as handle:
        reader = list(csv.reader(handle))

    assert reader[0] == ["epoch", "loss", "val_loss", "val_score"]
    data_rows = reader[1:]
    assert len(data_rows) == len(collected["loss_curve"])
    for idx, row in enumerate(data_rows):
        epoch_str, loss_str, val_loss_str, val_score_str = row
        assert int(epoch_str) == idx
        assert float(loss_str) == pytest.approx(collected["loss_curve"][idx])
        assert float(val_loss_str) == pytest.approx(collected["val_loss_curve"][idx])
        assert float(val_score_str) == pytest.approx(collected["val_score_curve"][idx])


def test_vamp2_loss_core_computation(benchmark, medium_trajs):
    """Benchmark the core VAMP-2 loss evaluation."""

    import torch

    from pmarlo.features.deeptica.core.inputs import prepare_features
    from pmarlo.features.deeptica.core.model import build_network
    from pmarlo.features.deeptica.core.pairs import build_pair_info
    from pmarlo.features.deeptica.losses import VAMP2Loss

    prep = prepare_features(medium_trajs, tau_schedule=(1, 5, 10), seed=42)
    cfg = MockConfig()
    network = build_network(cfg, prep.scaler, seed=42)
    network.eval()

    pair_info = build_pair_info(
        medium_trajs, prep.tau_schedule, pairs=None, weights=None
    )

    limit = min(2048, pair_info.idx_t.size)
    idx_t = np.asarray(pair_info.idx_t[:limit], dtype=np.int64)
    idx_tau = np.asarray(pair_info.idx_tau[:limit], dtype=np.int64)
    weights = np.asarray(pair_info.weights[:limit], dtype=np.float32)

    with torch.no_grad():
        z0 = network(torch.as_tensor(prep.Z[idx_t], dtype=torch.float32))
        zt = network(torch.as_tensor(prep.Z[idx_tau], dtype=torch.float32))

    weight_tensor = torch.as_tensor(weights, dtype=torch.float32)
    vamp2 = VAMP2Loss(dtype=torch.float32)

    def _evaluate():
        return vamp2(z0, zt, weights=weight_tensor)

    loss, score = benchmark(_evaluate)
    assert loss.ndim == 0
    assert score.ndim == 0
    metrics = vamp2.latest_metrics
    assert "cond_C00" in metrics and metrics["cond_C00"] is not None


def test_deeptica_model_forward_core(benchmark, medium_trajs):
    """Benchmark a single DeepTICA network forward pass on normalized data."""

    import torch

    from pmarlo.features.deeptica.core.inputs import prepare_features
    from pmarlo.features.deeptica.core.model import build_network

    prep = prepare_features(medium_trajs, tau_schedule=(1, 5, 10), seed=42)
    cfg = MockConfig(output_dim=3, hidden_units=(64, 32, 16))
    network = build_network(cfg, prep.scaler, seed=42)
    network.eval()

    batch = torch.as_tensor(prep.Z[:512], dtype=torch.float32)

    def _forward():
        with torch.no_grad():
            return network(batch)

    outputs = benchmark(_forward)
    assert outputs.shape == (batch.shape[0], cfg.output_dim)
    assert torch.isfinite(outputs).all()
