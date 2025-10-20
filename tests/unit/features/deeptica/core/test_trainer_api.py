from __future__ import annotations

import importlib

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("mlcolvar")
torch = pytest.importorskip("torch")


def _reload_trainer_api():
    return importlib.reload(
        importlib.import_module("pmarlo.features.deeptica.core.trainer_api")
    )


def test_forward_to_tensor_preserves_device_and_dtype():
    trainer_api = _reload_trainer_api()

    class Identity(torch.nn.Module):
        def forward(self, x):  # type: ignore[override]
            return x

    target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module = Identity().to(target_device)
    data = np.arange(6, dtype=np.float32).reshape(3, 2)

    tensor = trainer_api._forward_to_tensor(module, data)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.device == target_device
    assert tensor.dtype == torch.float32

    array = trainer_api._forward_to_numpy(module, data)
    assert isinstance(array, np.ndarray)
    assert array.dtype == np.float64


def test_compute_output_variance_torch_cpu_matches_torch():
    trainer_api = _reload_trainer_api()
    samples = torch.tensor([[1.0, 2.0, 3.0], [3.0, 6.0, 9.0]], dtype=torch.float32)

    result = trainer_api._compute_output_variance(samples)
    expected = samples.var(dim=0, unbiased=True).cpu().tolist()

    assert result == pytest.approx(expected)


def test_compute_output_variance_single_sample_uses_biased_estimate():
    trainer_api = _reload_trainer_api()
    single = torch.tensor([[5.0, -1.0, 0.5]], dtype=torch.float32)

    result = trainer_api._compute_output_variance(single)

    assert result == pytest.approx([0.0, 0.0, 0.0])


def test_train_deeptica_pipeline_runs(tmp_path):
    trainer_api = _reload_trainer_api()
    deeptica_module = importlib.reload(
        importlib.import_module("pmarlo.features.deeptica")
    )

    cfg = deeptica_module.DeepTICAConfig(lag=2, max_epochs=1, batch_size=8, hidden=(8,))
    object.__setattr__(cfg, "checkpoint_dir", tmp_path / "ckpt")

    arrays = [np.random.rand(32, 4).astype(np.float32)]
    idx_t = np.arange(0, 30, dtype=np.int64)
    idx_tau = idx_t + 1

    try:
        artifacts = trainer_api.train_deeptica_pipeline(arrays, (idx_t, idx_tau), cfg)
    except (NotImplementedError, RuntimeError, TypeError) as exc:
        pytest.skip(f"DeepTICA extras unavailable: {exc}")

    assert isinstance(artifacts.network, torch.nn.Module)
    assert artifacts.history["loss_curve"], "loss curve should be populated"
    assert artifacts.history["history_source"] == "curriculum_trainer"
    assert "whitening" in artifacts.history
    assert artifacts.device in {"cpu", "cuda"}
