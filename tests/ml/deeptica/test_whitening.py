import numpy as np
import pytest

from pmarlo.analysis.msm import ensure_msm_inputs_whitened
from pmarlo.ml.deeptica.whitening import apply_output_transform

pytest.importorskip("mlcolvar")
torch = pytest.importorskip("torch")

from pmarlo.features.deeptica import DeepTICAConfig, DeepTICAModel  # noqa: E402


class DummyScaler:
    def __init__(self, n_features: int) -> None:
        self.mean_ = np.zeros(n_features, dtype=np.float64)
        self.scale_ = np.ones(n_features, dtype=np.float64)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=np.float64)


def test_apply_output_transform_scales_outputs() -> None:
    Y = np.array([[1.0, -1.0], [2.0, 0.5]], dtype=np.float64)
    mean = np.array([0.5, -0.5], dtype=np.float64)
    transform = np.array([[2.0, 0.0], [0.0, 0.25]], dtype=np.float64)

    whitened = apply_output_transform(Y, mean, transform, already_applied=False)
    expected = (Y - mean) @ transform
    assert np.allclose(whitened, expected)

    unchanged = apply_output_transform(whitened, mean, transform, already_applied=True)
    assert np.allclose(unchanged, whitened)


def test_deeptica_model_transform_applies_whitening() -> None:
    cfg = DeepTICAConfig(lag=2, n_out=2)
    scaler = DummyScaler(n_features=2)
    net = torch.nn.Identity()
    history = {
        "output_mean": [0.0, 0.0],
        "output_transform": [[2.0, 0.0], [0.0, 0.5]],
        "output_transform_applied": False,
    }
    model = DeepTICAModel(cfg, scaler, net, training_history=history)

    X = np.array([[1.0, 2.0], [-1.0, 4.0]], dtype=np.float64)
    Y = model.transform(X)

    expected = X @ np.array([[2.0, 0.0], [0.0, 0.5]], dtype=np.float64)
    assert np.allclose(Y, expected)
    # training_history should remain reusable (no in-place flip to True)
    assert model.training_history.get("output_transform_applied") is False


def test_msm_whitening_helper_updates_dataset_metadata() -> None:
    dataset = {
        "X": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
        "__artifacts__": {
            "mlcv_deeptica": {
                "output_mean": [0.0, 0.0],
                "output_transform": [[1.0, 0.0], [0.0, 2.0]],
                "output_transform_applied": False,
            }
        },
    }

    applied = ensure_msm_inputs_whitened(dataset)
    assert applied is True
    assert np.allclose(
        dataset["X"], np.array([[1.0, 4.0], [3.0, 8.0]], dtype=np.float64)
    )
    assert dataset["__artifacts__"]["mlcv_deeptica"]["output_transform_applied"] is True
