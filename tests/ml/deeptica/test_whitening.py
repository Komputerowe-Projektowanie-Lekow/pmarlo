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
    expected -= expected.mean(axis=0, keepdims=True)
    if expected.shape[0] > expected.shape[1]:
        covariance = (expected.T @ expected) / float(expected.shape[0])
        chol = np.linalg.cholesky(covariance)
        expected = np.linalg.solve(chol.T, expected.T).T
        expected -= expected.mean(axis=0, keepdims=True)
    assert np.allclose(whitened, expected)

    unchanged = apply_output_transform(whitened, mean, transform, already_applied=True)
    assert np.allclose(unchanged, whitened)


def test_apply_output_transform_removes_residual_mean_drift() -> None:
    rng = np.random.default_rng(1905)
    mean = rng.normal(loc=0.0, scale=3.0, size=6)
    factors = rng.normal(size=(6, 6))
    covariance = factors @ factors.T + 6 * np.eye(6)
    cholesky = np.linalg.cholesky(covariance)

    base = rng.standard_normal(size=(4096, 6))
    correlated = base @ cholesky.T + mean
    transform = np.linalg.inv(cholesky.T)

    whitened = apply_output_transform(
        correlated, mean=mean, W=transform, already_applied=False
    )

    drift = np.mean(whitened, axis=0)
    assert np.allclose(drift, 0.0, atol=1e-12)
    covariance_whitened = np.cov(whitened, rowvar=False, ddof=0)
    assert np.allclose(covariance_whitened, np.eye(6), atol=1e-3)


def test_apply_output_transform_requires_metadata() -> None:
    Y = np.array([[1.0, 2.0]], dtype=np.float64)

    with pytest.raises(ValueError):
        apply_output_transform(Y, mean=None, W=np.eye(2), already_applied=False)

    with pytest.raises(ValueError):
        apply_output_transform(Y, mean=np.zeros(2), W=None, already_applied=False)


def test_apply_output_transform_respects_string_flags() -> None:
    Y = np.array([[1.0, 2.0]], dtype=np.float64)
    mean = np.zeros(2, dtype=np.float64)
    transform = np.eye(2, dtype=np.float64)

    whitened = apply_output_transform(Y, mean, transform, already_applied="false")
    expected = (Y - mean) @ transform
    expected -= expected.mean(axis=0, keepdims=True)
    assert np.allclose(whitened, expected)

    with pytest.raises(ValueError):
        apply_output_transform(Y, mean, transform, already_applied="definitely")


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
    expected -= expected.mean(axis=0, keepdims=True)
    if expected.shape[0] > expected.shape[1]:
        covariance = (expected.T @ expected) / float(expected.shape[0])
        chol = np.linalg.cholesky(covariance)
        expected = np.linalg.solve(chol.T, expected.T).T
        expected -= expected.mean(axis=0, keepdims=True)
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
        dataset["X"], np.array([[-1.0, -2.0], [1.0, 2.0]], dtype=np.float64)
    )
    assert dataset["__artifacts__"]["mlcv_deeptica"]["output_transform_applied"] is True
