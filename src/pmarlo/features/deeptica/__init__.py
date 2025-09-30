"""DeepTICA feature helpers with optional dependency fallbacks."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

logger = logging.getLogger(__name__)

__all__: list[str] = []


def _load_full_api() -> tuple[list[str], dict[str, Any]]:
    from . import _full as _full_impl  # pragma: no cover - optional extra

    exported = getattr(_full_impl, "__all__", None)
    if exported is None:
        exported = [name for name in vars(_full_impl) if not name.startswith("_")]
    namespace = {name: getattr(_full_impl, name) for name in exported}
    return list(exported), namespace


def _define_stub_api() -> tuple[list[str], dict[str, Any]]:
    DeepTICAConfig = _make_stub_config()
    IdentityNet = _make_identity_net()
    DeepTICAModel = _make_stub_model(DeepTICAConfig, IdentityNet)
    train_deeptica = _make_stub_trainer(DeepTICAConfig, DeepTICAModel, IdentityNet)

    exports: dict[str, Any] = {
        "DeepTICAConfig": DeepTICAConfig,
        "DeepTICAModel": DeepTICAModel,
        "train_deeptica": train_deeptica,
    }
    return list(exports), exports


def _make_stub_config():
    @dataclass(slots=True)
    class DeepTICAConfig:
        lag: int
        n_out: int = 2
        max_epochs: int = 5
        early_stopping: int = 2
        batch_size: int = 32
        hidden: Sequence[int] = field(default_factory=lambda: (32, 16))
        num_workers: int = 0
        linear_head: bool = False
        seed: int | None = None

    return DeepTICAConfig


def _make_identity_net():
    class _IdentityNet:
        def __init__(self, n_out: int) -> None:
            self.n_out = int(n_out)
            self._n_features = int(n_out)

        def __call__(self, X: np.ndarray) -> np.ndarray:
            arr = np.asarray(X, dtype=np.float64)
            if arr.ndim == 1:
                arr = arr.reshape(-1, self._n_features)
            if arr.shape[-1] != self._n_features:
                return np.zeros((arr.shape[0], self._n_features), dtype=np.float64)
            return arr

    return _IdentityNet


def _make_stub_model(DeepTICAConfig, IdentityNet):
    class DeepTICAModel:
        """Minimal DeepTICA model stub for dependency-free testing."""

        def __init__(
            self,
            config: Any,
            scaler: Any | None = None,
            net: Any | None = None,
            training_history: dict[str, Any] | None = None,
        ) -> None:
            self.config = config
            self.scaler = scaler
            self.net = net or IdentityNet(config.n_out)
            self.training_history = training_history or {}

        def transform(self, X: np.ndarray) -> np.ndarray:
            history = self.training_history
            mean = np.asarray(
                history.get(
                    "output_mean", np.zeros(self.config.n_out, dtype=np.float64)
                ),
                dtype=np.float64,
            )
            transform = np.asarray(
                history.get(
                    "output_transform", np.eye(self.config.n_out, dtype=np.float64)
                ),
                dtype=np.float64,
            )
            if bool(history.get("output_transform_applied", False)):
                return np.asarray(X, dtype=np.float64)
            return (np.asarray(X, dtype=np.float64) - mean) @ transform

    return DeepTICAModel


def _make_stub_trainer(DeepTICAConfig, DeepTICAModel, IdentityNet):
    def train_deeptica(
        X_list: Iterable[np.ndarray],
        lagged_pairs: tuple[np.ndarray, np.ndarray] | Sequence[np.ndarray],
        cfg: Any,
        *,
        weights: np.ndarray | None = None,
    ) -> Any:
        arrays = [np.asarray(arr, dtype=np.float64) for arr in X_list]
        if not arrays:
            raise ValueError("Expected at least one trajectory array")
        n_frames = sum(max(0, arr.shape[0]) for arr in arrays)
        history = _stub_training_history(cfg, n_frames)

        if weights is not None:
            weights_arr = np.asarray(weights, dtype=np.float64)
            if weights_arr.size not in {0, n_frames}:
                raise ValueError("weights must be empty or match the number of frames")

        model = DeepTICAModel(
            cfg,
            scaler=None,
            net=IdentityNet(cfg.n_out),
            training_history=history,
        )
        logger.warning(
            "DeepTICA optional dependencies missing; returning analytical stub model"
        )
        return model

    return train_deeptica


def _stub_training_history(cfg: Any, n_frames: int) -> dict[str, Any]:
    epochs = max(1, int(getattr(cfg, "max_epochs", 5)))
    loss_curve = np.linspace(1.0, 0.1, epochs, dtype=float).tolist()
    objective_curve = np.linspace(0.2, 0.95, epochs, dtype=float).tolist()
    val_curve = np.linspace(0.15, 0.9, epochs, dtype=float).tolist()
    grad_norm = np.linspace(0.5, 0.05, epochs, dtype=float).tolist()
    n_out = int(getattr(cfg, "n_out", 2))
    return {
        "loss_curve": loss_curve,
        "objective_curve": objective_curve,
        "val_score_curve": val_curve,
        "val_score": val_curve[-1],
        "var_z0_curve": np.full(epochs, 0.5, dtype=float).tolist(),
        "var_zt_curve": np.full(epochs, 0.55, dtype=float).tolist(),
        "cond_c00_curve": np.full(epochs, 0.6, dtype=float).tolist(),
        "cond_ctt_curve": np.full(epochs, 0.65, dtype=float).tolist(),
        "grad_norm_curve": grad_norm,
        "output_variance": np.full(n_out, 1.0, dtype=float).tolist(),
        "output_mean": np.zeros(n_out, dtype=float).tolist(),
        "output_transform": np.eye(n_out, dtype=float).tolist(),
        "output_transform_applied": False,
        "epochs_trained": epochs,
        "frames_seen": int(n_frames),
    }


try:  # pragma: no cover - optional extra
    __all__, _namespace = _load_full_api()
except Exception as exc:  # pragma: no cover - lightweight fallback path
    _IMPORT_ERROR: Exception | None = exc
else:
    globals().update(_namespace)
    _IMPORT_ERROR = None

if _IMPORT_ERROR is not None:
    __all__, _namespace = _define_stub_api()
    globals().update(_namespace)

__all__ = list(__all__)
