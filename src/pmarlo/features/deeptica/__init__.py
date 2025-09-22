"""DeepTICA feature helpers with optional dependency fallbacks."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

logger = logging.getLogger(__name__)

try:  # pragma: no cover - exercised when optional extras are installed
    import torch  # type: ignore
    import mlcolvar as _mlc  # type: ignore
    from sklearn.preprocessing import StandardScaler  # type: ignore
except Exception as exc:  # pragma: no cover - executed in lightweight test envs
    _IMPORT_ERROR = exc
else:  # pragma: no cover - exercised only in full environments
    from ._full import *  # type: ignore[F401,F403]
    __all__ = [name for name in globals().keys() if not name.startswith("_")]
    _IMPORT_ERROR = None

if _IMPORT_ERROR is None:
    pass
else:
    __all__ = ["DeepTICAConfig", "DeepTICAModel", "train_deeptica"]

    @dataclass(slots=True)
    class DeepTICAConfig:
        """Lightweight configuration used when mlcolvar/torch are unavailable."""

        lag: int
        n_out: int = 2
        max_epochs: int = 5
        early_stopping: int = 2
        batch_size: int = 32
        hidden: Sequence[int] = field(default_factory=lambda: (32, 16))
        num_workers: int = 0
        linear_head: bool = False
        seed: int | None = None

    class _IdentityNet:
        def __init__(self, n_out: int) -> None:
            self.n_out = int(n_out)

        def __call__(self, X: np.ndarray) -> np.ndarray:
            arr = np.asarray(X, dtype=np.float64)
            if arr.shape[-1] != self.n_out:
                return np.zeros((arr.shape[0], self.n_out), dtype=np.float64)
            return arr

    class DeepTICAModel:
        """Minimal DeepTICA model stub for dependency-free testing."""

        def __init__(
            self,
            config: DeepTICAConfig,
            scaler: Any | None = None,
            net: Any | None = None,
            training_history: dict[str, Any] | None = None,
        ) -> None:
            self.config = config
            self.scaler = scaler
            self.net = net or _IdentityNet(config.n_out)
            self.training_history = training_history or {}

        def transform(self, X: np.ndarray) -> np.ndarray:
            history = self.training_history
            mean = np.asarray(
                history.get("output_mean", np.zeros(self.config.n_out, dtype=np.float64)),
                dtype=np.float64,
            )
            transform = np.asarray(
                history.get(
                    "output_transform", np.eye(self.config.n_out, dtype=np.float64)
                ),
                dtype=np.float64,
            )
            applied = bool(history.get("output_transform_applied", False))
            data = np.asarray(X, dtype=np.float64)
            if applied:
                return data
            return (data - mean) @ transform

    def _compute_history(cfg: DeepTICAConfig, n_frames: int) -> dict[str, Any]:
        epochs = max(1, int(cfg.max_epochs))
        loss_curve = np.linspace(1.0, 0.1, epochs, dtype=float).tolist()
        objective_curve = np.linspace(0.2, 0.95, epochs, dtype=float).tolist()
        val_curve = np.linspace(0.15, 0.9, epochs, dtype=float).tolist()
        grad_norm = np.linspace(0.5, 0.05, epochs, dtype=float).tolist()
        history = {
            "loss_curve": loss_curve,
            "objective_curve": objective_curve,
            "val_score_curve": val_curve,
            "val_score": val_curve[-1],
            "var_z0_curve": np.full(epochs, 0.5, dtype=float).tolist(),
            "var_zt_curve": np.full(epochs, 0.55, dtype=float).tolist(),
            "cond_c00_curve": np.full(epochs, 0.6, dtype=float).tolist(),
            "cond_ctt_curve": np.full(epochs, 0.65, dtype=float).tolist(),
            "grad_norm_curve": grad_norm,
            "output_variance": np.full(cfg.n_out, 1.0, dtype=float).tolist(),
            "output_mean": np.zeros(cfg.n_out, dtype=float).tolist(),
            "output_transform": np.eye(cfg.n_out, dtype=float).tolist(),
            "output_transform_applied": False,
            "epochs_trained": epochs,
            "frames_seen": int(n_frames),
        }
        return history

    def train_deeptica(
        X_list: Iterable[np.ndarray],
        lagged_pairs: tuple[np.ndarray, np.ndarray] | Sequence[np.ndarray],
        cfg: DeepTICAConfig,
        *,
        weights: np.ndarray | None = None,
    ) -> DeepTICAModel:
        """Return a deterministic stub model when optional dependencies are missing."""

        arrays = [np.asarray(arr, dtype=np.float64) for arr in X_list]
        if not arrays:
            raise ValueError("Expected at least one trajectory array")
        n_frames = sum(max(0, arr.shape[0]) for arr in arrays)
        history = _compute_history(cfg, n_frames)
        model = DeepTICAModel(cfg, scaler=None, net=_IdentityNet(cfg.n_out), training_history=history)
        logger.warning(
            "DeepTICA optional dependencies missing; returning analytical stub model"
        )
        return model
