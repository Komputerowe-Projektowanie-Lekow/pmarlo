from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np

# Import heavy deps only when this module is used.
try:  # pragma: no cover - optional extra
    import torch
    from mlcolvar.cvs import DeepTICA
    from mlcolvar.features import StandardScaler
except Exception as e:  # pragma: no cover - optional extra
    raise ImportError("Install optional extra pmarlo[mlcv] to use Deep-TICA") from e


@dataclass(frozen=True)
class DeepTICAConfig:
    lag: int
    n_out: int = 2
    hidden: Tuple[int, ...] = (64, 64)
    activation: str = "tanh"
    learning_rate: float = 1e-3
    batch_size: int = 4096
    max_epochs: int = 200
    early_stopping: int = 20
    seed: int = 0
    reweight_mode: str = "scaled_time"  # or "none"


class DeepTICAModel:
    """Thin wrapper exposing a stable API around mlcolvar DeepTICA."""

    def __init__(self, cfg: DeepTICAConfig, scaler: Any, net: Any):
        self.cfg = cfg
        self.scaler = scaler
        self.net = net  # mlcolvar.cvs.DeepTICA

    def transform(self, X: np.ndarray) -> np.ndarray:
        Z = self.scaler.transform(np.asarray(X, dtype=np.float64))
        with torch.no_grad():
            y = self.net(Z)  # returns [n, n_out]
        return np.asarray(y, dtype=np.float64)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Config
        meta = json.dumps(
            asdict(self.cfg), sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        (path.with_suffix(".json")).write_text(meta, encoding="utf-8")
        # Net params
        torch.save({"state_dict": self.net.state_dict()}, path.with_suffix(".pt"))
        # Scaler params (numpy arrays)
        torch.save(
            {
                "mean": np.asarray(self.scaler.mean_),
                "std": np.asarray(self.scaler.scale_),
            },
            path.with_suffix(".scaler.pt"),
        )

    @classmethod
    def load(cls, path: Path) -> "DeepTICAModel":
        path = Path(path)
        cfg = DeepTICAConfig(
            **json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        )
        scaler_ckpt = torch.load(path.with_suffix(".scaler.pt"), map_location="cpu")
        scaler = StandardScaler()
        scaler.mean_ = np.asarray(scaler_ckpt["mean"], dtype=np.float64)
        scaler.scale_ = np.asarray(scaler_ckpt["std"], dtype=np.float64)
        net = DeepTICA(
            n_inputs=int(scaler.mean_.shape[0]),
            n_outputs=int(cfg.n_out),
            layers=tuple(int(h) for h in cfg.hidden),
            activation=str(cfg.activation),
        )
        state = torch.load(path.with_suffix(".pt"), map_location="cpu")
        net.load_state_dict(state["state_dict"])  # type: ignore[index]
        net.eval()
        return cls(cfg, scaler, net)

    def to_torchscript(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.net.eval()
        # Trace with double precision
        example = torch.zeros(1, int(self.scaler.mean_.shape[0]), dtype=torch.float64)
        ts = torch.jit.trace(self.net.to(torch.float64), example)
        out = path.with_suffix(".ts")
        ts.save(str(out))
        return out

    def plumed_snippet(self, model_path: Path) -> str:
        ts = Path(model_path).with_suffix(".ts").name
        # Emit one CV line per output for convenience; users can rename labels in PLUMED input.
        lines = [f"PYTORCH_MODEL FILE={ts} LABEL=mlcv"]
        for i in range(int(self.cfg.n_out)):
            lines.append(f"CV VALUE=mlcv.node-{i}")
        return "\n".join(lines) + "\n"


def train_deeptica(
    X_list: List[np.ndarray],
    pairs: Tuple[np.ndarray, np.ndarray],
    cfg: DeepTICAConfig,
    weights: Optional[np.ndarray] = None,
) -> DeepTICAModel:
    """Train Deep-TICA on concatenated features with provided time-lagged pairs.

    Parameters
    ----------
    X_list : list of [n_i, k] arrays
        Feature blocks (e.g., from shards); concatenated along axis=0.
    pairs : (idx_t, idx_tlag)
        Integer indices into the concatenated array representing lagged pairs.
    cfg : DeepTICAConfig
        Hyperparameters and optimization settings.
    weights : Optional[np.ndarray]
        Optional per-pair weights (e.g., scaled-time or bias reweighting).
    """

    X = np.concatenate([np.asarray(x, dtype=np.float64) for x in X_list], axis=0)
    scaler = StandardScaler().fit(X)
    Z = scaler.transform(X)

    net = DeepTICA(
        n_inputs=int(Z.shape[1]),
        n_outputs=int(cfg.n_out),
        layers=tuple(int(h) for h in cfg.hidden),
        activation=str(cfg.activation),
    )
    torch.manual_seed(int(cfg.seed))
    net.optimizer = torch.optim.Adam(net.parameters(), lr=float(cfg.learning_rate))

    idx_t, idx_tlag = pairs
    net.fit(
        Z,
        lagtime=int(cfg.lag),
        idx_t=np.asarray(idx_t, dtype=int),
        idx_tlag=np.asarray(idx_tlag, dtype=int),
        weights=None if weights is None else np.asarray(weights, dtype=np.float64),
        batch_size=int(cfg.batch_size),
        max_epochs=int(cfg.max_epochs),
        early_stopping_patience=int(cfg.early_stopping),
        shuffle=True,
    )
    net.eval()
    return DeepTICAModel(cfg, scaler, net)
