from __future__ import annotations

import json
import os as _os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np

# Standardize math defaults to float32 end-to-end
import torch  # type: ignore

torch.set_float32_matmul_precision("high")
torch.set_default_dtype(torch.float32)


def set_all_seeds(seed: int = 2024) -> None:
    """Set RNG seeds across Python, NumPy, and Torch (CPU/GPU)."""
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if (
        hasattr(torch, "cuda") and torch.cuda.is_available()
    ):  # pragma: no cover - optional
        try:
            torch.cuda.manual_seed_all(int(seed))
        except Exception:
            pass


class PmarloApiIncompatibilityError(RuntimeError):
    """Raised when mlcolvar API layout does not expose expected classes."""


# Official DeepTICA import and helpers (mlcolvar>=1.2)
try:  # pragma: no cover - optional extra
    import mlcolvar as _mlc  # type: ignore
except Exception as e:  # pragma: no cover - optional extra
    raise ImportError("Install optional extra pmarlo[mlcv] to use Deep-TICA") from e
try:  # pragma: no cover - optional extra
    from mlcolvar.cvs import DeepTICA  # type: ignore
    from mlcolvar.utils.timelagged import (
        create_timelagged_dataset as _create_timelagged_dataset,  # type: ignore
    )
except Exception as e:  # pragma: no cover - optional extra
    raise PmarloApiIncompatibilityError(
        "mlcolvar installed but DeepTICA not found in expected locations"
    ) from e

# External scaling via scikit-learn (avoid internal normalization)
from sklearn.preprocessing import StandardScaler  # type: ignore


@dataclass(frozen=True)
class DeepTICAConfig:
    lag: int
    n_out: int = 2
    hidden: Tuple[int, ...] = (64, 64)
    activation: str = "tanh"
    learning_rate: float = 1e-3
    batch_size: int = 1024
    max_epochs: int = 200
    early_stopping: int = 20
    weight_decay: float = 1e-4
    log_every: int = 1
    seed: int = 0
    reweight_mode: str = "scaled_time"  # or "none"
    # New knobs for loaders and validation split
    val_frac: float = 0.1
    num_workers: int = 2
    # Optimization and regularization knobs
    lr_schedule: str = "cosine"  # "none" | "cosine"
    warmup_epochs: int = 5
    dropout: float = 0.1
    layer_norm_in: bool = True
    # Dataset splitting/loader control
    val_split: str = "by_shard"  # "by_shard" | "random"
    batches_per_epoch: int = 200


class DeepTICAModel:
    """Thin wrapper exposing a stable API around mlcolvar DeepTICA."""

    def __init__(
        self,
        cfg: DeepTICAConfig,
        scaler: Any,
        net: Any,
        *,
        device: str = "cpu",
        training_history: dict | None = None,
    ):
        self.cfg = cfg
        self.scaler = scaler
        self.net = net  # mlcolvar.cvs.DeepTICA
        self.device = str(device)
        self.training_history = dict(training_history or {})

    def transform(self, X: np.ndarray) -> np.ndarray:
        Z = self.scaler.transform(np.asarray(X, dtype=np.float64))
        with torch.no_grad():
            try:
                y = self.net(Z)  # type: ignore[misc]
            except Exception:
                y = self.net(torch.as_tensor(Z, dtype=torch.float32))
            if isinstance(y, torch.Tensor):
                y = y.detach().cpu().numpy()
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
        # Persist training history alongside the model
        try:
            hist = dict(self.training_history or {})
            if hist:
                # Write compact JSON
                (path.with_suffix(".history.json")).write_text(
                    json.dumps(hist, sort_keys=True, indent=2), encoding="utf-8"
                )
                # If a CSV metrics file was produced by CSVLogger, copy it as history.csv
                metrics_csv = hist.get("metrics_csv")
                if metrics_csv:
                    import shutil  # lazy import

                    metrics_csv_p = Path(str(metrics_csv))
                    if metrics_csv_p.exists():
                        out_csv = path.with_suffix(".history.csv")
                        try:
                            shutil.copyfile(str(metrics_csv_p), str(out_csv))
                        except Exception:
                            # Best-effort: ignore copy errors
                            pass
        except Exception:
            # History persistence should not block model saving
            pass

    @classmethod
    def load(cls, path: Path) -> "DeepTICAModel":
        path = Path(path)
        cfg = DeepTICAConfig(
            **json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        )
        scaler_ckpt = torch.load(path.with_suffix(".scaler.pt"), map_location="cpu")
        scaler = StandardScaler(with_mean=True, with_std=True)
        # Rehydrate the necessary attributes for transform()
        scaler.mean_ = np.asarray(scaler_ckpt["mean"], dtype=np.float64)
        scaler.scale_ = np.asarray(scaler_ckpt["std"], dtype=np.float64)
        # Some sklearn versions also check these, so set conservatively if missing
        try:  # pragma: no cover - attribute presence varies across versions
            scaler.n_features_in_ = int(scaler.mean_.shape[0])  # type: ignore[attr-defined]
        except Exception:
            pass
        # Rebuild network using the official constructor, then wrap with pre/post layers
        in_dim = int(scaler.mean_.shape[0])
        layers = [in_dim, *[int(h) for h in cfg.hidden], int(cfg.n_out)]
        core = DeepTICA(layers=layers, n_cvs=int(cfg.n_out), options={"norm_in": False})
        import torch.nn as _nn  # type: ignore

        class _PrePostWrapper(_nn.Module):  # type: ignore[misc]
            def __init__(self, inner, in_features: int, *, ln_in: bool, p_drop: float):
                super().__init__()
                self.ln = _nn.LayerNorm(in_features) if ln_in else _nn.Identity()
                p = float(max(0.0, min(1.0, p_drop)))
                self.drop_in = _nn.Dropout(p=p) if p > 0 else _nn.Identity()
                self.inner = inner
                self.drop_out = _nn.Dropout(p=p) if p > 0 else _nn.Identity()

            def forward(self, x):  # type: ignore[override]
                x = self.ln(x)
                x = self.drop_in(x)
                y = self.inner(x)
                y = self.drop_out(y)
                return y

        net = _PrePostWrapper(
            core,
            in_dim,
            ln_in=bool(getattr(cfg, "layer_norm_in", True)),
            p_drop=float(getattr(cfg, "dropout", 0.1)),
        )
        state = torch.load(path.with_suffix(".pt"), map_location="cpu")
        net.load_state_dict(state["state_dict"])  # type: ignore[index]
        net.eval()
        return cls(cfg, scaler, net)

    def to_torchscript(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.net.eval()
        # Trace with single precision (typical for inference)
        example = torch.zeros(1, int(self.scaler.mean_.shape[0]), dtype=torch.float32)
        ts = torch.jit.trace(self.net.to(torch.float32), example)
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

    import time as _time

    t0 = _time.time()
    # Deterministic behavior
    set_all_seeds(int(getattr(cfg, "seed", 2024)))
    # Prepare features and fit external scaler (float32 pipeline)
    X = np.concatenate([np.asarray(x, dtype=np.float32) for x in X_list], axis=0)
    scaler = StandardScaler(with_mean=True, with_std=True).fit(
        np.asarray(X, dtype=np.float64)
    )
    # Transform, then switch to float32 for training
    Z = scaler.transform(np.asarray(X, dtype=np.float64)).astype(np.float32, copy=False)

    # Build network with official constructor; disable internal normalization
    in_dim = int(Z.shape[1])
    layers = [in_dim, *[int(h) for h in cfg.hidden], int(cfg.n_out)]
    core = DeepTICA(layers=layers, n_cvs=int(cfg.n_out), options={"norm_in": False})
    # Wrap with input LayerNorm and light dropout for stability on tiny nets
    import torch.nn as _nn  # type: ignore

    class _PrePostWrapper(_nn.Module):  # type: ignore[misc]
        def __init__(self, inner, in_features: int, *, ln_in: bool, p_drop: float):
            super().__init__()
            self.ln = _nn.LayerNorm(in_features) if ln_in else _nn.Identity()
            p = float(max(0.0, min(1.0, p_drop)))
            self.drop_in = _nn.Dropout(p=p) if p > 0 else _nn.Identity()
            self.inner = inner
            self.drop_out = _nn.Dropout(p=p) if p > 0 else _nn.Identity()

        def forward(self, x):  # type: ignore[override]
            x = self.ln(x)
            x = self.drop_in(x)
            y = self.inner(x)
            y = self.drop_out(y)
            return y

    net = _PrePostWrapper(
        core,
        in_dim,
        ln_in=bool(getattr(cfg, "layer_norm_in", True)),
        p_drop=float(getattr(cfg, "dropout", 0.1)),
    )
    torch.manual_seed(int(cfg.seed))

    idx_t, idx_tlag = pairs

    # Validate or construct per‑shard pairs to ensure x_t != x_{t+tau}
    def _build_uniform_pairs_per_shard(
        blocks: List[np.ndarray], lag: int
    ) -> tuple[np.ndarray, np.ndarray]:
        L = max(1, int(lag))
        i_parts: List[np.ndarray] = []
        j_parts: List[np.ndarray] = []
        off = 0
        for b in blocks:
            n = int(np.asarray(b).shape[0])
            if n > L:
                i = np.arange(0, n - L, dtype=np.int64)
                j = i + L
                i_parts.append(off + i)
                j_parts.append(off + j)
            off += n
        if not i_parts:
            return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.int64)
        return (
            np.concatenate(i_parts).astype(np.int64, copy=False),
            np.concatenate(j_parts).astype(np.int64, copy=False),
        )

    def _needs_repair(i: np.ndarray | None, j: np.ndarray | None) -> bool:
        if i is None or j is None:
            return True
        if i.size == 0 or j.size == 0:
            return True
        try:
            d = np.asarray(j, dtype=np.int64) - np.asarray(i, dtype=np.int64)
            if d.size == 0:
                return True
            return bool(np.min(d) <= 0)
        except Exception:
            return True

    if _needs_repair(idx_t, idx_tlag):
        idx_t, idx_tlag = _build_uniform_pairs_per_shard(X_list, int(cfg.lag))

    # Simple telemetry: evaluate a proxy objective before and after training.
    def _vamp2_proxy(Y: np.ndarray, i: np.ndarray, j: np.ndarray) -> float:
        if Y.size == 0 or i.size == 0:
            return 0.0
        A = Y[i]
        B = Y[j]
        # Mean-center
        A = A - np.mean(A, axis=0, keepdims=True)
        B = B - np.mean(B, axis=0, keepdims=True)
        # Normalize columns
        A_std = np.std(A, axis=0, ddof=1) + 1e-12
        B_std = np.std(B, axis=0, ddof=1) + 1e-12
        A = A / A_std
        B = B / B_std
        # Component-wise Pearson r, squared, averaged across outputs
        num = np.sum(A * B, axis=0)
        den = A.shape[0] - 1
        r = num / max(1.0, den)
        return float(np.mean(r * r))

    # Objective before training using current net init
    with torch.no_grad():
        try:
            Y0 = net(Z)  # type: ignore[misc]
        except Exception:
            # Best-effort: convert to torch tensor if required by the backend
            Y0 = net(torch.as_tensor(Z, dtype=torch.float32)).detach().cpu().numpy()  # type: ignore[assignment]
        if isinstance(Y0, torch.Tensor):
            Y0 = Y0.detach().cpu().numpy()
    obj_before = _vamp2_proxy(
        Y0, np.asarray(idx_t, dtype=int), np.asarray(idx_tlag, dtype=int)
    )

    # Build time‑lagged dataset for training
    ds = None
    try:
        # Normalize index arrays and construct default weights (ones) when not provided
        if idx_t is None or idx_tlag is None or (len(idx_t) == 0 or len(idx_tlag) == 0):
            n = int(Z.shape[0])
            L = int(cfg.lag)
            if L < n:
                idx_t = np.arange(0, n - L, dtype=int)
                idx_tlag = idx_t + L
            else:
                idx_t = np.asarray([], dtype=int)
                idx_tlag = np.asarray([], dtype=int)
        idx_t = np.asarray(idx_t, dtype=int)
        idx_tlag = np.asarray(idx_tlag, dtype=int)
        if weights is None:
            weights_arr = np.ones((int(idx_t.shape[0]),), dtype=np.float32)
        else:
            weights_arr = np.asarray(weights, dtype=np.float32).reshape(-1)

        # Ensure explicit float32 tensors for lagged pairs
        # If you use a scaler, after scaler.fit, cast outputs to float32
        # using torch tensors to standardize dtype end-to-end.
        try:
            x_t_np = Z[idx_t]
            x_tau_np = Z[idx_tlag]
            x_t_tensor = torch.as_tensor(x_t_np, dtype=torch.float32)
            x_tau_tensor = torch.as_tensor(x_tau_np, dtype=torch.float32)
        except Exception:
            # Fallback via precomputed Z
            x_t_tensor = torch.as_tensor(Z[idx_t], dtype=torch.float32)
            x_tau_tensor = torch.as_tensor(Z[idx_tlag], dtype=torch.float32)

        # Preflight assertions: pairs must differ and weights must be positive on average
        try:
            n_pairs = int(x_t_tensor.shape[0])
            if n_pairs > 0:
                sel = np.random.default_rng(int(cfg.seed)).choice(
                    n_pairs, size=min(256, n_pairs), replace=False
                )
                xa = x_t_tensor[sel]
                xb = x_tau_tensor[sel]
                if torch.allclose(xa, xb):
                    raise ValueError(
                        "Invalid training pairs: x_t and x_{t+tau} are identical for sampled batch. "
                        "Check lag construction; expected strictly positive lag per shard."
                    )
                if float(np.mean(weights_arr)) <= 0.0:
                    raise ValueError(
                        "Invalid training weights: mean(weight) must be > 0"
                    )
        except Exception as _chk_e:
            # Surface the error early with a clear message
            raise

        # Prefer creating an explicit DictDataset with required keys
        try:
            from mlcolvar.data import DictDataset as _DictDataset  # type: ignore

            # Enforce float32 for all tensors expected by mlcolvar>=1.2
            payload: dict[str, Any] = {
                "data": x_t_tensor.detach()
                .cpu()
                .numpy()
                .astype(np.float32, copy=False),
                "data_lag": x_tau_tensor.detach()
                .cpu()
                .numpy()
                .astype(np.float32, copy=False),
                "weights": np.asarray(weights_arr, dtype=np.float32),
                # Some mlcolvar utilities also propagate weights for lagged frames
                "weights_lag": np.asarray(weights_arr, dtype=np.float32),
            }
            ds = _DictDataset(payload)
        except Exception:
            # Minimal fallback dataset compatible with torch DataLoader
            class _PairDataset(torch.utils.data.Dataset):  # type: ignore[type-arg]
                def __init__(self, A: np.ndarray, B: np.ndarray, W: np.ndarray):
                    # Enforce float32 tensors for stability
                    self.A = torch.as_tensor(A, dtype=torch.float32)
                    self.B = torch.as_tensor(B, dtype=torch.float32)
                    self.W = np.asarray(W, dtype=np.float32).reshape(-1)

                def __len__(self) -> int:  # noqa: D401
                    return int(self.A.shape[0])

                def __getitem__(self, idx: int) -> dict[str, Any]:
                    # Return strictly float32 to satisfy training_step contract
                    w = np.float32(self.W[idx])
                    return {
                        "data": self.A[idx],
                        "data_lag": self.B[idx],
                        "weights": w,
                        "weights_lag": w,
                    }

            _A = x_t_tensor
            _B = x_tau_tensor
            _W = weights_arr
            ds = _PairDataset(_A, _B, _W)
    except Exception:
        # As a last resort, fallback to helper and wrap to enforce weights
        base = _create_timelagged_dataset(Z, lag=int(cfg.lag))

        class _EnsureWeightsDataset(torch.utils.data.Dataset):  # type: ignore[type-arg]
            def __init__(self, inner):
                self.inner = inner

            def __len__(self) -> int:
                return len(self.inner)

            def __getitem__(self, idx: int) -> dict[str, Any]:
                d = dict(self.inner[idx])
                if "weights" not in d:
                    d["weights"] = np.float32(1.0)
                if "weights_lag" not in d:
                    d["weights_lag"] = np.float32(1.0)
                return d

        ds = _EnsureWeightsDataset(base)

    # Train the model using Lightning Trainer per mlcolvar docs
    # Import lightning with compatibility between new and legacy package names
    Trainer = None
    CallbackBase = None
    EarlyStoppingCls = None
    ModelCheckpointCls = None
    CSVLoggerCls = None
    TensorBoardLoggerCls = None
    lightning_available = False
    # Prefer pytorch_lightning when available to match mlcolvar's dependency
    try:  # pytorch_lightning
        from pytorch_lightning import Trainer as _PLTrainer  # type: ignore
        from pytorch_lightning.callbacks import Callback as _PLCallback  # type: ignore
        from pytorch_lightning.callbacks import (
            EarlyStopping as _PLEarlyStopping,  # type: ignore
        )
        from pytorch_lightning.callbacks import (
            ModelCheckpoint as _PLModelCheckpoint,  # type: ignore
        )
        from pytorch_lightning.loggers import CSVLogger as _PLCSVLogger  # type: ignore
        from pytorch_lightning.loggers import (
            TensorBoardLogger as _PLTBLogger,  # type: ignore
        )

        Trainer = _PLTrainer
        CallbackBase = _PLCallback
        EarlyStoppingCls = _PLEarlyStopping
        ModelCheckpointCls = _PLModelCheckpoint
        CSVLoggerCls = _PLCSVLogger
        TensorBoardLoggerCls = _PLTBLogger
        lightning_available = True
    except Exception:
        try:  # lightning >=2
            from lightning import Trainer as _LTrainer  # type: ignore
            from lightning.pytorch.callbacks import (
                Callback as _LCallback,  # type: ignore
            )
            from lightning.pytorch.callbacks import (
                EarlyStopping as _LEarlyStopping,  # type: ignore
            )
            from lightning.pytorch.callbacks import (
                ModelCheckpoint as _LModelCheckpoint,  # type: ignore
            )
            from lightning.pytorch.loggers import (
                CSVLogger as _LCSVLogger,  # type: ignore
            )
            from lightning.pytorch.loggers import (
                TensorBoardLogger as _LTBLogger,  # type: ignore
            )

            Trainer = _LTrainer
            CallbackBase = _LCallback
            EarlyStoppingCls = _LEarlyStopping
            ModelCheckpointCls = _LModelCheckpoint
            CSVLoggerCls = _LCSVLogger
            TensorBoardLoggerCls = _LTBLogger
            lightning_available = True
        except Exception:
            lightning_available = False

    # Optional DictModule wrapper if available; otherwise build plain DataLoaders
    dm = None
    train_loader = None
    val_loader = None
    try:
        from mlcolvar.data import DictModule as _DictModule  # type: ignore

        # Split: validation fraction as configured (enforce minimum 5%)
        nw = max(1, int(getattr(cfg, "num_workers", 1) or 1))
        val_frac = float(getattr(cfg, "val_frac", 0.1))
        if not (val_frac >= 0.05):
            val_frac = 0.05
        dm = _DictModule(
            ds,
            batch_size=int(cfg.batch_size),
            shuffle=True,
            split={"train": float(max(0.0, 1.0 - val_frac)), "val": float(val_frac)},
            num_workers=int(nw),
        )
    except Exception:
        # Fallback: build explicit train/val split and DataLoaders over dict-style dataset
        try:
            N = int(len(ds))  # type: ignore[arg-type]
        except Exception:
            N = 0
        if N >= 2:
            val_frac = float(getattr(cfg, "val_frac", 0.1))
            if not (val_frac >= 0.05):
                val_frac = 0.05
            n_val = max(1, int(val_frac * N))
            n_train = max(1, N - n_val)
            gen = torch.Generator().manual_seed(int(cfg.seed))
            train_ds, val_ds = torch.utils.data.random_split(ds, [n_train, n_val], generator=gen)  # type: ignore[assignment]
            nw = max(1, int(getattr(cfg, "num_workers", 1) or 1))
            pw = bool(nw > 0)
            train_loader = torch.utils.data.DataLoader(  # type: ignore[assignment]
                train_ds,
                batch_size=int(cfg.batch_size),
                shuffle=True,
                drop_last=False,
                num_workers=int(nw),
                persistent_workers=pw,
                prefetch_factor=2 if nw > 0 else None,
            )
            val_loader = torch.utils.data.DataLoader(  # type: ignore[assignment]
                val_ds,
                batch_size=int(cfg.batch_size),
                shuffle=False,
                drop_last=False,
                num_workers=int(nw),
                persistent_workers=pw,
                prefetch_factor=2 if nw > 0 else None,
            )
        else:
            # Degenerate tiny dataset: no validation split
            nw = max(1, int(getattr(cfg, "num_workers", 1) or 1))
            train_loader = torch.utils.data.DataLoader(  # type: ignore[assignment]
                ds,
                batch_size=int(cfg.batch_size),
                shuffle=True,
                drop_last=False,
                num_workers=int(nw),
                persistent_workers=bool(nw > 0),
                prefetch_factor=2 if nw > 0 else None,
            )
            val_loader = None

    # History callback to collect per-epoch losses if exposed by the model
    class _LossHistory(CallbackBase if lightning_available else object):  # type: ignore[misc]
        def __init__(self):
            self.losses: list[float] = []
            self.val_losses: list[float] = []

        def on_train_epoch_end(self, trainer, pl_module):  # type: ignore[no-untyped-def]
            try:
                metrics = dict(getattr(trainer, "callback_metrics", {}) or {})
                for key in ("train_loss", "loss"):
                    if key in metrics:
                        val = float(metrics[key])
                        self.losses.append(val)
                        break
            except Exception:
                pass

        def on_validation_epoch_end(self, trainer, pl_module):  # type: ignore[no-untyped-def]
            try:
                metrics = dict(getattr(trainer, "callback_metrics", {}) or {})
                for key in ("val_loss",):
                    if key in metrics:
                        val = float(metrics[key])
                        self.val_losses.append(val)
                        break
            except Exception:
                pass

    if lightning_available and Trainer is not None:
        callbacks = []
        hist_cb = _LossHistory()
        callbacks.append(hist_cb)
        try:
            if EarlyStoppingCls is not None:
                # Use validation loss if available, otherwise use training loss
                monitor_metric = (
                    "val_loss"
                    if dm is not None or val_loader is not None
                    else "train_loss"
                )
                # Firm patience and delta to avoid premature stop on flat zeros
                _me = int(getattr(cfg, "max_epochs", 200))
                _pat = max(30, _me // 5)
                # Construct with compatibility across lightning versions
                try:
                    es = EarlyStoppingCls(
                        monitor=monitor_metric,
                        patience=int(_pat),
                        mode="min",
                        min_delta=float(1e-5),
                        stopping_threshold=None,
                        check_finite=True,
                    )
                except TypeError:
                    # Older versions may not support some kwargs
                    es = EarlyStoppingCls(
                        monitor=monitor_metric,
                        patience=int(_pat),
                        mode="min",
                        min_delta=float(1e-5),
                    )
                callbacks.append(es)
        except Exception:
            pass

        # Best-only checkpointing
        ckpt_callback = None
        ckpt_callback_corr = None
        try:
            project_root = Path.cwd()
            checkpoints_root = project_root / "checkpoints"
            # Unique version per run to avoid overwrite
            version_str = f"{int(t0)}-{_os.getpid()}"
            ckpt_dir = checkpoints_root / "deeptica" / version_str
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            if ModelCheckpointCls is not None:
                ckpt_callback = ModelCheckpointCls(
                    dirpath=str(ckpt_dir),
                    filename="epoch={epoch:03d}-step={step}-val={val_loss:.5f}",
                    monitor=(
                        "val_loss"
                        if dm is not None or val_loader is not None
                        else "train_loss"
                    ),
                    mode="min",
                    save_top_k=3,
                    save_last=True,
                    every_n_epochs=5,
                )
                callbacks.append(ckpt_callback)
                # A second checkpoint tracking validation correlation (maximize)
                try:
                    ckpt_callback_corr = ModelCheckpointCls(
                        dirpath=str(ckpt_dir),
                        filename="epoch={epoch:03d}-step={step}-corr={val_corr_0:.5f}",
                        monitor="val_corr_0",
                        mode="max",
                        save_top_k=3,
                        save_last=False,
                        every_n_epochs=5,
                    )
                    callbacks.append(ckpt_callback_corr)
                except Exception:
                    ckpt_callback_corr = None
        except Exception:
            ckpt_callback = None
            ckpt_callback_corr = None

        # Loggers: CSV always (under checkpoints), TensorBoard optional
        loggers = []
        metrics_csv_path = None
        try:
            if CSVLoggerCls is not None:
                checkpoints_root = Path.cwd() / "checkpoints"
                checkpoints_root.mkdir(parents=True, exist_ok=True)
                # Reuse version part from ckpt_dir when available
                try:
                    version_str = ckpt_dir.name  # type: ignore[name-defined]
                except Exception:
                    version_str = f"{int(t0)}-{_os.getpid()}"
                csv_logger = CSVLoggerCls(
                    save_dir=str(checkpoints_root), name="deeptica", version=version_str
                )
                loggers.append(csv_logger)
                # Resolve metrics.csv location for later export
                try:
                    log_dir = Path(
                        getattr(
                            csv_logger,
                            "log_dir",
                            Path(checkpoints_root) / "deeptica" / version_str,
                        )
                    ).resolve()
                    metrics_csv_path = log_dir / "metrics.csv"
                except Exception:
                    metrics_csv_path = None
            if TensorBoardLoggerCls is not None:
                tb_logger = TensorBoardLoggerCls(
                    save_dir=str(Path.cwd() / "runs"), name="deeptica_tb"
                )
                loggers.append(tb_logger)
        except Exception:
            pass

        # Enable progress bar via env flag when desired
        _pb_env = str(_os.getenv("PMARLO_MLCV_PROGRESS", "0")).strip().lower()
        _pb = _pb_env in {"1", "true", "yes", "on"}
        # Wrap underlying model in a LightningModule so PL Trainer can optimize it
        try:
            try:
                import pytorch_lightning as pl  # type: ignore
            except Exception:
                import lightning.pytorch as pl  # type: ignore

            class DeepTICALightningWrapper(pl.LightningModule):  # type: ignore
                def __init__(
                    self,
                    inner,
                    lr: float,
                    weight_decay: float,
                    history_dir: str | None = None,
                    *,
                    lr_schedule: str = "cosine",
                    warmup_epochs: int = 5,
                    max_epochs: int = 200,
                ):
                    super().__init__()
                    self.inner = inner
                    # keep hparams for checkpointing/logging
                    self.save_hyperparameters(
                        {
                            "lr": float(lr),
                            "weight_decay": float(weight_decay),
                            "lr_schedule": str(lr_schedule),
                            "warmup_epochs": int(max(0, warmup_epochs)),
                            "max_epochs": int(max_epochs),
                        }
                    )
                    # Expose inner DeepTICA submodules at the LightningModule level for summary
                    try:
                        import torch.nn as _nn  # type: ignore

                        # Resolve DeepTICA core even if wrapped in a pre/post module
                        _core = getattr(inner, "inner", inner)
                        # Attach known submodules when present (do not create new modules)
                        _nn_mod = getattr(_core, "nn", None)
                        if isinstance(_nn_mod, _nn.Module):
                            self.nn = _nn_mod  # type: ignore[attr-defined]
                        _tica_mod = getattr(_core, "tica", None)
                        if isinstance(_tica_mod, _nn.Module):
                            self.tica = _tica_mod  # type: ignore[attr-defined]
                        # Loss module/function may be exposed under different names; attach when Module
                        _loss_mod = getattr(_core, "loss", None)
                        if not isinstance(_loss_mod, _nn.Module):
                            _loss_mod = getattr(_core, "_loss", None)
                        if isinstance(_loss_mod, _nn.Module):
                            self.loss_fn = _loss_mod  # type: ignore[attr-defined]
                    except Exception:
                        pass
                    # history directory for per-epoch JSONL metric records
                    try:
                        self.history_dir = (
                            Path(history_dir) if history_dir is not None else None
                        )
                        if self.history_dir is not None:
                            self.history_dir.mkdir(parents=True, exist_ok=True)
                            self.history_file = self.history_dir / "history.jsonl"
                        else:
                            self.history_file = None
                    except Exception:
                        self.history_dir = None
                        self.history_file = None

                def forward(self, x):  # type: ignore[override]
                    return self.inner(x)

                def _norm_batch(self, batch):
                    if isinstance(batch, dict):
                        d = dict(batch)
                        for k in ("data", "data_lag"):
                            if k in d and isinstance(d[k], torch.Tensor):
                                d[k] = d[k].to(self.device, dtype=torch.float32)
                        if "weights" not in d and "data" in d:
                            d["weights"] = torch.ones(
                                d["data"].shape[0],
                                device=self.device,
                                dtype=torch.float32,
                            )
                        if "weights_lag" not in d and "data_lag" in d:
                            d["weights_lag"] = torch.ones(
                                d["data_lag"].shape[0],
                                device=self.device,
                                dtype=torch.float32,
                            )
                        return d
                    if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                        x_t, x_tau = batch[0], batch[1]
                        x_t = x_t.to(self.device, dtype=torch.float32)
                        x_tau = x_tau.to(self.device, dtype=torch.float32)
                        w = torch.ones(
                            x_t.shape[0], device=self.device, dtype=torch.float32
                        )
                        return {
                            "data": x_t,
                            "data_lag": x_tau,
                            "weights": w,
                            "weights_lag": w,
                        }
                    return batch

                def training_step(self, batch, batch_idx):  # type: ignore[override]
                    b = self._norm_batch(batch)
                    loss = None
                    # Use mlcolvar-provided loss if present
                    if hasattr(self.inner, "_loss"):
                        out = self.inner._loss(b)
                        if isinstance(out, (tuple, list)):
                            loss = out[0]
                        else:
                            loss = out  # type: ignore[assignment]
                    else:
                        y_t = self.inner(b["data"])  # type: ignore[index]
                        y_tau = self.inner(b["data_lag"])  # type: ignore[index]
                        loss = torch.nn.functional.mse_loss(y_t, y_tau)
                    self.log(
                        "train_loss", loss, on_step=False, on_epoch=True, prog_bar=True
                    )
                    return loss

                def validation_step(self, batch, batch_idx):  # type: ignore[override]
                    b = self._norm_batch(batch)
                    if hasattr(self.inner, "_loss"):
                        out = self.inner._loss(b)
                        loss = out[0] if isinstance(out, (tuple, list)) else out  # type: ignore[assignment]
                    else:
                        y_t = self.inner(b["data"])  # type: ignore[index]
                        y_tau = self.inner(b["data_lag"])  # type: ignore[index]
                        loss = torch.nn.functional.mse_loss(y_t, y_tau)
                    self.log(
                        "val_loss", loss, on_step=False, on_epoch=True, prog_bar=True
                    )
                    # Diagnostics: generalized eigenvalues, per-CV autocorr, whitening norm
                    try:
                        # Ensure predictions for diagnostics
                        y_t = self.inner(b["data"])  # type: ignore[index]
                        y_tau = self.inner(b["data_lag"])  # type: ignore[index]
                        # Center
                        y_t_c = y_t - torch.mean(y_t, dim=0, keepdim=True)
                        y_tau_c = y_tau - torch.mean(y_tau, dim=0, keepdim=True)
                        n = max(1, y_t.shape[0] - 1)
                        C0 = (y_t_c.T @ y_t_c) / float(n)
                        Ctt = (y_tau_c.T @ y_tau_c) / float(n)
                        Ctau = (y_t_c.T @ y_tau_c) / float(n)
                        # Whitening matrix C0^{-1/2}
                        evals, evecs = torch.linalg.eigh((C0 + C0.T) * 0.5)
                        eps = torch.tensor(1e-8, dtype=evals.dtype, device=evals.device)
                        inv_sqrt = torch.diag(
                            torch.rsqrt(torch.clamp(evals, min=float(eps)))
                        )
                        W = evecs @ inv_sqrt @ evecs.T
                        M = W @ Ctau @ W.T
                        Ms = (M + M.T) * 0.5
                        vals = torch.linalg.eigvalsh(Ms)
                        # Sort descending
                        vals, _ = torch.sort(vals, descending=True)
                        k = min(int(y_t.shape[1]), 4)
                        for i in range(k):
                            self.log(
                                f"val_eig_{i}",
                                vals[i].float(),
                                on_step=False,
                                on_epoch=True,
                                prog_bar=False,
                            )
                        # Per-CV autocorrelation
                        var_t = torch.diag(C0)
                        var_tau = torch.diag(Ctt)
                        corr = torch.diag(Ctau) / torch.sqrt(
                            torch.clamp(var_t * var_tau, min=1e-12)
                        )
                        for i in range(min(int(corr.shape[0]), 4)):
                            self.log(
                                f"val_corr_{i}",
                                corr[i].float(),
                                on_step=False,
                                on_epoch=True,
                                prog_bar=False,
                            )
                        # Whitening matrix Frobenius norm
                        whiten_norm = torch.linalg.norm(W, ord="fro")
                        self.log(
                            "val_whiten_norm",
                            whiten_norm.float(),
                            on_step=False,
                            on_epoch=True,
                            prog_bar=False,
                        )
                        # Append a single record per epoch to history.jsonl (first batch only)
                        if int(batch_idx) == 0:
                            try:
                                if getattr(self, "history_file", None) is not None:
                                    rec = {
                                        "epoch": int(self.current_epoch),
                                        "val_loss": float(loss.detach().cpu().item()),
                                        "val_eigs": [
                                            float(vals[i].detach().cpu().item())
                                            for i in range(k)
                                        ],
                                        "val_corr": [
                                            float(corr[i].detach().cpu().item())
                                            for i in range(min(int(corr.shape[0]), 4))
                                        ],
                                        "val_whiten_norm": float(
                                            whiten_norm.detach().cpu().item()
                                        ),
                                    }
                                    with open(
                                        self.history_file, "a", encoding="utf-8"
                                    ) as fh:
                                        fh.write(json.dumps(rec, sort_keys=True) + "\n")
                            except Exception:
                                pass
                    except Exception:
                        # Diagnostics are best-effort; do not fail validation if they error
                        pass
                    return loss

                def configure_optimizers(self):  # type: ignore[override]
                    # Adam with mild weight decay for stability
                    opt = torch.optim.Adam(
                        self.parameters(),
                        lr=float(self.hparams.lr),
                        weight_decay=float(self.hparams.weight_decay),
                    )
                    sched_name = (
                        str(getattr(self.hparams, "lr_schedule", "cosine"))
                        if hasattr(self, "hparams")
                        else "cosine"
                    )
                    warmup = (
                        int(getattr(self.hparams, "warmup_epochs", 5))
                        if hasattr(self, "hparams")
                        else 5
                    )
                    maxe = (
                        int(getattr(self.hparams, "max_epochs", 200))
                        if hasattr(self, "hparams")
                        else 200
                    )
                    if sched_name == "cosine":
                        try:
                            import math as _math  # noqa: F401

                            from torch.optim.lr_scheduler import (  # type: ignore
                                CosineAnnealingLR,
                                LambdaLR,
                                SequentialLR,
                            )

                            scheds = []
                            milestones = []
                            if warmup and warmup > 0:

                                def _lr_lambda(epoch: int):
                                    return min(
                                        1.0, float(epoch + 1) / float(max(1, warmup))
                                    )

                                scheds.append(LambdaLR(opt, lr_lambda=_lr_lambda))
                                milestones.append(int(warmup))
                            T_max = max(1, maxe - max(0, warmup))
                            scheds.append(CosineAnnealingLR(opt, T_max=T_max))
                            if len(scheds) > 1:
                                sch = SequentialLR(opt, scheds, milestones=milestones)
                            else:
                                sch = scheds[0]
                            return {
                                "optimizer": opt,
                                "lr_scheduler": {"scheduler": sch, "interval": "epoch"},
                            }
                        except Exception:
                            # Fallback to ReduceLROnPlateau if SequentialLR/LambdaLR unavailable
                            sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
                                opt, mode="min", factor=0.5, patience=5
                            )
                            return {
                                "optimizer": opt,
                                "lr_scheduler": {
                                    "scheduler": sch,
                                    "monitor": "val_loss",
                                },
                            }
                    else:
                        # No scheduler
                        return opt

            # Choose a persistent directory for per-epoch JSONL logging
            try:
                hist_dir = (
                    ckpt_dir
                    if "ckpt_dir" in locals() and ckpt_dir is not None
                    else (Path.cwd() / "runs" / "deeptica" / str(int(t0)))
                )
            except Exception:
                hist_dir = None
            wrapped = DeepTICALightningWrapper(
                net,
                lr=float(cfg.learning_rate),
                weight_decay=float(cfg.weight_decay),
                history_dir=str(hist_dir) if hist_dir is not None else None,
                lr_schedule=str(getattr(cfg, "lr_schedule", "cosine")),
                warmup_epochs=int(getattr(cfg, "warmup_epochs", 5)),
                max_epochs=int(getattr(cfg, "max_epochs", 200)),
            )
        except Exception:
            # If Lightning is completely unavailable, fall back to model.fit (handled below)
            wrapped = net

        # Enforce minimum training duration to avoid early flat-zero stalls
        _max_epochs = int(getattr(cfg, "max_epochs", 200))
        _min_epochs = max(1, min(50, _max_epochs // 4))
        trainer = Trainer(
            max_epochs=_max_epochs,
            min_epochs=_min_epochs,
            enable_progress_bar=_pb,
            logger=loggers if loggers else False,
            callbacks=callbacks,
            deterministic=True,
            log_every_n_steps=1,
            enable_checkpointing=True,
            gradient_clip_val=1.0,
        )

        if dm is not None:
            trainer.fit(model=wrapped, datamodule=dm)
        else:
            trainer.fit(
                model=wrapped,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader,
            )

        # Persist artifacts info
        try:
            if ckpt_callback is not None and getattr(
                ckpt_callback, "best_model_path", None
            ):
                best_path = str(getattr(ckpt_callback, "best_model_path"))
            else:
                best_path = None
            if ckpt_callback_corr is not None and getattr(
                ckpt_callback_corr, "best_model_path", None
            ):
                best_path_corr = str(getattr(ckpt_callback_corr, "best_model_path"))
            else:
                best_path_corr = None
        except Exception:
            best_path = None
            best_path_corr = None
    else:
        # Fallback: if the model exposes a .fit(...) method, use it (older mlcolvar)
        if hasattr(net, "fit"):
            try:
                getattr(net, "fit")(
                    ds,
                    batch_size=int(cfg.batch_size),
                    max_epochs=int(cfg.max_epochs),
                    early_stopping_patience=int(cfg.early_stopping),
                    shuffle=False,
                )
            except TypeError:
                # Older API: pass arrays and indices directly
                # Ensure weights are always provided (mlcolvar>=1.2 may require them)
                _w = weights_arr
                getattr(net, "fit")(
                    Z,
                    lagtime=int(cfg.lag),
                    idx_t=np.asarray(idx_t, dtype=int),
                    idx_tlag=np.asarray(idx_tlag, dtype=int),
                    weights=_w,
                    batch_size=int(cfg.batch_size),
                    max_epochs=int(cfg.max_epochs),
                    early_stopping_patience=int(cfg.early_stopping),
                    shuffle=False,
                )
            except Exception:
                # Last-resort minimal loop: no-op to avoid crash; metrics will reflect proxy objective only
                pass
        else:
            raise ImportError(
                "Lightning (lightning or pytorch_lightning) is required for Deep-TICA training"
            )
    net.eval()
    with torch.no_grad():
        try:
            Y1 = net(Z)  # type: ignore[misc]
        except Exception:
            Y1 = net(torch.as_tensor(Z, dtype=torch.float32)).detach().cpu().numpy()  # type: ignore[assignment]
        if isinstance(Y1, torch.Tensor):
            Y1 = Y1.detach().cpu().numpy()
    obj_after = _vamp2_proxy(
        Y1, np.asarray(idx_t, dtype=int), np.asarray(idx_tlag, dtype=int)
    )

    # Prefer losses collected during training if available; otherwise proxy objective
    loss_curve: list[float] | None = None
    val_loss_curve: list[float] | None = None
    try:
        if lightning_available:
            if hist_cb.losses:
                loss_curve = [float(x) for x in hist_cb.losses]
            if hist_cb.val_losses:
                val_loss_curve = [float(x) for x in hist_cb.val_losses]
    except Exception:
        loss_curve = None
        val_loss_curve = None

    history = {
        "loss_curve": (
            loss_curve
            if loss_curve is not None
            else [float(1.0 - obj_before), float(1.0 - obj_after)]
        ),
        "val_loss_curve": val_loss_curve,
        "objective_curve": [float(obj_before), float(obj_after)],
        "epochs": [0, int(cfg.max_epochs)],
        "log_every": int(cfg.log_every),
        "wall_time_s": float(max(0.0, _time.time() - t0)),
    }

    # Attach logger paths and best checkpoint if available
    try:
        if lightning_available:
            if "metrics_csv_path" in locals() and metrics_csv_path:
                history["metrics_csv"] = str(metrics_csv_path)
            if "best_path" in locals() and best_path:
                history["best_ckpt_path"] = str(best_path)
            if "best_path_corr" in locals() and best_path_corr:
                history["best_ckpt_path_corr"] = str(best_path_corr)
    except Exception:
        pass

    # Compute top eigenvalues at the end for summary (whitened generalized eigenvalues)
    try:
        with torch.no_grad():
            Y = net(torch.as_tensor(Z, dtype=torch.float32))  # type: ignore[assignment]
            if isinstance(Y, torch.Tensor):
                Y = Y.detach().cpu().numpy()
        # If pairs are available, use them to build y_t/y_tau; else fallback to consecutive lag
        if idx_t is None or idx_tlag is None or len(idx_t) == 0:
            L = int(max(1, getattr(cfg, "lag", 1)))
            i_eval = np.arange(0, max(0, Y.shape[0] - L), dtype=int)
            j_eval = i_eval + L
        else:
            i_eval = np.asarray(idx_t, dtype=int)
            j_eval = np.asarray(idx_tlag, dtype=int)
        y_t = np.asarray(Y, dtype=np.float64)[i_eval]
        y_tau = np.asarray(Y, dtype=np.float64)[j_eval]
        # Center and covariances
        y_t_c = y_t - np.mean(y_t, axis=0, keepdims=True)
        y_tau_c = y_tau - np.mean(y_tau, axis=0, keepdims=True)
        n_eval = max(1, y_t_c.shape[0] - 1)
        C0_np = (y_t_c.T @ y_t_c) / float(n_eval)
        Ctau_np = (y_t_c.T @ y_tau_c) / float(n_eval)
        # Whitening via eigh
        evals_np, evecs_np = np.linalg.eigh((C0_np + C0_np.T) * 0.5)
        evals_np = np.clip(evals_np, 1e-12, None)
        inv_sqrt = np.diag(1.0 / np.sqrt(evals_np))
        W_np = evecs_np @ inv_sqrt @ evecs_np.T
        M_np = W_np @ Ctau_np @ W_np.T
        M_sym = (M_np + M_np.T) * 0.5
        eigs_np = np.linalg.eigvalsh(M_sym)
        eigs_np = np.sort(eigs_np)[::-1]
        top_eigs = [float(x) for x in eigs_np[: min(int(cfg.n_out), 4)]]
    except Exception:
        top_eigs = None

    # Write a summary JSON into the checkpoint directory if available
    try:
        summary_dir = None
        if "ckpt_dir" in locals() and ckpt_dir is not None:
            summary_dir = ckpt_dir
        else:
            # fallback to CSV logger dir if present
            if "metrics_csv_path" in locals() and metrics_csv_path is not None:
                summary_dir = Path(metrics_csv_path).parent
        if summary_dir is not None:
            summary = {
                "config": asdict(cfg),
                "final_metrics": {
                    "train_loss_last": (
                        (history.get("loss_curve") or [None])[-1]
                        if isinstance(history.get("loss_curve"), list)
                        else None
                    ),
                    "val_loss_last": (
                        (history.get("val_loss_curve") or [None])[-1]
                        if isinstance(history.get("val_loss_curve"), list)
                        else None
                    ),
                },
                "wall_time_s": float(history.get("wall_time_s", 0.0)),
                "scaler": {
                    "n_features": int(
                        getattr(scaler, "n_features_in_", 0)
                        or len(getattr(scaler, "mean_", []))
                    ),
                    "mean": np.asarray(getattr(scaler, "mean_", []))
                    .astype(float)
                    .tolist(),
                    "std": np.asarray(getattr(scaler, "scale_", []))
                    .astype(float)
                    .tolist(),
                },
                "top_eigenvalues": top_eigs,
                "artifacts": {
                    "metrics_csv": (
                        str(metrics_csv_path)
                        if "metrics_csv_path" in locals()
                        and metrics_csv_path is not None
                        else None
                    ),
                    "best_by_loss": (
                        str(best_path)
                        if "best_path" in locals() and best_path is not None
                        else None
                    ),
                    "best_by_corr": (
                        str(best_path_corr)
                        if "best_path_corr" in locals() and best_path_corr is not None
                        else None
                    ),
                    "last_ckpt": (
                        str((summary_dir / "last.ckpt"))
                        if (summary_dir / "last.ckpt").exists()
                        else None
                    ),
                },
            }
            (Path(summary_dir) / "training_summary.json").write_text(
                json.dumps(summary, sort_keys=True, indent=2)
            )
    except Exception:
        pass

    device = "cuda" if (hasattr(torch, "cuda") and torch.cuda.is_available()) else "cpu"
    return DeepTICAModel(cfg, scaler, net, device=device, training_history=history)
