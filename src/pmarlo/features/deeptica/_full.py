from __future__ import annotations

import json
import logging
import os as _os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional, Tuple

import numpy as np

# Standardize math defaults to float32 end-to-end
import torch  # type: ignore

torch.set_float32_matmul_precision("high")
torch.set_default_dtype(torch.float32)


logger = logging.getLogger(__name__)


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
    pass  # type: ignore
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

from pmarlo.ml.deeptica.whitening import apply_output_transform

from .losses import VAMP2Loss


def _resolve_activation_module(name: str):
    import torch.nn as _nn  # type: ignore

    key = (name or "").strip().lower()
    if key in {"gelu", "gaussian"}:
        return _nn.GELU()
    if key in {"relu", "relu+"}:
        return _nn.ReLU()
    if key in {"elu"}:
        return _nn.ELU()
    if key in {"selu"}:
        return _nn.SELU()
    if key in {"leaky_relu", "lrelu"}:
        return _nn.LeakyReLU()
    return _nn.Tanh()


def _normalize_hidden_dropout(spec: Any, num_hidden: int) -> List[float]:
    """Expand a dropout specification to match the number of hidden transitions."""

    if num_hidden <= 0:
        return []

    values: List[float]
    if spec is None:
        values = [0.0] * num_hidden
    elif isinstance(spec, (int, float)) and not isinstance(spec, bool):
        values = [float(spec)] * num_hidden
    elif isinstance(spec, str):
        try:
            scalar = float(spec)
        except ValueError:
            scalar = 0.0
        values = [scalar] * num_hidden
    else:
        values = []
        if isinstance(spec, Iterable) and not isinstance(spec, (bytes, str)):
            for item in spec:
                try:
                    values.append(float(item))
                except Exception:
                    values.append(0.0)
        else:
            try:
                scalar = float(spec)
            except Exception:
                scalar = 0.0
            values = [scalar] * num_hidden

    if not values:
        values = [0.0] * num_hidden

    if len(values) < num_hidden:
        last = values[-1]
        values = values + [last] * (num_hidden - len(values))
    elif len(values) > num_hidden:
        values = values[:num_hidden]

    return [float(max(0.0, min(1.0, v))) for v in values]


def _override_core_mlp(
    core,
    layers,
    activation_name: str,
    linear_head: bool,
    *,
    hidden_dropout: Any = None,
    layer_norm_hidden: bool = False,
) -> None:
    """Override core MLP configuration with custom activations/dropout."""

    if linear_head or len(layers) <= 2:
        return
    try:
        import torch.nn as _nn  # type: ignore
    except Exception:
        return

    hidden_transitions = max(0, len(layers) - 2)
    dropout_values = _normalize_hidden_dropout(hidden_dropout, hidden_transitions)

    modules: list[_nn.Module] = []
    for idx in range(len(layers) - 1):
        in_features = int(layers[idx])
        out_features = int(layers[idx + 1])
        modules.append(_nn.Linear(in_features, out_features))
        if idx < len(layers) - 2:
            if layer_norm_hidden:
                modules.append(_nn.LayerNorm(out_features))
            modules.append(_resolve_activation_module(activation_name))
            drop_p = dropout_values[idx] if idx < len(dropout_values) else 0.0
            if drop_p > 0.0:
                modules.append(_nn.Dropout(p=float(drop_p)))

    if modules:
        core.nn = _nn.Sequential(*modules)  # type: ignore[attr-defined]


def _apply_output_whitening(
    net, Z, idx_tau, *, apply: bool = False, eig_floor: float = 1e-4
):
    import torch

    tensor = torch.as_tensor(Z, dtype=torch.float32)
    with torch.no_grad():
        outputs = net(tensor)
        if isinstance(outputs, torch.Tensor):
            outputs = outputs.detach().cpu().numpy()
    if outputs is None or outputs.size == 0:
        info: dict[str, Any] = {
            "output_variance": [],
            "var_zt": [],
            "cond_c00": None,
            "cond_ctt": None,
            "mean": [],
            "transform": [],
            "transform_applied": bool(apply),
        }
        return net, info

    mean = np.mean(outputs, axis=0)
    centered = outputs - mean
    n = max(1, centered.shape[0] - 1)
    C0 = (centered.T @ centered) / float(n)

    def _regularize(mat: np.ndarray) -> np.ndarray:
        sym = 0.5 * (mat + mat.T)
        dim = sym.shape[0]
        eye = np.eye(dim, dtype=np.float64)
        trace = float(np.trace(sym))
        trace = max(trace, 1e-12)
        mu = trace / float(max(1, dim))
        ridge = mu * 1e-5
        alpha = 0.02
        return (1.0 - alpha) * sym + (alpha * mu + ridge) * eye

    C0_reg = _regularize(C0)
    eigvals, eigvecs = np.linalg.eigh(C0_reg)
    eigvals = np.clip(eigvals, max(eig_floor, 1e-8), None)
    inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
    output_var = centered.var(axis=0, ddof=1).astype(float).tolist()
    cond_c00 = float(eigvals.max() / eigvals.min())

    var_zt = None
    cond_ctt = None
    if idx_tau is not None and len(Z) > 0:
        tau_tensor = torch.as_tensor(Z[idx_tau], dtype=torch.float32)
        with torch.no_grad():
            base = net if not isinstance(net, _WhitenWrapper) else net.inner
            tau_out = base(tau_tensor)
            if isinstance(tau_out, torch.Tensor):
                tau_out = tau_out.detach().cpu().numpy()
        tau_center = tau_out - mean
        var_zt = tau_center.var(axis=0, ddof=1).astype(float).tolist()
        n_tau = max(1, tau_center.shape[0] - 1)
        Ct = (tau_center.T @ tau_center) / float(n_tau)
        Ct_reg = _regularize(Ct)
        eig_ct = np.linalg.eigvalsh(Ct_reg)
        eig_ct = np.clip(eig_ct, max(eig_floor, 1e-8), None)
        cond_ctt = float(eig_ct.max() / eig_ct.min())

    if var_zt is None:
        var_zt = output_var

    transform = inv_sqrt if apply else np.eye(inv_sqrt.shape[0], dtype=np.float64)
    wrapped = _WhitenWrapper(net, mean, transform) if apply else net

    info = {
        "output_variance": output_var,
        "var_zt": var_zt,
        "cond_c00": cond_c00,
        "cond_ctt": cond_ctt,
        "mean": mean.astype(float).tolist(),
        "transform": inv_sqrt.astype(float).tolist(),
        "transform_applied": bool(apply),
    }
    return wrapped, info


# Provide a module-level whitening wrapper so helper functions can reference it
try:
    import torch.nn as _nn  # type: ignore
except Exception:  # pragma: no cover - optional in environments without torch
    _nn = None  # type: ignore

if _nn is not None:

    class _WhitenWrapper(_nn.Module):  # type: ignore[misc]
        def __init__(
            self,
            inner,
            mean: np.ndarray | torch.Tensor,
            transform: np.ndarray | torch.Tensor,
        ):
            super().__init__()
            self.inner = inner
            # Register buffers to move with the module's device
            self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
            self.register_buffer(
                "transform", torch.as_tensor(transform, dtype=torch.float32)
            )

        def forward(self, x):  # type: ignore[override]
            y = self.inner(x)
            y = y - self.mean
            return torch.matmul(y, self.transform.T)


@dataclass(frozen=True)
class DeepTICAConfig:
    lag: int
    n_out: int = 2
    hidden: Tuple[int, ...] = (32, 16)
    activation: str = "gelu"
    learning_rate: float = 3e-4
    batch_size: int = 1024
    max_epochs: int = 200
    early_stopping: int = 25
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
    dropout: float = 0.0
    dropout_input: Optional[float] = None
    hidden_dropout: Tuple[float, ...] = ()
    layer_norm_in: bool = False
    layer_norm_hidden: bool = False
    linear_head: bool = False
    # Dataset splitting/loader control
    val_split: str = "by_shard"  # "by_shard" | "random"
    batches_per_epoch: int = 200
    gradient_clip_val: float = 1.0
    gradient_clip_algorithm: str = "norm"
    tau_schedule: Tuple[int, ...] = ()
    val_tau: Optional[int] = None
    epochs_per_tau: int = 15
    vamp_eps: float = 1e-3
    vamp_eps_abs: float = 1e-6
    vamp_alpha: float = 0.15
    vamp_cond_reg: float = 1e-4
    grad_norm_warn: Optional[float] = None
    variance_warn_threshold: float = 1e-6
    mean_warn_threshold: float = 5.0

    @classmethod
    def small_data(
        cls,
        *,
        lag: int,
        n_out: int = 2,
        hidden: Tuple[int, ...] | None = None,
        dropout_input: Optional[float] = None,
        hidden_dropout: Iterable[float] | None = None,
        **overrides: Any,
    ) -> "DeepTICAConfig":
        """Preset tuned for scarce data with stronger regularization.

        Parameters
        ----------
        lag
            Required lag time for the curriculum.
        n_out
            Number of collective variables to learn.
        hidden
            Optional explicit hidden layer sizes. Defaults to a single modest layer.
        dropout_input
            Override the preset input dropout rate.
        hidden_dropout
            Override the hidden-layer dropout schedule.
        overrides
            Additional configuration overrides forwarded to ``DeepTICAConfig``.
        """

        base_hidden = hidden if hidden is not None else (32,)
        drop_in = 0.15 if dropout_input is None else float(dropout_input)
        if hidden_dropout is None:
            drop_hidden_seq = tuple(0.15 for _ in range(max(0, len(base_hidden))))
        else:
            drop_hidden_seq = tuple(float(v) for v in hidden_dropout)
        defaults = dict(
            lag=int(lag),
            n_out=int(n_out),
            hidden=tuple(int(h) for h in base_hidden),
            dropout_input=float(max(0.0, min(1.0, drop_in))),
            hidden_dropout=tuple(float(max(0.0, min(1.0, v))) for v in drop_hidden_seq),
            layer_norm_in=True,
            layer_norm_hidden=True,
        )
        defaults.update(overrides)
        # Type-safe construction by unpacking the dictionary
        from typing import cast

        return cls(**cast("dict[str, Any]", defaults))


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
        outputs = np.asarray(y, dtype=np.float64)
        history = getattr(self, "training_history", {}) or {}
        mean = history.get("output_mean") if isinstance(history, dict) else None
        transform = (
            history.get("output_transform") if isinstance(history, dict) else None
        )
        applied_flag = (
            history.get("output_transform_applied")
            if isinstance(history, dict)
            else None
        )
        if mean is not None and transform is not None:
            try:
                outputs = apply_output_transform(outputs, mean, transform, applied_flag)
            except Exception:
                # Best-effort: fall back to raw outputs if metadata is inconsistent
                pass
        return outputs

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
        hidden_cfg = tuple(int(h) for h in getattr(cfg, "hidden", ()) or ())
        if bool(getattr(cfg, "linear_head", False)):
            hidden_layers: tuple[int, ...] = ()
        else:
            hidden_layers = hidden_cfg if hidden_cfg else (32, 16)
        layers = [in_dim, *hidden_layers, int(cfg.n_out)]
        activation_name = (
            str(getattr(cfg, "activation", "gelu")).lower().strip() or "gelu"
        )
        hidden_dropout_cfg: Any = getattr(cfg, "hidden_dropout", ())
        layer_norm_hidden = bool(getattr(cfg, "layer_norm_hidden", False))
        try:
            core = DeepTICA(
                layers=layers,
                n_cvs=int(cfg.n_out),
                activation=activation_name,
                options={"norm_in": False},
            )
        except TypeError:
            core = DeepTICA(
                layers=layers,
                n_cvs=int(cfg.n_out),
                options={"norm_in": False},
            )
            _override_core_mlp(
                core,
                layers,
                activation_name,
                bool(getattr(cfg, "linear_head", False)),
                hidden_dropout=hidden_dropout_cfg,
                layer_norm_hidden=layer_norm_hidden,
            )
        else:
            _override_core_mlp(
                core,
                layers,
                activation_name,
                bool(getattr(cfg, "linear_head", False)),
                hidden_dropout=hidden_dropout_cfg,
                layer_norm_hidden=layer_norm_hidden,
            )
        import torch.nn as _nn  # type: ignore

        def _strip_batch_norm(module: _nn.Module) -> None:
            for name, child in module.named_children():
                if isinstance(child, _nn.modules.batchnorm._BatchNorm):
                    setattr(module, name, _nn.Identity())
                else:
                    _strip_batch_norm(child)

        class _PrePostWrapper(_nn.Module):  # type: ignore[misc]
            def __init__(self, inner, in_features: int, *, ln_in: bool, p_drop: float):
                super().__init__()
                self.ln = _nn.LayerNorm(in_features) if ln_in else _nn.Identity()
                p = float(max(0.0, min(1.0, p_drop)))
                self.drop_in = _nn.Dropout(p=p) if p > 0 else _nn.Identity()
                self.inner = inner
                self.drop_out = _nn.Identity()

            def forward(self, x):  # type: ignore[override]
                x = self.ln(x)
                x = self.drop_in(x)
                return self.inner(x)

        _strip_batch_norm(core)
        dropout_in = getattr(cfg, "dropout_input", None)
        if dropout_in is None:
            dropout_in = getattr(cfg, "dropout", 0.1)
        # Ensure dropout_in is not None before converting to float
        if dropout_in is None:
            dropout_in = 0.1
        net = _PrePostWrapper(
            core,
            in_dim,
            ln_in=bool(getattr(cfg, "layer_norm_in", True)),
            p_drop=float(dropout_in),
        )
        state = torch.load(path.with_suffix(".pt"), map_location="cpu")
        net.load_state_dict(state["state_dict"])  # type: ignore[index]
        net.eval()
        history: dict | None = None
        history_path = path.with_suffix(".history.json")
        if history_path.exists():
            try:
                history = json.loads(history_path.read_text(encoding="utf-8"))
            except Exception:
                history = None
        return cls(cfg, scaler, net, training_history=history)

    def to_torchscript(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.net.eval()
        # Trace with single precision (typical for inference)
        example = torch.zeros(1, int(self.scaler.mean_.shape[0]), dtype=torch.float32)
        # Work around LightningModule property access during JIT introspection
        try:

            def _mark_scripting_safe(mod):
                try:
                    if hasattr(mod, "_jit_is_scripting"):
                        setattr(mod, "_jit_is_scripting", True)
                except Exception:
                    pass
                try:
                    for _name, _child in getattr(mod, "named_modules", lambda: [])():
                        try:
                            if hasattr(_child, "_jit_is_scripting"):
                                setattr(_child, "_jit_is_scripting", True)
                        except Exception:
                            continue
                except Exception:
                    pass

            _mark_scripting_safe(self.net)
            base = getattr(self.net, "inner", None)
            if base is not None:
                _mark_scripting_safe(base)
        except Exception:
            pass
        ts = torch.jit.trace(self.net.to(torch.float32), example)
        out = path.with_suffix(".ts")
        try:
            ts.save(str(out))
        except Exception:
            # Fallback to torch.jit.save for broader compatibility
            torch.jit.save(ts, str(out))
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
    hidden_cfg = tuple(int(h) for h in getattr(cfg, "hidden", ()) or ())
    if bool(getattr(cfg, "linear_head", False)):
        hidden_layers: tuple[int, ...] = ()
    else:
        hidden_layers = hidden_cfg if hidden_cfg else (32, 16)
    layers = [in_dim, *hidden_layers, int(cfg.n_out)]
    activation_name = str(getattr(cfg, "activation", "gelu")).lower().strip() or "gelu"
    hidden_dropout_cfg: Any = getattr(cfg, "hidden_dropout", ())
    layer_norm_hidden = bool(getattr(cfg, "layer_norm_hidden", False))
    try:
        core = DeepTICA(
            layers=layers,
            n_cvs=int(cfg.n_out),
            activation=activation_name,
            options={"norm_in": False},
        )
    except TypeError:
        core = DeepTICA(
            layers=layers,
            n_cvs=int(cfg.n_out),
            options={"norm_in": False},
        )
        _override_core_mlp(
            core,
            layers,
            activation_name,
            bool(getattr(cfg, "linear_head", False)),
            hidden_dropout=hidden_dropout_cfg,
            layer_norm_hidden=layer_norm_hidden,
        )
    else:
        _override_core_mlp(
            core,
            layers,
            activation_name,
            bool(getattr(cfg, "linear_head", False)),
            hidden_dropout=hidden_dropout_cfg,
            layer_norm_hidden=layer_norm_hidden,
        )
    # Wrap with input LayerNorm and light dropout for stability on tiny nets
    import torch.nn as _nn  # type: ignore

    def _strip_batch_norm(module: _nn.Module) -> None:
        for name, child in module.named_children():
            if isinstance(child, _nn.modules.batchnorm._BatchNorm):
                setattr(module, name, _nn.Identity())
            else:
                _strip_batch_norm(child)

    class _PrePostWrapper(_nn.Module):  # type: ignore[misc]
        def __init__(self, inner, in_features: int, *, ln_in: bool, p_drop: float):
            super().__init__()
            self.ln = _nn.LayerNorm(in_features) if ln_in else _nn.Identity()
            p = float(max(0.0, min(1.0, p_drop)))
            self.drop_in = _nn.Dropout(p=p) if p > 0 else _nn.Identity()
            self.inner = inner
            self.drop_out = _nn.Identity()

        def forward(self, x):  # type: ignore[override]
            x = self.ln(x)
            x = self.drop_in(x)
            return self.inner(x)

    _strip_batch_norm(core)
    dropout_in = getattr(cfg, "dropout_input", None)
    if dropout_in is None:
        dropout_in = getattr(cfg, "dropout", 0.0)
    # Ensure dropout_in is not None before converting to float
    if dropout_in is None:
        dropout_in = 0.0
    dropout_in = float(max(0.0, min(1.0, float(dropout_in))))
    net = _PrePostWrapper(
        core,
        in_dim,
        ln_in=bool(getattr(cfg, "layer_norm_in", False)),
        p_drop=dropout_in,
    )
    torch.manual_seed(int(cfg.seed))

    tau_schedule = tuple(
        int(x) for x in (getattr(cfg, "tau_schedule", ()) or ()) if int(x) > 0
    )
    if not tau_schedule:
        tau_schedule = (int(cfg.lag),)

    idx_t, idx_tlag = pairs

    # Validate or construct per-shard pairs to ensure x_t != x_{t+tau}
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

    if len(tau_schedule) > 1:
        idx_parts: List[np.ndarray] = []
        j_parts: List[np.ndarray] = []
        for tau_val in tau_schedule:
            i_tau, j_tau = _build_uniform_pairs_per_shard(X_list, int(tau_val))
            if i_tau.size and j_tau.size:
                idx_parts.append(i_tau)
                j_parts.append(j_tau)
        if idx_parts:
            idx_t = np.concatenate(idx_parts).astype(np.int64, copy=False)
            idx_tlag = np.concatenate(j_parts).astype(np.int64, copy=False)
        else:
            idx_t = np.asarray([], dtype=np.int64)
            idx_tlag = np.asarray([], dtype=np.int64)
    else:
        if _needs_repair(idx_t, idx_tlag):
            idx_t, idx_tlag = _build_uniform_pairs_per_shard(
                X_list, int(tau_schedule[0])
            )

    idx_t = np.asarray(idx_t, dtype=np.int64)
    idx_tlag = np.asarray(idx_tlag, dtype=np.int64)

    shard_lengths = [int(np.asarray(b).shape[0]) for b in X_list]
    max_tau = int(max(tau_schedule)) if tau_schedule else int(cfg.lag)
    min_required = max_tau + 1
    short_shards = [
        idx for idx, length in enumerate(shard_lengths) if length < min_required
    ]
    total_possible = sum(max(0, length - max_tau) for length in shard_lengths)
    usable_pairs = int(min(idx_t.shape[0], idx_tlag.shape[0]))
    coverage = float(usable_pairs / total_possible) if total_possible else 0.0
    offsets = np.cumsum([0, *shard_lengths])
    pairs_by_shard = []
    for start, end in zip(offsets[:-1], offsets[1:]):
        mask = (idx_t >= start) & (idx_t < end)
        pairs_by_shard.append(int(np.count_nonzero(mask)))

    pair_diagnostics = {
        "usable_pairs": usable_pairs,
        "pairs_by_shard": pairs_by_shard,
        "short_shards": short_shards,
        "pair_coverage": coverage,
        "total_possible_pairs": int(total_possible),
        "lag_used": int(max_tau),
    }

    if short_shards:
        logger.warning(
            "%d/%d shards too short for lag %d",
            len(short_shards),
            len(shard_lengths),
            int(max_tau),
        )
    if usable_pairs == 0:
        logger.warning(
            "No usable lagged pairs remain after constructing curriculum with lag %d",
            int(max_tau),
        )
    elif coverage < 0.5:
        logger.warning(
            "Lagged pair coverage low: %.1f%% (%d/%d possible pairs)",
            coverage * 100.0,
            usable_pairs,
            int(total_possible),
        )
    else:
        logger.info(
            "Lagged pair diagnostics: usable=%d coverage=%.1f%% short_shards=%s",
            usable_pairs,
            coverage * 100.0,
            short_shards,
        )

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

    # Build time-lagged dataset for training
    ds = None
    try:
        # Normalize index arrays and construct default weights (ones) when not provided
        if idx_t is None or idx_tlag is None or (len(idx_t) == 0 or len(idx_tlag) == 0):
            n = int(Z.shape[0])
            L = int(tau_schedule[-1])
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
            if weights_arr.size == 1 and int(idx_t.shape[0]) > 1:
                weights_arr = np.full(
                    (int(idx_t.shape[0]),),
                    float(weights_arr[0]),
                    dtype=np.float32,
                )
            elif int(idx_t.shape[0]) != int(weights_arr.shape[0]):
                raise ValueError(
                    "weights must have the same length as the number of lagged pairs"
                )

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
            self.val_scores: list[float] = []

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
                for key in ("val_score", "val_vamp2"):
                    if key in metrics:
                        score = float(metrics[key])
                        self.val_scores.append(score)
                        break
            except Exception:
                pass

    if lightning_available and Trainer is not None:
        callbacks = []
        hist_cb = _LossHistory()
        callbacks.append(hist_cb)
        try:
            if EarlyStoppingCls is not None:
                has_val = dm is not None or val_loader is not None
                monitor_metric = "val_score" if has_val else "train_loss"
                mode = "max" if has_val else "min"
                patience_cfg = int(max(1, getattr(cfg, "early_stopping", 25)))
                # Construct with compatibility across lightning versions
                try:
                    es = EarlyStoppingCls(
                        monitor=monitor_metric,
                        patience=int(patience_cfg),
                        mode=mode,
                        min_delta=float(1e-6),
                        stopping_threshold=None,
                        check_finite=True,
                    )
                except TypeError:
                    es = EarlyStoppingCls(
                        monitor=monitor_metric,
                        patience=int(patience_cfg),
                        mode=mode,
                        min_delta=float(1e-6),
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
                filename_pattern = (
                    "epoch={epoch:03d}-step={step}-score={val_score:.5f}"
                    if dm is not None or val_loader is not None
                    else "epoch={epoch:03d}-step={step}-loss={train_loss:.5f}"
                )
                ckpt_callback = ModelCheckpointCls(
                    dirpath=str(ckpt_dir),
                    filename=filename_pattern,
                    monitor=(
                        "val_score"
                        if dm is not None or val_loader is not None
                        else "train_loss"
                    ),
                    mode="max" if dm is not None or val_loader is not None else "min",
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

            vamp_kwargs = {
                "eps": float(max(1e-9, getattr(cfg, "vamp_eps", 1e-3))),
                "eps_abs": float(max(0.0, getattr(cfg, "vamp_eps_abs", 1e-6))),
                "alpha": float(min(max(getattr(cfg, "vamp_alpha", 0.15), 0.0), 1.0)),
                "cond_reg": float(max(0.0, getattr(cfg, "vamp_cond_reg", 1e-4))),
            }

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
                    grad_norm_warn: float | None = None,
                    variance_warn_threshold: float = 1e-6,
                    mean_warn_threshold: float = 5.0,
                ):
                    super().__init__()
                    self.inner = inner
                    # Type-safe construction of VAMP2Loss with explicit kwargs
                    from typing import cast

                    self.vamp_loss = VAMP2Loss(**cast("dict[str, Any]", vamp_kwargs))
                    self._train_loss_accum: list[float] = []
                    self._val_loss_accum: list[float] = []
                    self._val_score_accum: list[float] = []
                    self._grad_norm_accum: list[float] = []
                    self._val_var_z0_accum: list[list[float]] = []
                    self._val_var_zt_accum: list[list[float]] = []
                    self._val_mean_z0_accum: list[list[float]] = []
                    self._val_mean_zt_accum: list[list[float]] = []
                    self._cond_c0_accum: list[float] = []
                    self._cond_ctt_accum: list[float] = []
                    self._c0_eig_min_accum: list[float] = []
                    self._c0_eig_max_accum: list[float] = []
                    self._ctt_eig_min_accum: list[float] = []
                    self._ctt_eig_max_accum: list[float] = []
                    self.train_loss_curve: list[float] = []
                    self.val_loss_curve: list[float] = []
                    self.val_score_curve: list[float] = []
                    self.var_z0_curve: list[list[float]] = []
                    self.var_zt_curve: list[list[float]] = []
                    self.var_z0_curve_components: list[list[float]] = []
                    self.var_zt_curve_components: list[list[float]] = []
                    self.mean_z0_curve: list[list[float]] = []
                    self.mean_zt_curve: list[list[float]] = []
                    self.cond_c0_curve: list[float] = []
                    self.cond_ctt_curve: list[float] = []
                    self.grad_norm_curve: list[float] = []
                    self.c0_eig_min_curve: list[float] = []
                    self.c0_eig_max_curve: list[float] = []
                    self.ctt_eig_min_curve: list[float] = []
                    self.ctt_eig_max_curve: list[float] = []
                    self.grad_norm_warn = (
                        float(grad_norm_warn) if grad_norm_warn is not None else None
                    )
                    self.variance_warn_threshold = float(variance_warn_threshold)
                    self.mean_warn_threshold = float(mean_warn_threshold)
                    self._last_grad_warning_step: int | None = None
                    self._grad_warning_pending = False
                    self._last_grad_norm: float | None = None
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
                        self.history_dir: Path | None = (
                            Path(history_dir) if history_dir is not None else None
                        )
                        if self.history_dir is not None:
                            self.history_dir.mkdir(parents=True, exist_ok=True)
                            self.history_file: Path | None = (
                                self.history_dir / "history.jsonl"
                            )
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
                    y_t = self.inner(b["data"])  # type: ignore[index]
                    y_tau = self.inner(b["data_lag"])  # type: ignore[index]
                    loss, score = self.vamp_loss(y_t, y_tau, weights=b.get("weights"))
                    self._train_loss_accum.append(float(loss.detach().cpu().item()))
                    self.log(
                        "train_loss", loss, on_step=False, on_epoch=True, prog_bar=True
                    )
                    self.log(
                        "train_vamp2",
                        score,
                        on_step=False,
                        on_epoch=True,
                        prog_bar=False,
                    )
                    if self._grad_warning_pending and self._last_grad_norm is not None:
                        try:
                            self.log(
                                "grad_norm_exceeded",
                                torch.tensor(
                                    float(self._last_grad_norm),
                                    device=(
                                        loss.device if hasattr(loss, "device") else None
                                    ),
                                    dtype=torch.float32,
                                ),
                                prog_bar=False,
                                logger=True,
                            )
                        except Exception:
                            pass
                        self._grad_warning_pending = False
                    return loss

                def on_after_backward(self):  # type: ignore[override]
                    grad_sq = []
                    for param in self.parameters():
                        if param.grad is not None:
                            grad_sq.append(
                                torch.sum(param.grad.detach().to(torch.float64) ** 2)
                            )
                    if grad_sq:
                        grad_norm = float(
                            torch.sqrt(torch.stack(grad_sq).sum()).cpu().item()
                        )
                    else:
                        grad_norm = 0.0
                    self._grad_norm_accum.append(float(grad_norm))
                    self._last_grad_norm = float(grad_norm)
                    if self.grad_norm_warn is not None and float(grad_norm) > float(
                        self.grad_norm_warn
                    ):
                        step_idx = None
                        try:
                            step_idx = int(getattr(self.trainer, "global_step", 0))
                        except Exception:
                            step_idx = None
                        if step_idx is not None:
                            if self._last_grad_warning_step != step_idx:
                                logger.warning(
                                    "Gradient norm %.3f exceeded warning threshold %.3f at step %d",
                                    float(grad_norm),
                                    float(self.grad_norm_warn),
                                    int(step_idx),
                                )
                                self._last_grad_warning_step = step_idx
                        else:
                            logger.warning(
                                "Gradient norm %.3f exceeded warning threshold %.3f",
                                float(grad_norm),
                                float(self.grad_norm_warn),
                            )
                        self._grad_warning_pending = True

                def validation_step(self, batch, batch_idx):  # type: ignore[override]
                    b = self._norm_batch(batch)
                    y_t = self.inner(b["data"])  # type: ignore[index]
                    y_tau = self.inner(b["data_lag"])  # type: ignore[index]
                    loss, score = self.vamp_loss(y_t, y_tau, weights=b.get("weights"))
                    self._val_loss_accum.append(float(loss.detach().cpu().item()))
                    self._val_score_accum.append(float(score.detach().cpu().item()))
                    self.log(
                        "val_loss", loss, on_step=False, on_epoch=True, prog_bar=True
                    )
                    # Diagnostics: generalized eigenvalues, per-CV autocorr, whitening norm
                    try:
                        with torch.no_grad():
                            y_t_eval = y_t.detach()
                            y_tau_eval = y_tau.detach()

                            def _regularize_cov(cov: torch.Tensor) -> torch.Tensor:
                                cov_sym = (cov + cov.transpose(-1, -2)) * 0.5
                                dim = cov_sym.shape[-1]
                                eye = torch.eye(
                                    dim, device=cov_sym.device, dtype=cov_sym.dtype
                                )
                                trace_floor = torch.tensor(
                                    1e-12, dtype=cov_sym.dtype, device=cov_sym.device
                                )
                                tr = torch.clamp(torch.trace(cov_sym), min=trace_floor)
                                mu = tr / float(max(1, dim))
                                ridge = mu * float(self.vamp_loss.eps)
                                alpha = 0.02
                                return (1.0 - alpha) * cov_sym + (
                                    alpha * mu + ridge
                                ) * eye

                            y_t_c = y_t_eval - torch.mean(y_t_eval, dim=0, keepdim=True)
                            y_tau_c = y_tau_eval - torch.mean(
                                y_tau_eval, dim=0, keepdim=True
                            )
                            n = max(1, y_t_eval.shape[0] - 1)
                            C0 = (y_t_c.T @ y_t_c) / float(n)
                            Ctt = (y_tau_c.T @ y_tau_c) / float(n)
                            Ctau = (y_t_c.T @ y_tau_c) / float(n)

                            C0_reg = _regularize_cov(C0)
                            Ctt_reg = _regularize_cov(Ctt)
                            evals, evecs = torch.linalg.eigh(C0_reg)
                            eps_floor = torch.tensor(
                                1e-12, dtype=evals.dtype, device=evals.device
                            )
                            inv_sqrt = torch.diag(
                                torch.rsqrt(torch.clamp(evals, min=eps_floor))
                            )
                            W = evecs @ inv_sqrt @ evecs.T
                            M = W @ Ctau @ W.T
                            Ms = (M + M.T) * 0.5
                            vals = torch.linalg.eigvalsh(Ms)
                            vals, _ = torch.sort(vals, descending=True)
                            k = min(int(y_t_eval.shape[1]), 4)
                            for i in range(k):
                                self.log(
                                    f"val_eig_{i}",
                                    vals[i].float(),
                                    on_step=False,
                                    on_epoch=True,
                                    prog_bar=False,
                                )
                            var_z0 = torch.var(y_t, dim=0, unbiased=True)
                            var_zt = torch.var(y_tau, dim=0, unbiased=True)
                            self._val_var_z0_accum.append(
                                var_z0.detach().cpu().tolist()
                            )
                            self._val_var_zt_accum.append(
                                var_zt.detach().cpu().tolist()
                            )
                            mean_z0 = torch.mean(y_t, dim=0)
                            mean_zt = torch.mean(y_tau, dim=0)
                            self._val_mean_z0_accum.append(
                                mean_z0.detach().cpu().tolist()
                            )
                            self._val_mean_zt_accum.append(
                                mean_zt.detach().cpu().tolist()
                            )
                            evals_c0 = torch.clamp(evals, min=eps_floor)
                            cond_c0 = float(
                                (evals_c0.max() / evals_c0.min()).detach().cpu().item()
                            )
                            evals_ctt = torch.linalg.eigvalsh(Ctt_reg)
                            evals_ctt = torch.clamp(evals_ctt, min=eps_floor)
                            cond_ctt = float(
                                (evals_ctt.max() / evals_ctt.min())
                                .detach()
                                .cpu()
                                .item()
                            )
                            self._c0_eig_min_accum.append(
                                float(evals_c0.min().detach().cpu().item())
                            )
                            self._c0_eig_max_accum.append(
                                float(evals_c0.max().detach().cpu().item())
                            )
                            self._ctt_eig_min_accum.append(
                                float(evals_ctt.min().detach().cpu().item())
                            )
                            self._ctt_eig_max_accum.append(
                                float(evals_ctt.max().detach().cpu().item())
                            )
                            self._cond_c0_accum.append(cond_c0)
                            self._cond_ctt_accum.append(cond_ctt)
                            var_t = torch.diag(C0_reg)
                            var_tau = torch.diag(Ctt_reg)
                            corr = torch.diag(Ctau) / torch.sqrt(
                                torch.clamp(var_t * var_tau, min=eps_floor)
                            )
                            for i in range(min(int(corr.shape[0]), 4)):
                                self.log(
                                    f"val_corr_{i}",
                                    corr[i].float(),
                                    on_step=False,
                                    on_epoch=True,
                                    prog_bar=False,
                                )
                            whiten_norm = torch.linalg.norm(W, ord="fro")
                            self.log(
                                "val_whiten_norm",
                                whiten_norm.float(),
                                on_step=False,
                                on_epoch=True,
                                prog_bar=False,
                            )
                            if int(batch_idx) == 0:
                                try:
                                    if getattr(self, "history_file", None) is not None:
                                        rec = {
                                            "epoch": int(self.current_epoch),
                                            "val_loss": float(
                                                loss.detach().cpu().item()
                                            ),
                                            "val_score": float(
                                                score.detach().cpu().item()
                                            ),
                                            "val_vamp2": float(
                                                score.detach().cpu().item()
                                            ),
                                            "val_eigs": [
                                                float(vals[i].detach().cpu().item())
                                                for i in range(k)
                                            ],
                                            "val_corr": [
                                                float(corr[i].detach().cpu().item())
                                                for i in range(
                                                    min(int(corr.shape[0]), 4)
                                                )
                                            ],
                                            "val_whiten_norm": float(
                                                whiten_norm.detach().cpu().item()
                                            ),
                                            "var_z0": [
                                                float(x)
                                                for x in var_z0.detach().cpu().tolist()
                                            ],
                                            "var_zt": [
                                                float(x)
                                                for x in var_zt.detach().cpu().tolist()
                                            ],
                                            "cond_C00": float(cond_c0),
                                            "cond_Ctt": float(cond_ctt),
                                        }
                                        with open(
                                            self.history_file, "a", encoding="utf-8"
                                        ) as fh:
                                            fh.write(
                                                json.dumps(rec, sort_keys=True) + "\n"
                                            )
                                except Exception:
                                    pass
                    except Exception:
                        # Diagnostics are best-effort; do not fail validation if they error
                        pass
                    return loss

                def on_train_epoch_start(self):  # type: ignore[override]
                    self._train_loss_accum.clear()
                    self._grad_norm_accum.clear()
                    self._grad_warning_pending = False
                    self._last_grad_norm = None

                def on_train_epoch_end(self):  # type: ignore[override]
                    if self._train_loss_accum:
                        avg = float(
                            sum(self._train_loss_accum) / len(self._train_loss_accum)
                        )
                        self.train_loss_curve.append(avg)
                        self.log(
                            "train_loss_epoch",
                            torch.tensor(avg, device=self.device, dtype=torch.float32),
                            prog_bar=False,
                        )
                    if self._grad_norm_accum:
                        avg_grad = float(
                            sum(self._grad_norm_accum) / len(self._grad_norm_accum)
                        )
                        self.grad_norm_curve.append(avg_grad)
                        self.log(
                            "grad_norm_epoch",
                            torch.tensor(
                                avg_grad, device=self.device, dtype=torch.float32
                            ),
                            prog_bar=False,
                        )
                    self._train_loss_accum.clear()
                    self._grad_norm_accum.clear()

                def on_validation_epoch_start(self):  # type: ignore[override]
                    self._val_loss_accum.clear()
                    self._val_score_accum.clear()
                    self._val_var_z0_accum.clear()
                    self._val_var_zt_accum.clear()
                    self._val_mean_z0_accum.clear()
                    self._val_mean_zt_accum.clear()
                    self._cond_c0_accum.clear()
                    self._cond_ctt_accum.clear()
                    self._c0_eig_min_accum.clear()
                    self._c0_eig_max_accum.clear()
                    self._ctt_eig_min_accum.clear()
                    self._ctt_eig_max_accum.clear()

                def on_validation_epoch_end(self):  # type: ignore[override]
                    avg_loss = None
                    avg_score = None
                    if self._val_loss_accum:
                        avg_loss = float(
                            sum(self._val_loss_accum) / len(self._val_loss_accum)
                        )
                        self.val_loss_curve.append(avg_loss)
                        self.log(
                            "val_loss_epoch",
                            torch.tensor(
                                avg_loss, device=self.device, dtype=torch.float32
                            ),
                            prog_bar=False,
                        )
                    if self._val_score_accum:
                        avg_score = float(
                            sum(self._val_score_accum) / len(self._val_score_accum)
                        )
                        self.val_score_curve.append(avg_score)
                        score_tensor = torch.tensor(
                            avg_score, device=self.device, dtype=torch.float32
                        )
                        self.log("val_score", score_tensor, prog_bar=True)
                    if self._val_var_z0_accum:
                        arr = np.asarray(self._val_var_z0_accum, dtype=float)
                        avg_var_z0 = np.mean(arr, axis=0).tolist()
                        comp = [float(x) for x in avg_var_z0]
                        self.var_z0_curve.append(comp)
                        self.var_z0_curve_components.append(comp)
                        self.log(
                            "val_var_z0",
                            torch.tensor(
                                float(np.mean(avg_var_z0)),
                                device=self.device,
                                dtype=torch.float32,
                            ),
                            prog_bar=False,
                        )
                        for idx, value in enumerate(comp):
                            try:
                                self.log(
                                    f"val_var_z0_{idx}",
                                    torch.tensor(
                                        float(value),
                                        device=self.device,
                                        dtype=torch.float32,
                                    ),
                                    prog_bar=False,
                                )
                            except Exception:
                                continue
                        if comp and float(min(comp)) < self.variance_warn_threshold:
                            logger.warning(
                                "Validation variance for some CV dropped below %.2e (min %.2e)",
                                float(self.variance_warn_threshold),
                                float(min(comp)),
                            )
                    if self._val_var_zt_accum:
                        arr = np.asarray(self._val_var_zt_accum, dtype=float)
                        avg_var_zt = np.mean(arr, axis=0).tolist()
                        comp_tau = [float(x) for x in avg_var_zt]
                        self.var_zt_curve.append(comp_tau)
                        self.var_zt_curve_components.append(comp_tau)
                        self.log(
                            "val_var_zt",
                            torch.tensor(
                                float(np.mean(avg_var_zt)),
                                device=self.device,
                                dtype=torch.float32,
                            ),
                            prog_bar=False,
                        )
                        for idx, value in enumerate(comp_tau):
                            try:
                                self.log(
                                    f"val_var_zt_{idx}",
                                    torch.tensor(
                                        float(value),
                                        device=self.device,
                                        dtype=torch.float32,
                                    ),
                                    prog_bar=False,
                                )
                            except Exception:
                                continue
                        if (
                            comp_tau
                            and float(min(comp_tau)) < self.variance_warn_threshold
                        ):
                            logger.warning(
                                "Validation lagged variance for some CV dropped below %.2e (min %.2e)",
                                float(self.variance_warn_threshold),
                                float(min(comp_tau)),
                            )
                    if self._val_mean_z0_accum:
                        arr = np.asarray(self._val_mean_z0_accum, dtype=float)
                        avg_mean_z0 = np.mean(arr, axis=0).tolist()
                        comp_mean_z0 = [float(x) for x in avg_mean_z0]
                        self.mean_z0_curve.append(comp_mean_z0)
                        for idx, value in enumerate(comp_mean_z0):
                            try:
                                self.log(
                                    f"val_mean_z0_{idx}",
                                    torch.tensor(
                                        float(value),
                                        device=self.device,
                                        dtype=torch.float32,
                                    ),
                                    prog_bar=False,
                                )
                            except Exception:
                                continue
                        if comp_mean_z0:
                            drift = max(abs(v) for v in comp_mean_z0)
                            if drift > self.mean_warn_threshold:
                                logger.warning(
                                    "Validation CV mean drift %.3f exceeds threshold %.3f",
                                    float(drift),
                                    float(self.mean_warn_threshold),
                                )
                    if self._val_mean_zt_accum:
                        arr = np.asarray(self._val_mean_zt_accum, dtype=float)
                        avg_mean_zt = np.mean(arr, axis=0).tolist()
                        comp_mean_zt = [float(x) for x in avg_mean_zt]
                        self.mean_zt_curve.append(comp_mean_zt)
                        for idx, value in enumerate(comp_mean_zt):
                            try:
                                self.log(
                                    f"val_mean_zt_{idx}",
                                    torch.tensor(
                                        float(value),
                                        device=self.device,
                                        dtype=torch.float32,
                                    ),
                                    prog_bar=False,
                                )
                            except Exception:
                                continue
                        if comp_mean_zt:
                            drift = max(abs(v) for v in comp_mean_zt)
                            if drift > self.mean_warn_threshold:
                                logger.warning(
                                    "Validation lagged CV mean drift %.3f exceeds threshold %.3f",
                                    float(drift),
                                    float(self.mean_warn_threshold),
                                )
                    if self._cond_c0_accum:
                        avg_cond_c0 = float(
                            sum(self._cond_c0_accum) / len(self._cond_c0_accum)
                        )
                        self.cond_c0_curve.append(avg_cond_c0)
                        self.log(
                            "cond_C00",
                            torch.tensor(
                                avg_cond_c0, device=self.device, dtype=torch.float32
                            ),
                            prog_bar=False,
                        )
                    if self._cond_ctt_accum:
                        avg_cond_ctt = float(
                            sum(self._cond_ctt_accum) / len(self._cond_ctt_accum)
                        )
                        self.cond_ctt_curve.append(avg_cond_ctt)
                        self.log(
                            "cond_Ctt",
                            torch.tensor(
                                avg_cond_ctt, device=self.device, dtype=torch.float32
                            ),
                            prog_bar=False,
                        )
                    if self._c0_eig_min_accum:
                        avg_c0_min = float(
                            sum(self._c0_eig_min_accum) / len(self._c0_eig_min_accum)
                        )
                        self.c0_eig_min_curve.append(avg_c0_min)
                        self.log(
                            "c0_eig_min",
                            torch.tensor(
                                avg_c0_min, device=self.device, dtype=torch.float32
                            ),
                            prog_bar=False,
                        )
                    if self._c0_eig_max_accum:
                        avg_c0_max = float(
                            sum(self._c0_eig_max_accum) / len(self._c0_eig_max_accum)
                        )
                        self.c0_eig_max_curve.append(avg_c0_max)
                        self.log(
                            "c0_eig_max",
                            torch.tensor(
                                avg_c0_max, device=self.device, dtype=torch.float32
                            ),
                            prog_bar=False,
                        )
                    if self._ctt_eig_min_accum:
                        avg_ctt_min = float(
                            sum(self._ctt_eig_min_accum) / len(self._ctt_eig_min_accum)
                        )
                        self.ctt_eig_min_curve.append(avg_ctt_min)
                        self.log(
                            "ctt_eig_min",
                            torch.tensor(
                                avg_ctt_min, device=self.device, dtype=torch.float32
                            ),
                            prog_bar=False,
                        )
                    if self._ctt_eig_max_accum:
                        avg_ctt_max = float(
                            sum(self._ctt_eig_max_accum) / len(self._ctt_eig_max_accum)
                        )
                        self.ctt_eig_max_curve.append(avg_ctt_max)
                        self.log(
                            "ctt_eig_max",
                            torch.tensor(
                                avg_ctt_max, device=self.device, dtype=torch.float32
                            ),
                            prog_bar=False,
                        )
                    self._val_loss_accum.clear()
                    self._val_score_accum.clear()
                    self._val_var_z0_accum.clear()
                    self._val_var_zt_accum.clear()
                    self._val_mean_z0_accum.clear()
                    self._val_mean_zt_accum.clear()
                    self._cond_c0_accum.clear()
                    self._cond_ctt_accum.clear()
                    self._c0_eig_min_accum.clear()
                    self._c0_eig_max_accum.clear()
                    self._ctt_eig_min_accum.clear()
                    self._ctt_eig_max_accum.clear()

                def configure_optimizers(self):  # type: ignore[override]
                    # AdamW with mild weight decay for stability
                    weight_decay = float(self.hparams.weight_decay)
                    if weight_decay <= 0.0:
                        weight_decay = 1e-4
                    opt = torch.optim.AdamW(
                        self.parameters(),
                        lr=float(self.hparams.lr),
                        weight_decay=weight_decay,
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
                grad_norm_warn=(
                    float(getattr(cfg, "grad_norm_warn", 0.0))
                    if getattr(cfg, "grad_norm_warn", None) is not None
                    else None
                ),
                variance_warn_threshold=float(
                    getattr(cfg, "variance_warn_threshold", 1e-6)
                ),
                mean_warn_threshold=float(getattr(cfg, "mean_warn_threshold", 5.0)),
            )
        except Exception:
            # If Lightning is completely unavailable, fall back to model.fit (handled below)
            wrapped = net

        # Enforce minimum training duration to avoid early flat-zero stalls
        _max_epochs = int(getattr(cfg, "max_epochs", 200))
        _min_epochs = max(1, min(50, _max_epochs // 4))
        clip_val = float(max(0.0, getattr(cfg, "gradient_clip_val", 0.0)))
        clip_alg = str(getattr(cfg, "gradient_clip_algorithm", "norm"))
        trainer_kwargs = {
            "max_epochs": _max_epochs,
            "min_epochs": _min_epochs,
            "enable_progress_bar": _pb,
            "logger": loggers if loggers else False,
            "callbacks": callbacks,
            "deterministic": True,
            "log_every_n_steps": 1,
            "enable_checkpointing": True,
            "gradient_clip_val": clip_val,
            "gradient_clip_algorithm": clip_alg,
        }
        try:
            trainer = Trainer(**trainer_kwargs)
        except TypeError:
            trainer_kwargs.pop("gradient_clip_algorithm", None)
            trainer = Trainer(**trainer_kwargs)

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
                    lagtime=int(tau_schedule[-1]),
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
    net, whitening_info = _apply_output_whitening(net, Z, idx_tlag, apply=False)
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
    try:
        arr = np.asarray(Y1, dtype=np.float64)
        if arr.shape[0] > 1:
            var_arr = np.var(arr, axis=0, ddof=1)
        else:
            var_arr = np.var(arr, axis=0, ddof=0)
        output_variance = var_arr.astype(float).tolist()
        logger.info("DeepTICA output variance: %s", output_variance)
    except Exception:
        output_variance = None

    # Prefer losses collected during training if available; otherwise proxy objective
    train_curve: list[float] | None = None
    val_curve: list[float] | None = None
    score_curve: list[float] | None = None
    var_z0_curve: list[list[float]] | None = None
    var_zt_curve: list[list[float]] | None = None
    cond_c0_curve: list[float] | None = None
    cond_ctt_curve: list[float] | None = None
    grad_norm_curve: list[float] | None = None
    var_z0_components: list[list[float]] | None = None
    var_zt_components: list[list[float]] | None = None
    mean_z0_curve: list[list[float]] | None = None
    mean_zt_curve: list[list[float]] | None = None
    c0_eig_min_curve: list[float] | None = None
    c0_eig_max_curve: list[float] | None = None
    ctt_eig_min_curve: list[float] | None = None
    ctt_eig_max_curve: list[float] | None = None
    try:
        if lightning_available:
            if hasattr(wrapped, "train_loss_curve") and getattr(
                wrapped, "train_loss_curve"
            ):
                train_curve = [float(x) for x in getattr(wrapped, "train_loss_curve")]
            if hasattr(wrapped, "val_loss_curve") and getattr(
                wrapped, "val_loss_curve"
            ):
                val_curve = [float(x) for x in getattr(wrapped, "val_loss_curve")]
            if hasattr(wrapped, "val_score_curve") and getattr(
                wrapped, "val_score_curve"
            ):
                score_curve = [float(x) for x in getattr(wrapped, "val_score_curve")]
            if hasattr(wrapped, "var_z0_curve") and getattr(wrapped, "var_z0_curve"):
                var_z0_curve = [
                    [float(v) for v in arr] for arr in getattr(wrapped, "var_z0_curve")
                ]
            if hasattr(wrapped, "var_zt_curve") and getattr(wrapped, "var_zt_curve"):
                var_zt_curve = [
                    [float(v) for v in arr] for arr in getattr(wrapped, "var_zt_curve")
                ]
            if hasattr(wrapped, "var_z0_curve_components") and getattr(
                wrapped, "var_z0_curve_components"
            ):
                var_z0_components = [
                    [float(v) for v in arr]
                    for arr in getattr(wrapped, "var_z0_curve_components")
                ]
            if hasattr(wrapped, "var_zt_curve_components") and getattr(
                wrapped, "var_zt_curve_components"
            ):
                var_zt_components = [
                    [float(v) for v in arr]
                    for arr in getattr(wrapped, "var_zt_curve_components")
                ]
            if hasattr(wrapped, "mean_z0_curve") and getattr(wrapped, "mean_z0_curve"):
                mean_z0_curve = [
                    [float(v) for v in arr] for arr in getattr(wrapped, "mean_z0_curve")
                ]
            if hasattr(wrapped, "mean_zt_curve") and getattr(wrapped, "mean_zt_curve"):
                mean_zt_curve = [
                    [float(v) for v in arr] for arr in getattr(wrapped, "mean_zt_curve")
                ]
            if hasattr(wrapped, "cond_c0_curve") and getattr(wrapped, "cond_c0_curve"):
                cond_c0_curve = [float(x) for x in getattr(wrapped, "cond_c0_curve")]
            if hasattr(wrapped, "cond_ctt_curve") and getattr(
                wrapped, "cond_ctt_curve"
            ):
                cond_ctt_curve = [float(x) for x in getattr(wrapped, "cond_ctt_curve")]
            if hasattr(wrapped, "grad_norm_curve") and getattr(
                wrapped, "grad_norm_curve"
            ):
                grad_norm_curve = [
                    float(x) for x in getattr(wrapped, "grad_norm_curve")
                ]
            if hasattr(wrapped, "c0_eig_min_curve") and getattr(
                wrapped, "c0_eig_min_curve"
            ):
                c0_eig_min_curve = [
                    float(x) for x in getattr(wrapped, "c0_eig_min_curve")
                ]
            if hasattr(wrapped, "c0_eig_max_curve") and getattr(
                wrapped, "c0_eig_max_curve"
            ):
                c0_eig_max_curve = [
                    float(x) for x in getattr(wrapped, "c0_eig_max_curve")
                ]
            if hasattr(wrapped, "ctt_eig_min_curve") and getattr(
                wrapped, "ctt_eig_min_curve"
            ):
                ctt_eig_min_curve = [
                    float(x) for x in getattr(wrapped, "ctt_eig_min_curve")
                ]
            if hasattr(wrapped, "ctt_eig_max_curve") and getattr(
                wrapped, "ctt_eig_max_curve"
            ):
                ctt_eig_max_curve = [
                    float(x) for x in getattr(wrapped, "ctt_eig_max_curve")
                ]
            if hist_cb.losses and not train_curve:
                train_curve = [float(x) for x in hist_cb.losses]
            if hist_cb.val_losses and not val_curve:
                val_curve = [float(x) for x in hist_cb.val_losses]
            if getattr(hist_cb, "val_scores", None) and not score_curve:
                score_curve = [float(x) for x in hist_cb.val_scores]
    except Exception:
        train_curve = None
        val_curve = None
        score_curve = None
        var_z0_curve = None
        var_zt_curve = None
        cond_c0_curve = None
        cond_ctt_curve = None
        grad_norm_curve = None

    if train_curve is None:
        train_curve = [float(1.0 - obj_before), float(1.0 - obj_after)]
    history_epochs = list(range(1, len(train_curve) + 1))
    if score_curve is None:
        score_curve = [float(obj_before), float(obj_after)]
        if len(history_epochs) < len(score_curve):
            history_epochs = list(range(len(score_curve)))
    else:
        if len(history_epochs) < len(score_curve):
            history_epochs = list(range(1, len(score_curve) + 1))
    if var_z0_curve is None:
        var_z0_curve = []
    if var_zt_curve is None:
        var_zt_curve = []
    if cond_c0_curve is None:
        cond_c0_curve = []
    if cond_ctt_curve is None:
        cond_ctt_curve = []
    if grad_norm_curve is None:
        grad_norm_curve = []
    if var_z0_components is None:
        var_z0_components = var_z0_curve
    if var_zt_components is None:
        var_zt_components = var_zt_curve
    if mean_z0_curve is None:
        mean_z0_curve = []
    if mean_zt_curve is None:
        mean_zt_curve = []
    if c0_eig_min_curve is None:
        c0_eig_min_curve = []
    if c0_eig_max_curve is None:
        c0_eig_max_curve = []
    if ctt_eig_min_curve is None:
        ctt_eig_min_curve = []
    if ctt_eig_max_curve is None:
        ctt_eig_max_curve = []

    history: dict[str, Any] = {
        "loss_curve": train_curve,
        "val_loss_curve": val_curve,
        "objective_curve": score_curve,
        "val_score_curve": score_curve,
        "val_score": score_curve,
        "var_z0_curve": var_z0_curve,
        "var_zt_curve": var_zt_curve,
        "var_z0_curve_components": var_z0_components,
        "var_zt_curve_components": var_zt_components,
        "mean_z0_curve": mean_z0_curve,
        "mean_zt_curve": mean_zt_curve,
        "cond_c00_curve": cond_c0_curve,
        "cond_ctt_curve": cond_ctt_curve,
        "grad_norm_curve": grad_norm_curve,
        "c0_eig_min_curve": c0_eig_min_curve,
        "c0_eig_max_curve": c0_eig_max_curve,
        "ctt_eig_min_curve": ctt_eig_min_curve,
        "ctt_eig_max_curve": ctt_eig_max_curve,
        "initial_objective": float(obj_before),
        "epochs": history_epochs,
        "log_every": int(cfg.log_every),
        "wall_time_s": float(max(0.0, _time.time() - t0)),
        "tau_schedule": [int(x) for x in tau_schedule],
        "pair_diagnostics": pair_diagnostics,
        "usable_pairs": pair_diagnostics.get("usable_pairs"),
        "pair_coverage": pair_diagnostics.get("pair_coverage"),
        "pairs_by_shard": pair_diagnostics.get("pairs_by_shard"),
        "short_shards": pair_diagnostics.get("short_shards"),
    }

    history["output_variance"] = whitening_info.get("output_variance")
    history["output_mean"] = whitening_info.get("mean")
    history["output_transform"] = whitening_info.get("transform")
    history["output_transform_applied"] = whitening_info.get("transform_applied")

    if history.get("var_z0_curve"):
        history["var_z0_curve"][-1] = whitening_info.get("output_variance")
    else:
        history["var_z0_curve"] = [whitening_info.get("output_variance")]

    if history.get("var_z0_curve_components"):
        history["var_z0_curve_components"][-1] = whitening_info.get("output_variance")
    else:
        history["var_z0_curve_components"] = [whitening_info.get("output_variance")]

    if history.get("var_zt_curve"):
        history["var_zt_curve"][-1] = whitening_info.get("var_zt")
    else:
        history["var_zt_curve"] = [whitening_info.get("var_zt")]

    if history.get("var_zt_curve_components"):
        history["var_zt_curve_components"][-1] = whitening_info.get("var_zt")
    else:
        history["var_zt_curve_components"] = [whitening_info.get("var_zt")]

    if history.get("cond_c00_curve"):
        history["cond_c00_curve"][-1] = whitening_info.get("cond_c00")
    else:
        history["cond_c00_curve"] = [whitening_info.get("cond_c00")]

    if history.get("cond_ctt_curve"):
        history["cond_ctt_curve"][-1] = whitening_info.get("cond_ctt")
    else:
        history["cond_ctt_curve"] = [whitening_info.get("cond_ctt")]

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
                    "output_variance": output_variance,
                    "train_loss_last": (
                        (
                            history.get("loss_curve", [None])
                            if isinstance(history.get("loss_curve"), list)
                            else [None]
                        )[-1]
                    ),
                    "val_loss_last": (
                        (
                            history.get("val_loss_curve", [None])
                            if isinstance(history.get("val_loss_curve"), list)
                            else [None]
                        )[-1]
                    ),
                    "val_score_last": (
                        (
                            history.get("val_score_curve", [None])
                            if isinstance(history.get("val_score_curve"), list)
                            else [None]
                        )[-1]
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
