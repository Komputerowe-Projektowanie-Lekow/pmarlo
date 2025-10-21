from __future__ import annotations

import copy
import json
import logging
import types
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, List, Optional, Tuple, cast

import numpy as np
import torch  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore

from pmarlo import constants as const
from pmarlo.ml.deeptica.whitening import apply_output_transform
from pmarlo.utils.path_utils import ensure_directory
from pmarlo.utils.seed import set_global_seed as _set_global_seed

from .core.model import apply_output_whitening as core_apply_output_whitening
from .core.model import construct_deeptica_core as core_construct_deeptica_core
from .core.model import normalize_hidden_dropout as core_normalize_hidden_dropout
from .core.model import override_core_mlp as core_override_core_mlp
from .core.model import resolve_activation_module as core_resolve_activation_module
from .core.model import resolve_hidden_layers as core_resolve_hidden_layers
from .core.model import resolve_input_dropout as core_resolve_input_dropout
from .core.model import strip_batch_norm as core_strip_batch_norm
from .core.model import (
    wrap_with_preprocessing_layers as core_wrap_with_preprocessing_layers,
)
from .core.trainer_api import train_deeptica_mlcolvar, train_deeptica_pipeline
from .core.utils import safe_float as core_safe_float

_DEEPTICA_IMPORT_ERROR: Exception | None = None
try:  # pragma: no cover - optional extra
    from mlcolvar.cvs import DeepTICA  # type: ignore
except Exception as exc:  # pragma: no cover - optional extra
    DeepTICA = None  # type: ignore
    _DEEPTICA_IMPORT_ERROR = exc
else:  # pragma: no cover - optional extra
    _DEEPTICA_IMPORT_ERROR = None


logger = logging.getLogger(__name__)

_TRAINING_BACKENDS: dict[str, Any] = {
    "lightning": train_deeptica_pipeline,
    "curriculum": train_deeptica_pipeline,
    "mlcolvar": train_deeptica_mlcolvar,
}


class PmarloApiIncompatibilityError(RuntimeError):
    """Raised when mlcolvar API layout does not expose expected classes."""


if DeepTICA is None:
    if isinstance(_DEEPTICA_IMPORT_ERROR, ImportError):
        raise ImportError(
            "Install optional extra pmarlo[mlcv] to use Deep-TICA"
        ) from _DEEPTICA_IMPORT_ERROR
    raise PmarloApiIncompatibilityError(
        "mlcolvar installed but DeepTICA not found in expected locations"
    ) from _DEEPTICA_IMPORT_ERROR


def set_all_seeds(seed: int = 2024) -> None:
    """Compatibility wrapper maintained for callers importing from this module."""

    _set_global_seed(int(seed))


torch.set_float32_matmul_precision("high")
torch.set_default_dtype(torch.float32)


def _resolve_activation_module(name: str):
    return core_resolve_activation_module(name)


def _coerce_dropout_sequence(spec: Any) -> List[float]:
    if spec is None:
        return []
    if isinstance(spec, Iterable) and not isinstance(spec, (bytes, str)):
        return [core_safe_float(item) for item in spec]
    return [core_safe_float(spec)]


def _safe_float(value: Any) -> float:
    return core_safe_float(value)


def _normalize_hidden_dropout(spec: Any, num_hidden: int) -> List[float]:
    return list(core_normalize_hidden_dropout(spec, int(num_hidden)))


def _override_core_mlp(
    core,
    layers,
    activation_name: str,
    linear_head: bool,
    *,
    hidden_dropout: Any = None,
    layer_norm_hidden: bool = False,
) -> None:
    core_override_core_mlp(
        core,
        layers,
        activation_name,
        linear_head,
        hidden_dropout=hidden_dropout,
        layer_norm_hidden=layer_norm_hidden,
    )


def _apply_output_whitening(
    net,
    Z,
    idx_tau,
    *,
    apply: bool = False,
    eig_floor: float = const.DEEPTICA_DEFAULT_EIGEN_FLOOR,
):
    return core_apply_output_whitening(
        net,
        Z,
        idx_tau,
        apply=apply,
        eig_floor=eig_floor,
    )


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


    class _WhitenTransformModule(_nn.Module):  # type: ignore[misc]
        """Apply a fixed whitening transform to its inputs."""

        def __init__(
            self,
            mean: np.ndarray | torch.Tensor,
            transform: np.ndarray | torch.Tensor,
        ) -> None:
            super().__init__()
            self.register_buffer(
                "mean", torch.as_tensor(mean, dtype=torch.float32, device="cpu")
            )
            self.register_buffer(
                "transform",
                torch.as_tensor(transform, dtype=torch.float32, device="cpu"),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            y = x - self.mean
            return torch.matmul(y, self.transform.T)


@dataclass(frozen=True)
class DeepTICAConfig:
    lag: int
    n_out: int = 2
    hidden: Tuple[int, ...] = (32, 16)
    activation: str = "gelu"
    learning_rate: float = const.DEEPTICA_DEFAULT_LEARNING_RATE
    batch_size: int = 1024
    max_epochs: int = 200
    early_stopping: int = 25
    weight_decay: float = const.DEEPTICA_DEFAULT_WEIGHT_DECAY
    log_every: int = 1
    seed: int = 0
    device: str = "cpu"
    trainer_backend: str = "lightning"
    reweight_mode: str = "scaled_time"  # or "none"
    # New knobs for loaders and validation split
    val_frac: float = 0.1
    num_workers: int = 2
    # Optimization and regularization knobs
    lr_schedule: str = "cosine"  # "none" | "cosine"
    warmup_epochs: int = 5
    dropout: float = 0.0
    dropout_input: Optional[float] = None
    hidden_dropout: Tuple[float, ...] = field(default_factory=tuple)
    layer_norm_in: bool = False
    layer_norm_hidden: bool = False
    linear_head: bool = False
    # Dataset splitting/loader control
    val_split: str = "by_shard"  # "by_shard" | "random"
    batches_per_epoch: int = 200
    gradient_clip_val: float = 1.0
    gradient_clip_algorithm: str = "norm"
    tau_schedule: Tuple[int, ...] = field(default_factory=tuple)
    val_tau: Optional[int] = None
    epochs_per_tau: int = 15
    vamp_eps: float = const.DEEPTICA_DEFAULT_VAMP_EPS
    vamp_eps_abs: float = const.DEEPTICA_DEFAULT_VAMP_EPS_ABS
    vamp_alpha: float = 0.15
    vamp_cond_reg: float = const.DEEPTICA_DEFAULT_VAMP_COND_REG
    grad_norm_warn: Optional[float] = None
    variance_warn_threshold: float = const.DEEPTICA_DEFAULT_VARIANCE_WARN_THRESHOLD
    mean_warn_threshold: float = 5.0

    def __post_init__(self) -> None:
        backend = str(getattr(self, "trainer_backend", "lightning")).strip().lower()
        if backend not in _TRAINING_BACKENDS:
            valid = ", ".join(sorted(_TRAINING_BACKENDS))
            raise ValueError(f"trainer_backend must be one of: {valid}")
        object.__setattr__(self, "trainer_backend", backend)

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
        ensure_directory(path.parent)
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
        cfg = _load_deeptica_config(path)
        scaler = _load_scaler_checkpoint(path)
        core = _construct_deeptica_core(cfg, scaler)
        net = _wrap_with_preprocessing_layers(core, cfg, scaler)
        state = torch.load(path.with_suffix(".pt"), map_location="cpu")
        net.load_state_dict(state["state_dict"])  # type: ignore[index]
        net.eval()
        history = _load_training_history(path)
        return cls(cfg, scaler, net, training_history=history)

    def to_torchscript(self, path: Path) -> Path:
        path = Path(path)
        ensure_directory(path.parent)
        traceable = _build_traceable_deeptica_module(self.net)
        traceable.eval()
        traceable = traceable.to(device=torch.device("cpu"), dtype=torch.float32)
        # Trace with single precision (typical for inference)
        example = torch.zeros(1, int(self.scaler.mean_.shape[0]), dtype=torch.float32)
        ts = torch.jit.trace(traceable, example)
        out = path.with_suffix(".ts")
        try:
            ts.save(str(out))
        except Exception:
            # Fallback to torch.jit.save for broader compatibility
            torch.jit.save(ts, str(out))
        return out

    def plumed_snippet(self, model_path: Path) -> str:
        """Return a PLUMED snippet that references the exported TorchScript model."""
        ts = Path(model_path).with_suffix(".ts").name
        lines = [f"PYTORCH_MODEL FILE={ts} LABEL=mlcv"]
        for i in range(int(self.cfg.n_out)):
            lines.append(f"CV VALUE=mlcv.node-{i}")
        return "\n".join(lines) + "\n"


def _load_deeptica_config(path: Path) -> DeepTICAConfig:
    data = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    return DeepTICAConfig(**data)


def _load_scaler_checkpoint(path: Path) -> StandardScaler:
    scaler_ckpt = torch.load(path.with_suffix(".scaler.pt"), map_location="cpu")
    scaler = StandardScaler(with_mean=True, with_std=True)
    scaler.mean_ = np.asarray(scaler_ckpt["mean"], dtype=np.float64)
    scaler.scale_ = np.asarray(scaler_ckpt["std"], dtype=np.float64)
    try:  # pragma: no cover - attribute presence varies across versions
        scaler.n_features_in_ = int(scaler.mean_.shape[0])  # type: ignore[attr-defined]
    except Exception:
        pass
    return scaler


def _construct_deeptica_core(cfg: Any, scaler: StandardScaler):
    return core_construct_deeptica_core(cfg, scaler)


def _resolve_hidden_layers(cfg: Any) -> tuple[int, ...]:
    return core_resolve_hidden_layers(cfg)


def _wrap_with_preprocessing_layers(core: Any, cfg: Any, scaler: StandardScaler):
    return core_wrap_with_preprocessing_layers(core, cfg, scaler)


def _resolve_input_dropout(cfg: Any) -> float:
    return core_resolve_input_dropout(cfg)


def _strip_batch_norm(module: Any) -> None:
    core_strip_batch_norm(module)


def _load_training_history(path: Path) -> Optional[dict[str, Any]]:
    history_path = path.with_suffix(".history.json")
    if not history_path.exists():
        return None
    try:
        data = json.loads(history_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if isinstance(data, dict):
        return cast(dict[str, Any], data)
    return None


def _mark_module_scripting_safe(module: Any, *, _seen: set[int] | None = None) -> None:
    if _seen is None:
        _seen = set()
    marker = id(module)
    if marker in _seen:
        return
    _seen.add(marker)
    try:
        if hasattr(module, "_jit_is_scripting"):
            setattr(module, "_jit_is_scripting", True)
    except Exception:
        return
    try:
        iterator: Iterable[tuple[str, Any]] = getattr(
            module, "named_modules", lambda: []
        )()
    except Exception:
        return
    for _name, child in iterator:
        _mark_module_scripting_safe(child, _seen=_seen)


def _ensure_module_children_resolve(module: Any) -> None:
    try:
        import torch.nn as _nn  # type: ignore
    except Exception:  # pragma: no cover - optional dependency
        return

    if not isinstance(module, _nn.Module):
        return

    stack: list[_nn.Module] = [module]
    while stack:
        current = stack.pop()
        try:
            named = list(current.named_children())
        except Exception:
            named = []
        invalid = [name for name, _child in named if name not in current._modules]
        if invalid:
            # Restore default traversal behaviour so tracing can locate child modules
            current.named_children = types.MethodType(_nn.Module.named_children, current)  # type: ignore[assignment]
            current.children = types.MethodType(_nn.Module.children, current)  # type: ignore[assignment]
            current.named_modules = types.MethodType(_nn.Module.named_modules, current)  # type: ignore[assignment]
            current.modules = types.MethodType(_nn.Module.modules, current)  # type: ignore[assignment]
        for child in current._modules.values():
            if isinstance(child, _nn.Module):
                stack.append(child)


def _normalize_module_name(name: str) -> str:
    sanitized = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in name)
    return sanitized or "module"


def _build_traceable_deeptica_module(net: Any) -> torch.nn.Module:
    import torch.nn as _nn  # type: ignore

    modules = _collect_deeptica_modules_for_inference(net)
    ordered: list[tuple[str, torch.nn.Module]] = []
    counters: dict[str, int] = {}
    for name, module in modules:
        if module is None:
            continue
        module_copy = copy.deepcopy(module)
        module_copy.eval()
        module_copy = module_copy.to(device=torch.device("cpu"), dtype=torch.float32)
        _ensure_module_children_resolve(module_copy)
        key = _normalize_module_name(name)
        suffix = counters.get(key, 0)
        counters[key] = suffix + 1
        unique = key if suffix == 0 else f"{key}_{suffix}"
        ordered.append((unique, module_copy))
    if not ordered:
        ordered.append(("identity", _nn.Identity()))
    return _nn.Sequential(OrderedDict(ordered))


def _collect_deeptica_modules_for_inference(
    module: Any,
) -> list[tuple[str, torch.nn.Module | None]]:
    import torch.nn as _nn  # type: ignore

    collected: list[tuple[str, torch.nn.Module | None]] = []
    # Handle whitening wrappers generated during training
    if hasattr(module, "mean") and hasattr(module, "transform") and hasattr(module, "inner"):
        collected.extend(_collect_deeptica_modules_for_inference(module.inner))
        collected.append(("whiten", _WhitenTransformModule(module.mean, module.transform)))
        return collected
    # Handle preprocessing wrapper that applies layer norm and dropout
    if hasattr(module, "ln") and hasattr(module, "drop_in") and hasattr(module, "inner"):
        ln = getattr(module, "ln", None)
        drop_in = getattr(module, "drop_in", None)
        if isinstance(ln, _nn.Module):
            collected.append(("ln", ln))
        if isinstance(drop_in, _nn.Module):
            collected.append(("drop_in", drop_in))
        collected.extend(_collect_deeptica_modules_for_inference(module.inner))
        return collected
    # Core DeepTICA module from mlcolvar
    blocks = getattr(module, "BLOCKS", None)
    if isinstance(blocks, Iterable):
        preprocessing = getattr(module, "preprocessing", None)
        postprocessing = getattr(module, "postprocessing", None)
        if isinstance(preprocessing, _nn.Module):
            collected.append(("preprocessing", preprocessing))
        for block_name in blocks:
            block = getattr(module, block_name, None)
            if isinstance(block, _nn.Module):
                collected.append((str(block_name), block))
        if isinstance(postprocessing, _nn.Module):
            collected.append(("postprocessing", postprocessing))
        return collected
    if isinstance(module, _nn.Module):
        return [(module.__class__.__name__.lower(), module)]
    return []


def train_deeptica(
    X_list: List[np.ndarray],
    pairs: Tuple[np.ndarray, np.ndarray],
    cfg: DeepTICAConfig,
    weights: Optional[np.ndarray] = None,
) -> DeepTICAModel:
    """Train Deep-TICA using the modular core pipeline."""

    backend = str(getattr(cfg, "trainer_backend", "lightning")).strip().lower()
    training_fn = _TRAINING_BACKENDS.get(backend)
    if training_fn is None:
        valid = ", ".join(sorted(_TRAINING_BACKENDS))
        raise ValueError(
            f"Unsupported DeepTICA trainer backend '{backend}'. Valid backends: {valid}"
        )

    artifacts = training_fn(X_list, pairs, cfg, weights=weights)
    return DeepTICAModel(
        cfg,
        artifacts.scaler,
        artifacts.network,
        device=artifacts.device,
        training_history=artifacts.history,
    )
