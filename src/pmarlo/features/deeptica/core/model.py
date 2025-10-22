from __future__ import annotations

import types
from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import torch  # type: ignore
from scipy.linalg import eigh, eigvalsh
from sklearn.covariance import ShrunkCovariance

from pmarlo import constants as const

from pmarlo.utils.seed import set_global_seed

from .utils import safe_float

__all__ = [
    "apply_output_whitening",
    "construct_deeptica_core",
    "normalize_hidden_dropout",
    "override_core_mlp",
    "resolve_activation_module",
    "resolve_hidden_layers",
    "resolve_input_dropout",
    "strip_batch_norm",
    "wrap_with_preprocessing_layers",
    "wrap_network",
    "build_network",
    "WhitenWrapper",
    "PrePostWrapper",
]


_DEFAULT_SHRINKAGE = 0.02


def resolve_activation_module(name: str):
    import torch.nn as _nn  # type: ignore

    key = (name or "").strip().lower()
    if key in {"gelu", "gaussian"}:
        return _nn.GELU()
    if key in {"relu", "relu+"}:
        return _nn.ReLU()
    if key == "elu":
        return _nn.ELU()
    if key == "selu":
        return _nn.SELU()
    if key in {"leaky_relu", "lrelu"}:
        return _nn.LeakyReLU()
    return _nn.Tanh()


def normalize_hidden_dropout(spec: Any, transitions: int) -> list[float]:
    if transitions <= 0:
        return []
    if spec is None:
        return [0.0] * transitions
    if isinstance(spec, (int, float)) and not isinstance(spec, bool):
        return [float(spec)] * transitions
    if isinstance(spec, str):
        return [safe_float(spec)] * transitions
    if isinstance(spec, Iterable) and not isinstance(spec, (bytes, str)):
        values = [safe_float(item) for item in spec]
        if not values:
            values = [0.0]
        if len(values) < transitions:
            values.extend([values[-1]] * (transitions - len(values)))
        return values[:transitions]
    return [safe_float(spec)] * transitions


def override_core_mlp(
    core,
    layers: Iterable[int],
    activation_name: str,
    linear_head: bool,
    *,
    hidden_dropout: Any = None,
    layer_norm_hidden: bool = False,
) -> None:
    """Replace ``core.nn`` with a freshly constructed MLP."""

    import torch.nn as _nn  # type: ignore

    layer_sizes = list(map(int, layers))
    if len(layer_sizes) <= 1:
        core.nn = _nn.Identity()  # type: ignore[attr-defined]
        return

    dropout_values = normalize_hidden_dropout(hidden_dropout, len(layer_sizes) - 2)
    modules: list[_nn.Module] = []
    for idx in range(len(layer_sizes) - 1):
        in_features = layer_sizes[idx]
        out_features = layer_sizes[idx + 1]
        modules.append(_nn.Linear(in_features, out_features))
        is_last = idx == len(layer_sizes) - 2
        if not is_last:
            if layer_norm_hidden:
                modules.append(_nn.LayerNorm(out_features))
            modules.append(resolve_activation_module(activation_name))
            drop_p = dropout_values[idx] if idx < len(dropout_values) else 0.0
            if drop_p > 0:
                modules.append(_nn.Dropout(p=float(drop_p)))
        elif not linear_head:
            modules.append(resolve_activation_module(activation_name))

    core.nn = _nn.Sequential(*modules) if modules else _nn.Identity()  # type: ignore[attr-defined]


def _ensure_forward_callable(module: torch.nn.Module) -> torch.nn.Module:
    """Ensure ``module`` exposes a usable ``forward`` implementation.

    Some DeepTICA implementations rely on a backing ``.nn``/``.net`` attribute
    without overriding :meth:`torch.nn.Module.forward`. That breaks downstream
    code that invokes the module directly. We adapt those modules once here so
    all later invocations behave like a standard ``nn.Module``.
    """

    base_forward = torch.nn.Module.forward  # type: ignore[attr-defined]
    module_forward = module.__class__.forward  # type: ignore[attr-defined]
    if module_forward is not base_forward:
        return module

    delegate_attr: str | None = None
    for attr in ("net", "nn"):
        candidate = getattr(module, attr, None)
        if isinstance(candidate, torch.nn.Module) or callable(candidate):
            delegate_attr = attr
            break

    if delegate_attr is None:
        raise TypeError(
            f"{module.__class__.__name__} does not implement forward() "
            "and exposes no callable 'net'/'nn' attribute."
        )

    def _forward(self, x, attr=delegate_attr):
        target = getattr(self, attr, None)
        if isinstance(target, torch.nn.Module):
            return target(x)
        if callable(target):
            return target(x)
        raise TypeError(
            f"{self.__class__.__name__}.{attr} is not callable, "
            "cannot synthesize forward()."
        )

    module.forward = types.MethodType(_forward, module)
    return module


def apply_output_whitening(
    net: torch.nn.Module,
    Z: np.ndarray,
    idx_tau: np.ndarray | None,
    *,
    apply: bool = False,
    eig_floor: float = const.DEEPTICA_DEFAULT_EIGEN_FLOOR,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    tensor = torch.as_tensor(Z, dtype=torch.float32)
    with torch.no_grad():
        outputs = net(tensor)
        if isinstance(outputs, torch.Tensor):
            outputs = outputs.detach().cpu().numpy()
    if outputs is None or outputs.size == 0:
        return net, {
            "output_variance": [],
            "var_zt": [],
            "cond_c00": None,
            "cond_ctt": None,
            "mean": [],
            "transform": [],
            "transform_applied": bool(apply),
        }

    mean = np.mean(outputs, axis=0)
    centered = outputs - mean

    covariance = _estimate_covariance(centered)
    inv_sqrt, cond_c00 = _derive_whitening_transform(covariance, eig_floor=eig_floor)
    output_var = centered.var(axis=0, ddof=1).astype(float).tolist()

    var_zt = output_var
    cond_ctt = None
    if idx_tau is not None and idx_tau.size:
        tau_tensor = torch.as_tensor(Z[idx_tau], dtype=torch.float32)
        with torch.no_grad():
            base = net if not isinstance(net, WhitenWrapper) else net.inner
            tau_out = base(tau_tensor)
            if isinstance(tau_out, torch.Tensor):
                tau_out = tau_out.detach().cpu().numpy()
        tau_center = tau_out - mean
        var_zt = tau_center.var(axis=0, ddof=1).astype(float).tolist()
        tau_covariance = _estimate_covariance(tau_center)
        eig_ct = eigvalsh(tau_covariance, check_finite=False)
        floor_value = max(eig_floor, const.NUMERIC_MIN_EIGEN_CLIP)
        eig_ct = np.clip(eig_ct, floor_value, None)
        cond_ctt = float(eig_ct.max() / eig_ct.min())

    transform = inv_sqrt if apply else np.eye(inv_sqrt.shape[0], dtype=np.float64)
    wrapped = WhitenWrapper(net, mean, transform) if apply else net

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


class _SimpleDeepTICACore(torch.nn.Module):  # type: ignore[misc]
    """Plain PyTorch implementation mirroring the legacy DeepTICA interface."""

    def __init__(self) -> None:
        super().__init__()
        import torch.nn as _nn  # type: ignore

        self.nn: _nn.Module = _nn.Identity()

    def forward(self, x):  # type: ignore[override]
        return self.nn(x)


def construct_deeptica_core(cfg: Any, scaler) -> tuple[Any, list[int]]:
    in_dim = int(np.asarray(getattr(scaler, "mean_", []), dtype=np.float64).shape[0])
    out_dim = int(getattr(cfg, "n_out", getattr(cfg, "output_dim", 2)))
    layers = [in_dim, *resolve_hidden_layers(cfg), out_dim]
    activation_name = str(getattr(cfg, "activation", "gelu")).lower().strip() or "gelu"

    core = _SimpleDeepTICACore()
    override_core_mlp(
        core,
        layers,
        activation_name,
        bool(getattr(cfg, "linear_head", False)),
        hidden_dropout=getattr(cfg, "hidden_dropout", None),
        layer_norm_hidden=bool(getattr(cfg, "layer_norm_hidden", False)),
    )
    strip_batch_norm(core)
    core = _ensure_forward_callable(core)
    return core, layers


def resolve_hidden_layers(cfg: Any) -> tuple[int, ...]:
    hidden_cfg = tuple(int(h) for h in getattr(cfg, "hidden", ()) or ())
    if bool(getattr(cfg, "linear_head", False)):
        return ()
    return hidden_cfg if hidden_cfg else (32, 16)


def wrap_with_preprocessing_layers(core: Any, cfg: Any, scaler) -> torch.nn.Module:
    in_dim = int(getattr(scaler, "mean_", np.zeros(1)).shape[0])
    dropout_in = resolve_input_dropout(cfg)
    ln_in = bool(getattr(cfg, "layer_norm_in", True))
    return PrePostWrapper(core, in_dim, ln_in=ln_in, p_drop=float(dropout_in))


def resolve_input_dropout(cfg: Any) -> float:
    dropout_in = getattr(cfg, "dropout_input", None)
    if dropout_in is None:
        dropout_in = getattr(cfg, "dropout", 0.1)
    return float(dropout_in if dropout_in is not None else 0.1)


def strip_batch_norm(module: Any) -> None:
    import torch.nn as _nn  # type: ignore

    for name, child in module.named_children():
        if isinstance(child, _nn.modules.batchnorm._BatchNorm):
            setattr(module, name, _nn.Identity())
        else:
            strip_batch_norm(child)


def wrap_network(cfg: Any, scaler, *, seed: int) -> torch.nn.Module:
    set_global_seed(seed)
    core, layers = construct_deeptica_core(cfg, scaler)
    # Note: override_core_mlp and strip_batch_norm are already called in construct_deeptica_core
    net = wrap_with_preprocessing_layers(core, cfg, scaler)
    torch.manual_seed(int(seed))
    return net


def build_network(cfg: Any, scaler, *, seed: int) -> torch.nn.Module:
    """Public alias for :func:`wrap_network` used by orchestration code."""

    return wrap_network(cfg, scaler, seed=seed)


def _estimate_covariance(samples: np.ndarray) -> np.ndarray:
    samples = np.asarray(samples, dtype=np.float64)
    if samples.size == 0:
        return np.eye(0, dtype=np.float64)

    n_samples, n_features = samples.shape
    if n_features == 0:
        return np.empty((0, 0), dtype=np.float64)

    if n_samples <= 1:
        return np.eye(n_features, dtype=np.float64)

    estimator = ShrunkCovariance(
        shrinkage=_DEFAULT_SHRINKAGE, assume_centered=True, store_precision=False
    )
    estimator.fit(samples)
    covariance = estimator.covariance_.astype(np.float64, copy=False)
    covariance = 0.5 * (covariance + covariance.T)

    trace = float(np.trace(covariance))
    trace = max(trace, const.NUMERIC_MIN_POSITIVE)
    mu = trace / float(max(1, n_features))
    ridge = mu * const.NUMERIC_RIDGE_SCALE
    if ridge > 0:
        covariance = covariance + np.eye(n_features, dtype=np.float64) * ridge

    return covariance


def _derive_whitening_transform(
    covariance: np.ndarray, *, eig_floor: float
) -> tuple[np.ndarray, float]:
    eig_floor = max(eig_floor, const.NUMERIC_MIN_EIGEN_CLIP)
    eigvals, eigvecs = eigh(covariance, check_finite=False)
    eigvals = np.clip(eigvals, eig_floor, None)
    cond = float(eigvals.max() / eigvals.min())
    transform = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
    transform = np.asarray(np.real_if_close(transform), dtype=np.float64)
    return transform, cond


class WhitenWrapper(torch.nn.Module):  # type: ignore[misc]
    def __init__(
        self,
        inner,
        mean: np.ndarray | torch.Tensor,
        transform: np.ndarray | torch.Tensor,
    ) -> None:
        super().__init__()
        self.inner = inner
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer(
            "transform", torch.as_tensor(transform, dtype=torch.float32)
        )

    def forward(self, x):  # type: ignore[override]
        y = self.inner(x)
        y = y - self.mean
        return torch.matmul(y, self.transform.T)


class PrePostWrapper(torch.nn.Module):  # type: ignore[misc]
    def __init__(self, inner: Any, in_features: int, *, ln_in: bool, p_drop: float):
        super().__init__()
        import torch.nn as _nn  # type: ignore

        self.ln = _nn.LayerNorm(in_features) if ln_in else _nn.Identity()
        prob = float(max(0.0, min(1.0, p_drop)))
        self.drop_in = _nn.Dropout(p=prob) if prob > 0 else _nn.Identity()
        self.inner = inner

    def forward(self, x):  # type: ignore[override]
        x = self.ln(x)
        x = self.drop_in(x)
        return self.inner(x)
