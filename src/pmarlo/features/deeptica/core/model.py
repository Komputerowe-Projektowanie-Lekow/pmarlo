from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import torch  # type: ignore

from pmarlo import constants as const

try:  # pragma: no cover - optional extra
    from mlcolvar.cvs import DeepTICA  # type: ignore
except Exception as exc:  # pragma: no cover
    raise ImportError("Install optional extra pmarlo[mlcv] to use Deep-TICA") from exc

from .utils import safe_float, set_all_seeds

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
    if linear_head:
        return
    import torch.nn as _nn  # type: ignore

    layers = list(map(int, layers))
    if len(layers) <= 2:
        return

    dropout_values = normalize_hidden_dropout(hidden_dropout, len(layers) - 2)
    modules: list[_nn.Module] = []
    for idx in range(len(layers) - 1):
        in_features = layers[idx]
        out_features = layers[idx + 1]
        modules.append(_nn.Linear(in_features, out_features))
        if idx < len(layers) - 2:
            if layer_norm_hidden:
                modules.append(_nn.LayerNorm(out_features))
            modules.append(resolve_activation_module(activation_name))
            drop_p = dropout_values[idx] if idx < len(dropout_values) else 0.0
            if drop_p > 0:
                modules.append(_nn.Dropout(p=float(drop_p)))
    if modules:
        core.nn = _nn.Sequential(*modules)  # type: ignore[attr-defined]


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
    n = max(1, centered.shape[0] - 1)
    C0 = (centered.T @ centered) / float(n)

    C0_reg = _regularize(C0)
    eigvals, eigvecs = np.linalg.eigh(C0_reg)
    eigvals = np.clip(eigvals, max(eig_floor, const.NUMERIC_MIN_EIGEN_CLIP), None)
    inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
    output_var = centered.var(axis=0, ddof=1).astype(float).tolist()
    cond_c00 = float(eigvals.max() / eigvals.min())

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
        n_tau = max(1, tau_center.shape[0] - 1)
        Ct = (tau_center.T @ tau_center) / float(n_tau)
        Ct_reg = _regularize(Ct)
        eig_ct = np.linalg.eigvalsh(Ct_reg)
        eig_ct = np.clip(eig_ct, max(eig_floor, const.NUMERIC_MIN_EIGEN_CLIP), None)
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


def construct_deeptica_core(cfg: Any, scaler) -> tuple[Any, list[int]]:
    in_dim = int(np.asarray(getattr(scaler, "mean_", []), dtype=np.float64).shape[0])
    layers = [in_dim, *resolve_hidden_layers(cfg), int(getattr(cfg, "n_out", 2))]
    activation_name = str(getattr(cfg, "activation", "gelu")).lower().strip() or "gelu"
    core = DeepTICA(
        layers=layers,
        n_cvs=int(getattr(cfg, "n_out", 2)),
        activation=activation_name,
        options={"norm_in": False},
    )
    override_core_mlp(
        core,
        layers,
        activation_name,
        bool(getattr(cfg, "linear_head", False)),
        hidden_dropout=getattr(cfg, "hidden_dropout", None),
        layer_norm_hidden=bool(getattr(cfg, "layer_norm_hidden", False)),
    )
    strip_batch_norm(core)
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
    set_all_seeds(seed)
    core, layers = construct_deeptica_core(cfg, scaler)
    override_core_mlp(
        core,
        layers,
        str(getattr(cfg, "activation", "gelu")).lower().strip() or "gelu",
        bool(getattr(cfg, "linear_head", False)),
        hidden_dropout=getattr(cfg, "hidden_dropout", None),
        layer_norm_hidden=bool(getattr(cfg, "layer_norm_hidden", False)),
    )
    strip_batch_norm(core)
    net = wrap_with_preprocessing_layers(core, cfg, scaler)
    torch.manual_seed(int(seed))
    return net


def build_network(cfg: Any, scaler, *, seed: int) -> torch.nn.Module:
    """Public alias for :func:`wrap_network` used by orchestration code."""

    return wrap_network(cfg, scaler, seed=seed)


def _regularize(mat: np.ndarray) -> np.ndarray:
    sym = 0.5 * (mat + mat.T)
    dim = sym.shape[0]
    eye = np.eye(dim, dtype=np.float64)
    trace = float(np.trace(sym))
    trace = max(trace, const.NUMERIC_MIN_POSITIVE)
    mu = trace / float(max(1, dim))
    ridge = mu * const.NUMERIC_RIDGE_SCALE
    alpha = 0.02
    return (1.0 - alpha) * sym + (alpha * mu + ridge) * eye


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
