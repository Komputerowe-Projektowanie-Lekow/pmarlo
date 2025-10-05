from __future__ import annotations

import json
import logging
import os as _os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional, Tuple

import numpy as np

# Standardize math defaults to float32 end-to-end
import torch  # type: ignore


logger = logging.getLogger(__name__)


def set_all_seeds(seed: int = 2024) -> None:
    """Compatibility wrapper around the core RNG seeding helper."""
    _core_set_all_seeds(int(seed))


class PmarloApiIncompatibilityError(RuntimeError):
    """Raised when mlcolvar API layout does not expose expected classes."""


# Official DeepTICA import and helpers (mlcolvar>=1.2)
try:  # pragma: no cover - optional extra
    from mlcolvar.cvs import DeepTICA  # type: ignore
except Exception as e:  # pragma: no cover - optional extra
    if isinstance(e, ImportError):
        raise ImportError("Install optional extra pmarlo[mlcv] to use Deep-TICA") from e
    raise PmarloApiIncompatibilityError(
        "mlcolvar installed but DeepTICA not found in expected locations"
    ) from e

# External scaling via scikit-learn (avoid internal normalization)
from sklearn.preprocessing import StandardScaler  # type: ignore

from pmarlo.ml.deeptica.whitening import apply_output_transform

from .core.inputs import FeaturePrep, prepare_features
from .core.model import (
    PrePostWrapper as _CorePrePostWrapper,
    WhitenWrapper as _CoreWhitenWrapper,
    apply_output_whitening as core_apply_output_whitening,
    construct_deeptica_core as core_construct_deeptica_core,
    normalize_hidden_dropout as core_normalize_hidden_dropout,
    override_core_mlp as core_override_core_mlp,
    resolve_activation_module as core_resolve_activation_module,
    resolve_hidden_layers as core_resolve_hidden_layers,
    resolve_input_dropout as core_resolve_input_dropout,
    strip_batch_norm as core_strip_batch_norm,
    wrap_with_preprocessing_layers as core_wrap_with_preprocessing_layers,
)
from .core.pairs import PairInfo, build_pair_info
from .core.utils import safe_float as core_safe_float, set_all_seeds as _core_set_all_seeds
from .losses import VAMP2Loss


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
    eig_floor: float = 1e-4,
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
        cfg = _load_deeptica_config(path)
        scaler = _load_scaler_checkpoint(path)
        core = _construct_deeptica_core(cfg, scaler)
        net = _wrap_with_preprocessing_layers(core, cfg, scaler)
        state = torch.load(path.with_suffix(".pt"), map_location="cpu")
        net.load_state_dict(state["state_dict"])  # type: ignore[index]
        net.eval()
        history = _load_training_history(path)
        return cls(cfg, scaler, net, training_history=history)


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

def _load_training_history(path: Path) -> Optional[dict]:
    history_path = path.with_suffix(".history.json")
    if not history_path.exists():
        return None
    try:
        return json.loads(history_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    def to_torchscript(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.net.eval()
        # Trace with single precision (typical for inference)
        example = torch.zeros(1, int(self.scaler.mean_.shape[0]), dtype=torch.float32)
        _mark_module_scripting_safe(self.net)
        base = getattr(self.net, "inner", None)
        if base is not None:
            _mark_module_scripting_safe(base)
        ts = torch.jit.trace(self.net.to(torch.float32), example)
        out = path.with_suffix(".ts")
        try:
            ts.save(str(out))
        except Exception:
            # Fallback to torch.jit.save for broader compatibility
            torch.jit.save(ts, str(out))
        return out


def _mark_module_scripting_safe(module: Any) -> None:
    try:
        if hasattr(module, "_jit_is_scripting"):
            setattr(module, "_jit_is_scripting", True)
    except Exception:
        return
    try:
        iterator = getattr(module, "named_modules", lambda: [])()
    except Exception:
        return
    for _name, child in iterator:
        _mark_module_scripting_safe(child)

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
    """Train Deep-TICA using the consolidated curriculum trainer."""

    import time as _time

    t0 = _time.time()
    set_all_seeds(int(getattr(cfg, "seed", 2024)))

    X = np.concatenate([np.asarray(block, dtype=np.float32) for block in X_list], axis=0)
    scaler = StandardScaler(with_mean=True, with_std=True).fit(
        np.asarray(X, dtype=np.float64)
    )
    Z = scaler.transform(np.asarray(X, dtype=np.float64)).astype(np.float32, copy=False)

    core_model, _ = core_construct_deeptica_core(cfg, scaler)
    net = core_wrap_with_preprocessing_layers(core_model, cfg, scaler)
    torch.manual_seed(int(getattr(cfg, "seed", 0)))

    tau_schedule = tuple(
        int(x) for x in (getattr(cfg, "tau_schedule", ()) or ()) if int(x) > 0
    ) or (int(cfg.lag),)

    pair_info = build_pair_info(
        [np.asarray(block) for block in X_list],
        tau_schedule,
        pairs=pairs,
        weights=weights,
    )
    idx_t = np.asarray(pair_info.idx_t, dtype=np.int64)
    idx_tau = np.asarray(pair_info.idx_tau, dtype=np.int64)
    weights_arr = np.asarray(pair_info.weights, dtype=np.float32).reshape(-1)
    pair_diagnostics = dict(pair_info.diagnostics)

    usable_pairs = int(pair_diagnostics.get("usable_pairs", idx_t.size))
    coverage = float(pair_diagnostics.get("pair_coverage", 0.0))
    short_shards = list(pair_diagnostics.get("short_shards", []))
    total_possible = int(pair_diagnostics.get("total_possible_pairs", 0))
    lag_used = int(pair_diagnostics.get("lag_used", tau_schedule[-1]))

    if short_shards:
        logger.warning(
            "%d/%d shards too short for lag %d",
            len(short_shards),
            len(X_list),
            lag_used,
        )
    if usable_pairs == 0:
        logger.warning(
            "No usable lagged pairs remain after constructing curriculum with lag %d",
            lag_used,
        )
    elif coverage < 0.5:
        logger.warning(
            "Lagged pair coverage low: %.1f%% (%d/%d possible pairs)",
            coverage * 100.0,
            usable_pairs,
            total_possible,
        )
    else:
        logger.info(
            "Lagged pair diagnostics: usable=%d coverage=%.1f%% short_shards=%s",
            usable_pairs,
            coverage * 100.0,
            short_shards,
        )

    with torch.no_grad():
        try:
            outputs0 = net(Z)  # type: ignore[misc]
        except Exception:
            outputs0 = net(torch.as_tensor(Z, dtype=torch.float32)).detach().cpu().numpy()  # type: ignore[assignment]
        if isinstance(outputs0, torch.Tensor):
            outputs0 = outputs0.detach().cpu().numpy()
    obj_before = _vamp2_proxy(np.asarray(outputs0, dtype=np.float64), idx_t, idx_tau)

    sequences = _split_sequences(
        Z, [int(np.asarray(block).shape[0]) for block in X_list]
    )

    history: dict[str, Any]
    history_source = "curriculum_trainer"
    summary_dir: Optional[Path] = None
    try:
        from pmarlo.ml.deeptica.trainer import DeepTICACurriculumTrainer  # type: ignore

        curriculum_cfg = _build_curriculum_config(
            cfg,
            tau_schedule,
            run_stamp=f"{int(t0)}-{_os.getpid()}",
        )
        summary_dir = curriculum_cfg.checkpoint_dir
        trainer = DeepTICACurriculumTrainer(net, curriculum_cfg)
        history = trainer.fit(sequences)
    except Exception as exc:  # pragma: no cover - relies on optional extras
        logger.warning(
            "Curriculum trainer unavailable; falling back to legacy .fit(): %s",
            exc,
        )
        history = {}
        history_source = "legacy-fit"
        _legacy_fit_model(
            net,
            cfg,
            Z,
            tau_schedule,
            idx_t,
            idx_tau,
            weights_arr,
        )

    net, whitening_info = _apply_output_whitening(net, Z, idx_tau, apply=False)
    net.eval()
    with torch.no_grad():
        try:
            outputs = net(Z)  # type: ignore[misc]
        except Exception:
            outputs = net(torch.as_tensor(Z, dtype=torch.float32)).detach().cpu().numpy()  # type: ignore[assignment]
        if isinstance(outputs, torch.Tensor):
            outputs = outputs.detach().cpu().numpy()

    outputs_arr = np.asarray(outputs, dtype=np.float64)
    obj_after = _vamp2_proxy(outputs_arr, idx_t, idx_tau)
    output_variance = _compute_output_variance(outputs_arr)
    top_eigs = _estimate_top_eigenvalues(outputs_arr, idx_t, idx_tau, cfg)

    history = dict(history)
    history.setdefault("tau_schedule", [int(t) for t in tau_schedule])
    history.setdefault("val_tau", lag_used)
    history.setdefault("epochs_per_tau", int(getattr(cfg, "epochs_per_tau", 15)))
    history.setdefault("loss_curve", [])
    history.setdefault("val_loss_curve", [])
    history.setdefault("val_score_curve", [])
    history.setdefault("grad_norm_curve", [])
    history["history_source"] = history_source
    history["wall_time_s"] = float(history.get("wall_time_s", _time.time() - t0))
    history["vamp2_before"] = float(obj_before)
    history["vamp2_after"] = float(obj_after)
    history["output_variance"] = output_variance
    if top_eigs is not None:
        history["top_eigenvalues"] = top_eigs
    history["pair_diagnostics_overall"] = pair_diagnostics
    history["usable_pairs"] = usable_pairs
    history["pair_coverage"] = coverage
    history["pairs_by_shard"] = pair_diagnostics.get("pairs_by_shard", [])
    history["short_shards"] = short_shards
    history["total_possible_pairs"] = total_possible
    history["lag_used"] = lag_used
    history["weights_mean"] = float(np.mean(weights_arr)) if weights_arr.size else 0.0
    history["weights_count"] = int(weights_arr.size)

    pair_diag_entry = history.get("pair_diagnostics")
    if isinstance(pair_diag_entry, dict):
        pair_diag_entry.setdefault("overall", pair_diagnostics)
    else:
        history["pair_diagnostics"] = {"overall": pair_diagnostics}

    history["output_mean"] = whitening_info.get("mean")
    history["output_transform"] = whitening_info.get("transform")
    history["output_transform_applied"] = whitening_info.get("transform_applied", False)
    history["whitening"] = whitening_info

    summary_dir = summary_dir or _resolve_summary_directory(history)
    _write_training_summary(summary_dir, cfg, history, output_variance, top_eigs)

    device = "cuda" if (hasattr(torch, "cuda") and torch.cuda.is_available()) else "cpu"
    return DeepTICAModel(cfg, scaler, net, device=device, training_history=history)


def _vamp2_proxy(Y: np.ndarray, idx_t: np.ndarray, idx_tau: np.ndarray) -> float:
    if Y.size == 0 or idx_t.size == 0 or idx_tau.size == 0:
        return 0.0
    A = Y[idx_t]
    B = Y[idx_tau]
    A = A - np.mean(A, axis=0, keepdims=True)
    B = B - np.mean(B, axis=0, keepdims=True)
    A_std = np.std(A, axis=0, ddof=1) + 1e-12
    B_std = np.std(B, axis=0, ddof=1) + 1e-12
    A = A / A_std
    B = B / B_std
    num = np.sum(A * B, axis=0)
    den = max(1.0, A.shape[0] - 1)
    r = num / den
    return float(np.mean(r * r))


def _split_sequences(Z: np.ndarray, lengths: List[int]) -> List[np.ndarray]:
    sequences: List[np.ndarray] = []
    offset = 0
    total = Z.shape[0]
    for length in lengths:
        length = int(max(0, length))
        if length == 0:
            sequences.append(np.zeros((0, Z.shape[1]), dtype=np.float32))
            continue
        end = min(offset + length, total)
        sequences.append(Z[offset:end])
        offset = end
    if not sequences:
        sequences.append(Z)
    return sequences


def _build_curriculum_config(
    cfg: DeepTICAConfig,
    tau_schedule: Tuple[int, ...],
    *,
    run_stamp: str,
):
    from pmarlo.ml.deeptica.trainer import CurriculumConfig  # type: ignore

    schedule = tuple(sorted({int(t) for t in tau_schedule if int(t) > 0})) or (int(cfg.lag),)
    val_frac = float(getattr(cfg, "val_frac", 0.1))
    if not (0.0 < val_frac < 1.0):
        val_frac = 0.1
    grad_clip = getattr(cfg, "gradient_clip_val", None)
    if grad_clip is not None:
        grad_clip = float(grad_clip)
        if grad_clip <= 0:
            grad_clip = None
    batches_per_epoch = getattr(cfg, "batches_per_epoch", None)
    if batches_per_epoch is not None:
        batches_per_epoch = int(batches_per_epoch)
        if batches_per_epoch <= 0:
            batches_per_epoch = None
    checkpoint_dir = getattr(cfg, "checkpoint_dir", None)
    if checkpoint_dir:
        checkpoint_path = Path(checkpoint_dir)
    else:
        checkpoint_path = Path.cwd() / "checkpoints" / "deeptica" / run_stamp
    cfg_kwargs = dict(
        tau_schedule=schedule,
        val_tau=int(getattr(cfg, "val_tau", 0) or schedule[-1]),
        epochs_per_tau=int(max(1, getattr(cfg, "epochs_per_tau", 15))),
        warmup_epochs=int(max(0, getattr(cfg, "warmup_epochs", 5))),
        batch_size=int(max(1, getattr(cfg, "batch_size", 256))),
        learning_rate=float(getattr(cfg, "learning_rate", 3e-4)),
        weight_decay=float(max(0.0, getattr(cfg, "weight_decay", 1e-4))),
        val_fraction=val_frac,
        shuffle=True,
        num_workers=int(max(0, getattr(cfg, "num_workers", 0))),
        device=str(getattr(cfg, "device", "auto")),
        grad_clip_norm=grad_clip,
        log_every=int(max(1, getattr(cfg, "log_every", 1))),
        checkpoint_dir=checkpoint_path,
        vamp_eps=float(getattr(cfg, "vamp_eps", 1e-3)),
        vamp_eps_abs=float(getattr(cfg, "vamp_eps_abs", 1e-6)),
        vamp_alpha=float(getattr(cfg, "vamp_alpha", 0.15)),
        vamp_cond_reg=float(max(0.0, getattr(cfg, "vamp_cond_reg", 1e-4))),
        seed=int(getattr(cfg, "seed", 0)),
        max_batches_per_epoch=batches_per_epoch,
    )
    curriculum_cfg = CurriculumConfig(**cfg_kwargs)
    if curriculum_cfg.checkpoint_dir is not None:
        Path(curriculum_cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    return curriculum_cfg


def _legacy_fit_model(
    net,
    cfg: DeepTICAConfig,
    Z: np.ndarray,
    tau_schedule: Tuple[int, ...],
    idx_t: np.ndarray,
    idx_tau: np.ndarray,
    weights: np.ndarray,
) -> None:
    if not hasattr(net, "fit"):
        return
    try:
        net.fit(  # type: ignore[attr-defined]
            Z,
            lagtime=int(tau_schedule[-1]),
            idx_t=np.asarray(idx_t, dtype=int),
            idx_tlag=np.asarray(idx_tau, dtype=int),
            weights=np.asarray(weights, dtype=np.float32),
            batch_size=int(getattr(cfg, "batch_size", 256)),
            max_epochs=int(getattr(cfg, "max_epochs", 200)),
            early_stopping_patience=int(getattr(cfg, "early_stopping", 25)),
            shuffle=False,
        )
    except TypeError:
        try:
            net.fit(  # type: ignore[attr-defined]
                Z,
                lagtime=int(tau_schedule[-1]),
                batch_size=int(getattr(cfg, "batch_size", 256)),
                max_epochs=int(getattr(cfg, "max_epochs", 200)),
            )
        except Exception as exc:  # pragma: no cover - diagnostic only
            logger.warning("Legacy DeepTICA fit fallback failed: %s", exc)
    except Exception as exc:  # pragma: no cover - diagnostic only
        logger.warning("Legacy DeepTICA fit raised an error: %s", exc)


def _compute_output_variance(outputs: np.ndarray) -> Optional[List[float]]:
    try:
        if outputs.size == 0:
            return []
        if outputs.shape[0] > 1:
            var_arr = np.var(outputs, axis=0, ddof=1)
        else:
            var_arr = np.var(outputs, axis=0, ddof=0)
        return np.asarray(var_arr, dtype=float).tolist()
    except Exception:
        return None


def _estimate_top_eigenvalues(
    outputs: np.ndarray,
    idx_t: np.ndarray,
    idx_tau: np.ndarray,
    cfg: DeepTICAConfig,
) -> Optional[List[float]]:
    try:
        if idx_t.size == 0 or idx_tau.size == 0:
            return None
        y_t = outputs[idx_t]
        y_tau = outputs[idx_tau]
        y_t_c = y_t - np.mean(y_t, axis=0, keepdims=True)
        y_tau_c = y_tau - np.mean(y_tau, axis=0, keepdims=True)
        n = max(1, y_t_c.shape[0] - 1)
        C0 = (y_t_c.T @ y_t_c) / float(n)
        Ct = (y_t_c.T @ y_tau_c) / float(n)
        evals, evecs = np.linalg.eigh((C0 + C0.T) * 0.5)
        evals = np.clip(evals, 1e-12, None)
        inv_sqrt = evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T
        M = inv_sqrt @ Ct @ inv_sqrt.T
        eigs = np.linalg.eigvalsh((M + M.T) * 0.5)
        eigs = np.sort(eigs)[::-1]
        return [float(x) for x in eigs[: min(int(getattr(cfg, "n_out", 2)), eigs.size)]]
    except Exception:
        return None


def _resolve_summary_directory(history: dict[str, Any]) -> Optional[Path]:
    metrics_csv = history.get("metrics_csv")
    if metrics_csv:
        try:
            return Path(str(metrics_csv)).resolve().parent
        except Exception:  # pragma: no cover - defensive
            return None
    best_ckpt = history.get("best_checkpoint")
    if best_ckpt:
        try:
            return Path(str(best_ckpt)).resolve().parent
        except Exception:  # pragma: no cover - defensive
            return None
    return None


def _write_training_summary(
    summary_dir: Optional[Path],
    cfg: DeepTICAConfig,
    history: dict[str, Any],
    output_variance: Optional[List[float]],
    top_eigs: Optional[List[float]],
) -> None:
    if summary_dir is None:
        return
    try:
        summary_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "config": asdict(cfg),
            "vamp2_before": history.get("vamp2_before"),
            "vamp2_after": history.get("vamp2_after"),
            "output_variance": output_variance,
            "top_eigenvalues": top_eigs,
            "artifacts": {
                "metrics_csv": history.get("metrics_csv"),
                "best_checkpoint": history.get("best_checkpoint"),
            },
        }
        (summary_dir / "training_summary.json").write_text(
            json.dumps(summary, sort_keys=True, indent=2),
            encoding="utf-8",
        )
    except Exception:  # pragma: no cover - diagnostic only
        pass
