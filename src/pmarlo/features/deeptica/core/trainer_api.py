"""High-level orchestration helpers for DeepTICA training."""

from __future__ import annotations

import json
import logging
import numbers
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn

from pmarlo import constants as const
from pmarlo.ml.deeptica.trainer import (
    CurriculumConfig,
    DeepTICACurriculumTrainer,
)
from pmarlo.utils.path_utils import ensure_directory
from pmarlo.utils.seed import set_global_seed

from .dataset import (
    TorchLoaderBundle,
    create_dataset,
    create_loaders,
    create_torch_pair_loaders,
    split_sequences,
)
from .history import vamp2_proxy
from .inputs import FeaturePrep, prepare_features
from .model import apply_output_whitening, build_network
from .pairs import PairInfo, build_pair_info

logger = logging.getLogger(__name__)

__all__ = [
    "TrainingArtifacts",
    "train_deeptica_pipeline",
    "train_deeptica_mlcolvar",
    "train_deeptica_deeptime",
]


@dataclass(slots=True)
class TrainingArtifacts:
    """Container with the essential pieces returned by a training run."""

    scaler: Any
    network: nn.Module
    history: dict[str, Any]
    device: str


@dataclass(slots=True)
class _TrainingPrep:
    """Shared preparation bundle for DeepTICA training backends."""

    arrays: list[np.ndarray]
    prep: FeaturePrep
    net: nn.Module
    core: nn.Module
    pair_info: PairInfo
    idx_t: np.ndarray
    idx_tau: np.ndarray
    weights: np.ndarray
    pair_diagnostics: dict[str, Any]
    sequences: list[np.ndarray]
    lengths: list[int]
    obj_before: float
    requested_lag: int
    usable_pairs: int
    coverage: float
    short_shards: list[int]
    total_possible: int
    lag_used: int
    summary_dir: Optional[Path]
    tau_schedule: tuple[int, ...]
    start_time: float


@dataclass(slots=True)
class _TrainingOutcome:
    """Result bundle returned by backend-specific trainers."""

    history: dict[str, Any]
    summary_dir: Optional[Path]
    device: str


def _prepare_training_prep(
    X_list: Sequence[np.ndarray],
    pairs: Tuple[np.ndarray, np.ndarray],
    cfg: Any,
    weights: np.ndarray | None,
) -> _TrainingPrep:
    if not X_list:
        raise ValueError("Expected at least one trajectory array for DeepTICA")

    start_time = time.time()
    seed = int(getattr(cfg, "seed", 2024))
    set_global_seed(seed)

    tau_schedule = _resolve_tau_schedule(cfg)
    arrays = [np.asarray(block, dtype=np.float32) for block in X_list]
    prep: FeaturePrep = prepare_features(arrays, tau_schedule=tau_schedule, seed=seed)

    net = build_network(cfg, prep.scaler, seed=seed)
    core = getattr(net, "inner", None)
    if not isinstance(core, nn.Module):
        raise RuntimeError(
            "Wrapped DeepTICA module is missing expected 'inner' network"
        )

    pair_info: PairInfo = build_pair_info(
        arrays, prep.tau_schedule, pairs=pairs, weights=weights
    )
    idx_t = np.asarray(pair_info.idx_t, dtype=np.int64)
    idx_tau = np.asarray(pair_info.idx_tau, dtype=np.int64)
    weights_arr = np.asarray(pair_info.weights, dtype=np.float32).reshape(-1)
    pair_diagnostics = dict(pair_info.diagnostics)

    requested_lag = int(prep.tau_schedule[-1])
    usable_pairs, coverage, short_shards, total_possible, lag_used = (
        _log_pair_diagnostics(pair_diagnostics, len(arrays), requested_lag)
    )

    net.eval()
    outputs0 = _forward_to_numpy(net, prep.Z)
    obj_before = float(vamp2_proxy(outputs0, idx_t, idx_tau))

    lengths = [np.asarray(block).shape[0] for block in arrays]
    sequences = split_sequences(prep.Z, lengths)

    return _TrainingPrep(
        arrays=arrays,
        prep=prep,
        net=net,
        core=core,
        pair_info=pair_info,
        idx_t=idx_t,
        idx_tau=idx_tau,
        weights=weights_arr,
        pair_diagnostics=pair_diagnostics,
        sequences=sequences,
        lengths=lengths,
        obj_before=obj_before,
        requested_lag=requested_lag,
        usable_pairs=usable_pairs,
        coverage=coverage,
        short_shards=short_shards,
        total_possible=total_possible,
        lag_used=lag_used,
        summary_dir=None,
        tau_schedule=prep.tau_schedule,
        start_time=start_time,
    )


def _train_with_curriculum(
    prep: _TrainingPrep,
    cfg: Any,
) -> _TrainingOutcome:
    curriculum_cfg = _build_curriculum_config(
        cfg,
        prep.tau_schedule,
        run_stamp=f"{int(prep.start_time)}-{os.getpid()}",
        config_cls=CurriculumConfig,
    )
    summary_dir = (
        Path(curriculum_cfg.checkpoint_dir)
        if curriculum_cfg.checkpoint_dir is not None
        else None
    )
    trainer = DeepTICACurriculumTrainer(prep.net, curriculum_cfg)
    history = trainer.fit(prep.sequences)
    device_spec = getattr(trainer.cfg, "device", "auto")
    if str(device_spec).lower() == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = str(device_spec)
    return _TrainingOutcome(
        history=dict(history), summary_dir=summary_dir, device=device
    )


def _train_with_mlcolvar(
    prep: _TrainingPrep,
    cfg: Any,
) -> _TrainingOutcome:
    try:
        import lightning.pytorch as pl  # type: ignore[import]
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "PyTorch Lightning is required for the 'mlcolvar' DeepTICA backend"
        ) from exc

    dataset = create_dataset(prep.prep.Z, prep.idx_t, prep.idx_tau, prep.weights)
    bundle = create_loaders(dataset, cfg)
    if bundle.dict_module is None:
        raise RuntimeError("mlcolvar DictModule unavailable for training")

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=1,
        max_epochs=int(getattr(cfg, "max_epochs", 200)),
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        log_every_n_steps=int(max(1, getattr(cfg, "log_every", 1))),
    )
    trainer.fit(prep.core, datamodule=bundle.dict_module)

    prep.core.eval()
    prep.net.eval()
    prep.core.cpu()
    prep.net.cpu()

    history: dict[str, Any] = {
        "history_source": "mlcolvar_trainer",
        "epochs_completed": int(getattr(trainer, "current_epoch", -1)) + 1,
    }

    metrics = getattr(trainer, "callback_metrics", None)
    if metrics:
        final_metrics: dict[str, float] = {}
        for key, value in metrics.items():
            if isinstance(value, numbers.Real):
                final_metrics[key] = float(value)
            elif isinstance(value, torch.Tensor) and value.numel() == 1:
                final_metrics[key] = float(value.detach().cpu().item())
        if final_metrics:
            history["final_metrics"] = final_metrics

    device = "cuda" if accelerator == "gpu" else "cpu"
    return _TrainingOutcome(history=history, summary_dir=None, device=device)


def _finalize_training_artifacts(
    prep: _TrainingPrep,
    outcome: _TrainingOutcome,
    cfg: Any,
) -> TrainingArtifacts:
    net, whitening_info = apply_output_whitening(
        prep.net, prep.prep.Z, prep.idx_tau, apply=False
    )
    net.eval()

    outputs = _forward_to_numpy(net, prep.prep.Z)
    outputs_arr = np.asarray(outputs, dtype=np.float64)
    obj_after = float(vamp2_proxy(outputs_arr, prep.idx_t, prep.idx_tau))
    output_variance = _compute_output_variance(outputs_arr)
    top_eigs = _estimate_top_eigenvalues(outputs_arr, prep.idx_t, prep.idx_tau, cfg)

    history = dict(outcome.history)
    history.setdefault("history_source", "curriculum_trainer")
    history.setdefault("tau_schedule", [int(t) for t in prep.tau_schedule])
    history.setdefault("val_tau", prep.lag_used)
    history.setdefault("epochs_per_tau", int(getattr(cfg, "epochs_per_tau", 15)))
    history.setdefault("loss_curve", [])
    history.setdefault("val_loss_curve", [])
    history.setdefault("val_score_curve", [])
    history.setdefault("grad_norm_curve", [])
    history["wall_time_s"] = float(
        history.get("wall_time_s", time.time() - prep.start_time)
    )
    history["vamp2_before"] = float(prep.obj_before)
    history["vamp2_after"] = obj_after
    history["output_variance"] = output_variance
    if top_eigs is not None:
        history["top_eigenvalues"] = top_eigs

    history["pair_diagnostics_overall"] = prep.pair_diagnostics
    n_frames = int(prep.prep.Z.shape[0])
    usable_pairs_capped = min(prep.usable_pairs, n_frames)
    if prep.total_possible > 0:
        coverage_capped = min(
            prep.coverage, usable_pairs_capped / float(prep.total_possible)
        )
    else:
        coverage_capped = 0.0
    history["usable_pairs"] = usable_pairs_capped
    history["pair_coverage"] = coverage_capped
    history["pairs_by_shard"] = prep.pair_diagnostics.get("pairs_by_shard", [])
    history["short_shards"] = prep.short_shards
    history["total_possible_pairs"] = prep.total_possible
    history["lag_used"] = prep.lag_used
    history["weights_mean"] = float(np.mean(prep.weights)) if prep.weights.size else 0.0
    history["weights_count"] = int(prep.weights.size)

    pair_diag_entry = history.get("pair_diagnostics")
    if isinstance(pair_diag_entry, dict):
        pair_diag_entry.setdefault("overall", prep.pair_diagnostics)
    else:
        history["pair_diagnostics"] = {"overall": prep.pair_diagnostics}

    history["output_mean"] = whitening_info.get("mean")
    history["output_transform"] = whitening_info.get("transform")
    history["output_transform_applied"] = whitening_info.get("transform_applied", False)
    history["whitening"] = whitening_info

    summary_dir = outcome.summary_dir or _resolve_summary_directory(history)
    _write_training_summary(summary_dir, cfg, history, output_variance, top_eigs)
    if summary_dir is not None:
        history.setdefault("summary_dir", str(summary_dir))

    device = outcome.device or "cpu"
    device = (
        device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    return TrainingArtifacts(
        scaler=prep.prep.scaler,
        network=net,
        history=history,
        device=device,
    )


def train_deeptica_pipeline(
    X_list: Sequence[np.ndarray],
    pairs: Tuple[np.ndarray, np.ndarray],
    cfg: Any,
    *,
    weights: np.ndarray | None = None,
) -> TrainingArtifacts:
    """Run the curriculum-backed DeepTICA training loop."""

    prep = _prepare_training_prep(X_list, pairs, cfg, weights)
    outcome = _train_with_curriculum(prep, cfg)
    return _finalize_training_artifacts(prep, outcome, cfg)


def train_deeptica_mlcolvar(
    X_list: Sequence[np.ndarray],
    pairs: Tuple[np.ndarray, np.ndarray],
    cfg: Any,
    *,
    weights: np.ndarray | None = None,
) -> TrainingArtifacts:
    """Train DeepTICA using the Lightning-backed mlcolvar Trainer."""

    prep = _prepare_training_prep(X_list, pairs, cfg, weights)
    outcome = _train_with_mlcolvar(prep, cfg)
    return _finalize_training_artifacts(prep, outcome, cfg)


def _train_with_deeptime(prep: _TrainingPrep, cfg: Any) -> _TrainingOutcome:
    try:
        from deeptime.decomposition.deep import VAMPNet  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "deeptime is required for the 'deeptime' DeepTICA backend"
        ) from exc

    loaders: TorchLoaderBundle = create_torch_pair_loaders(
        prep.prep.Z,
        prep.idx_t,
        prep.idx_tau,
        prep.weights,
        cfg,
        seed=prep.prep.seed,
    )

    device_spec = str(getattr(cfg, "device", "auto")).lower().strip()
    if device_spec == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(getattr(cfg, "device"))

    prep.net.to(device)
    vampnet = VAMPNet(
        lobe=prep.net,
        device=device,
        learning_rate=float(
            getattr(cfg, "learning_rate", const.DEEPTICA_DEFAULT_LEARNING_RATE)
        ),
        epsilon=float(getattr(cfg, "vamp_eps", const.DEEPTICA_DEFAULT_VAMP_EPS)),
        score_mode="regularize",
    )

    start_time = time.time()
    train_scores: list[tuple[int, float]] = []
    val_scores: list[tuple[int, float]] = []

    def _train_callback(step: int, score_tensor: torch.Tensor) -> None:
        score = float(score_tensor.detach().cpu().item())
        train_scores.append((int(step), score))

    def _val_callback(step: int, score_tensor: torch.Tensor) -> None:
        score = float(score_tensor.detach().cpu().item())
        val_scores.append((int(step), score))

    vampnet.fit(
        loaders.train_loader,
        n_epochs=int(getattr(cfg, "max_epochs", 200)),
        validation_loader=loaders.val_loader,
        train_score_callback=_train_callback,
        validation_score_callback=_val_callback,
    )

    wall_time = float(time.time() - start_time)

    prep.net.eval()
    prep.net.cpu()

    history: dict[str, Any] = {
        "history_source": "deeptime_vampnet",
        "epochs_completed": int(getattr(cfg, "max_epochs", 200)),
        "train_scores": [[int(s), float(v)] for s, v in train_scores],
        "val_scores": [[int(s), float(v)] for s, v in val_scores],
        "loss_curve": [float(-v) for _, v in train_scores],
        "val_loss_curve": [float(-v) for _, v in val_scores],
        "val_score_curve": [float(v) for _, v in val_scores],
        "grad_norm_curve": [],
        "wall_time_s": wall_time,
        "train_samples": int(loaders.train_size),
        "val_samples": int(loaders.val_size),
    }

    if train_scores:
        val_map = {int(step): float(score) for step, score in val_scores}
        steps: list[dict[str, float | int]] = []
        for step, score in train_scores:
            entry: dict[str, float | int] = {
                "step": int(step),
                "train_vamp2": float(score),
            }
            if step in val_map:
                entry["val_vamp2"] = val_map[step]
            steps.append(entry)
        history["steps"] = steps

    device_label = device.type
    if device.index is not None:
        device_label = f"{device_label}:{device.index}"

    return _TrainingOutcome(history=history, summary_dir=None, device=device_label)


def train_deeptica_deeptime(
    X_list: Sequence[np.ndarray],
    pairs: Tuple[np.ndarray, np.ndarray],
    cfg: Any,
    *,
    weights: np.ndarray | None = None,
) -> TrainingArtifacts:
    prep = _prepare_training_prep(X_list, pairs, cfg, weights)
    outcome = _train_with_deeptime(prep, cfg)
    return _finalize_training_artifacts(prep, outcome, cfg)


def _resolve_tau_schedule(cfg: Any) -> tuple[int, ...]:
    schedule = tuple(
        int(x) for x in (getattr(cfg, "tau_schedule", ()) or ()) if int(x) > 0
    )
    if schedule:
        return schedule
    lag = int(getattr(cfg, "lag", 0) or 0)
    if lag <= 0:
        raise ValueError("DeepTICA configuration must define a positive lag")
    return (lag,)


def _module_device(module: nn.Module) -> torch.device:
    for param in module.parameters():
        return param.device
    for buffer in module.buffers():
        return buffer.device
    return torch.device("cpu")


def _forward_to_tensor(net: nn.Module, data: np.ndarray) -> torch.Tensor:
    device = _module_device(net)
    with torch.no_grad():
        try:
            outputs = net(data)  # type: ignore[misc]
        except Exception:
            tensor = torch.as_tensor(data, dtype=torch.float32, device=device)
            outputs = net(tensor)
        if not isinstance(outputs, torch.Tensor):
            outputs = torch.as_tensor(
                np.asarray(outputs, dtype=np.float32), device=device
            )
        else:
            outputs = outputs.to(device)
    return outputs.detach()


def _forward_to_numpy(net: nn.Module, data: np.ndarray) -> np.ndarray:
    outputs = _forward_to_tensor(net, data)
    return outputs.cpu().numpy().astype(np.float64, copy=False)


def _log_pair_diagnostics(
    diagnostics: dict[str, Any],
    n_shards: int,
    requested_lag: int,
) -> tuple[int, float, list[int], int, int]:
    usable_pairs = int(diagnostics.get("usable_pairs", 0))
    coverage = float(diagnostics.get("pair_coverage", 0.0))
    short_shards = list(diagnostics.get("short_shards", []))
    total_possible = int(diagnostics.get("total_possible_pairs", 0))
    lag_used = int(diagnostics.get("lag_used", requested_lag))

    if short_shards:
        logger.warning(
            "%d/%d shards too short for lag %d",
            len(short_shards),
            n_shards,
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

    return usable_pairs, coverage, short_shards, total_possible, lag_used


def _build_curriculum_config(
    cfg: Any,
    tau_schedule: tuple[int, ...],
    *,
    run_stamp: str,
    config_cls,
):
    from pathlib import Path as _Path

    schedule = tuple(sorted({int(t) for t in tau_schedule if int(t) > 0})) or (
        int(tau_schedule[-1]),
    )
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
    checkpoint_path = _Path(checkpoint_dir) if checkpoint_dir else None
    cfg_kwargs = dict(
        tau_schedule=schedule,
        val_tau=int(getattr(cfg, "val_tau", 0) or schedule[-1]),
        epochs_per_tau=int(max(1, getattr(cfg, "epochs_per_tau", 15))),
        warmup_epochs=int(max(0, getattr(cfg, "warmup_epochs", 5))),
        batch_size=int(max(1, getattr(cfg, "batch_size", 256))),
        learning_rate=float(
            getattr(cfg, "learning_rate", const.DEEPTICA_DEFAULT_LEARNING_RATE)
        ),
        weight_decay=float(
            max(0.0, getattr(cfg, "weight_decay", const.DEEPTICA_DEFAULT_WEIGHT_DECAY))
        ),
        val_fraction=val_frac,
        shuffle=True,
        num_workers=int(max(0, getattr(cfg, "num_workers", 0))),
        device=str(getattr(cfg, "device", "auto")),
        grad_clip_norm=grad_clip,
        log_every=int(max(1, getattr(cfg, "log_every", 1))),
        checkpoint_dir=checkpoint_path,
        vamp_eps=float(getattr(cfg, "vamp_eps", const.DEEPTICA_DEFAULT_VAMP_EPS)),
        vamp_eps_abs=float(
            getattr(cfg, "vamp_eps_abs", const.DEEPTICA_DEFAULT_VAMP_EPS_ABS)
        ),
        vamp_alpha=float(getattr(cfg, "vamp_alpha", 0.15)),
        vamp_cond_reg=float(
            max(
                0.0,
                getattr(
                    cfg,
                    "vamp_cond_reg",
                    const.DEEPTICA_DEFAULT_VAMP_COND_REG,
                ),
            )
        ),
        seed=int(getattr(cfg, "seed", 0)),
        max_batches_per_epoch=batches_per_epoch,
    )
    curriculum_cfg = config_cls(**cfg_kwargs)
    return curriculum_cfg


def _compute_output_variance(
    outputs: np.ndarray | torch.Tensor,
) -> Optional[list[float]]:
    try:
        if isinstance(outputs, torch.Tensor):
            tensor = outputs.detach()
        else:
            tensor = torch.as_tensor(outputs)
        if tensor.numel() == 0:
            return []
        tensor = tensor.to(dtype=torch.float64)
        if tensor.ndim == 0:
            return [float(tensor.cpu().item())]
        unbiased = tensor.shape[0] > 1
        var_tensor = tensor.var(dim=0, unbiased=unbiased)
        flattened = var_tensor.cpu().reshape(-1)
        return [float(x) for x in flattened]
    except Exception:
        return None


def _estimate_top_eigenvalues(
    outputs: np.ndarray,
    idx_t: np.ndarray,
    idx_tau: np.ndarray,
    cfg: Any,
) -> Optional[list[float]]:
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
        evals = np.clip(evals, const.NUMERIC_MIN_POSITIVE, None)
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
    cfg: Any,
    history: dict[str, Any],
    output_variance: Optional[list[float]],
    top_eigs: Optional[list[float]],
) -> None:
    if summary_dir is None:
        return
    try:
        ensure_directory(summary_dir)
        try:
            cfg_dict = asdict(cfg)
        except TypeError:
            cfg_dict = dict(getattr(cfg, "__dict__", {}))
        summary = {
            "config": cfg_dict,
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
