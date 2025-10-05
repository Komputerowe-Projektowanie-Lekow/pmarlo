"""High-level orchestration helpers for DeepTICA training."""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn

from .dataset import split_sequences
from .history import vamp2_proxy
from .inputs import FeaturePrep, prepare_features
from .model import apply_output_whitening, build_network
from .pairs import PairInfo, build_pair_info
from .utils import set_all_seeds

logger = logging.getLogger(__name__)

__all__ = ["TrainingArtifacts", "train_deeptica_pipeline"]


@dataclass(slots=True)
class TrainingArtifacts:
    """Container with the essential pieces returned by a training run."""

    scaler: Any
    network: nn.Module
    history: dict[str, Any]
    device: str


def train_deeptica_pipeline(
    X_list: Sequence[np.ndarray],
    pairs: Tuple[np.ndarray, np.ndarray],
    cfg: Any,
    *,
    weights: np.ndarray | None = None,
) -> TrainingArtifacts:
    """Run the DeepTICA training loop and return fitted artefacts."""

    if not X_list:
        raise ValueError("Expected at least one trajectory array for DeepTICA")

    t0 = time.time()
    seed = int(getattr(cfg, "seed", 2024))
    set_all_seeds(seed)

    tau_schedule = _resolve_tau_schedule(cfg)
    arrays = [np.asarray(block, dtype=np.float32) for block in X_list]
    prep: FeaturePrep = prepare_features(arrays, tau_schedule=tau_schedule, seed=seed)

    net = build_network(cfg, prep.scaler, seed=seed)
    pair_info: PairInfo = build_pair_info(
        arrays, prep.tau_schedule, pairs=pairs, weights=weights
    )

    idx_t = np.asarray(pair_info.idx_t, dtype=np.int64)
    idx_tau = np.asarray(pair_info.idx_tau, dtype=np.int64)
    weights_arr = np.asarray(pair_info.weights, dtype=np.float32).reshape(-1)
    pair_diagnostics = dict(pair_info.diagnostics)

    fallback_lag = int(prep.tau_schedule[-1])
    usable_pairs, coverage, short_shards, total_possible, lag_used = _log_pair_diagnostics(
        pair_diagnostics, len(arrays), fallback_lag
    )

    net.eval()
    outputs0 = _forward_to_numpy(net, prep.Z)
    obj_before = vamp2_proxy(outputs0, idx_t, idx_tau)

    lengths = [np.asarray(block).shape[0] for block in arrays]
    sequences = split_sequences(prep.Z, lengths)

    history: dict[str, Any]
    history_source = "curriculum_trainer"
    summary_dir: Optional[Path] = None

    try:
        from pmarlo.ml.deeptica.trainer import CurriculumConfig, DeepTICACurriculumTrainer

        curriculum_cfg = _build_curriculum_config(
            cfg,
            prep.tau_schedule,
            run_stamp=f"{int(t0)}-{os.getpid()}",
            config_cls=CurriculumConfig,
        )
        summary_dir = (
            Path(curriculum_cfg.checkpoint_dir)
            if curriculum_cfg.checkpoint_dir is not None
            else None
        )
        trainer = DeepTICACurriculumTrainer(net, curriculum_cfg)
        history = trainer.fit(sequences)
    except Exception as exc:  # pragma: no cover - depends on optional extras
        logger.warning(
            "Curriculum trainer unavailable; falling back to legacy .fit(): %s",
            exc,
        )
        history = {}
        history_source = "legacy-fit"
        _legacy_fit_model(
            net,
            cfg,
            prep.Z,
            prep.tau_schedule,
            idx_t,
            idx_tau,
            weights_arr,
        )

    net, whitening_info = apply_output_whitening(net, prep.Z, idx_tau, apply=False)
    net.eval()
    outputs = _forward_to_numpy(net, prep.Z)

    outputs_arr = np.asarray(outputs, dtype=np.float64)
    obj_after = vamp2_proxy(outputs_arr, idx_t, idx_tau)
    output_variance = _compute_output_variance(outputs_arr)
    top_eigs = _estimate_top_eigenvalues(outputs_arr, idx_t, idx_tau, cfg)

    history = dict(history)
    history.setdefault("tau_schedule", [int(t) for t in prep.tau_schedule])
    history.setdefault("val_tau", lag_used)
    history.setdefault("epochs_per_tau", int(getattr(cfg, "epochs_per_tau", 15)))
    history.setdefault("loss_curve", [])
    history.setdefault("val_loss_curve", [])
    history.setdefault("val_score_curve", [])
    history.setdefault("grad_norm_curve", [])
    history["history_source"] = history_source
    history["wall_time_s"] = float(history.get("wall_time_s", time.time() - t0))
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
    if summary_dir is not None:
        history.setdefault("summary_dir", str(summary_dir))

    device = "cuda" if getattr(torch, "cuda", None) and torch.cuda.is_available() else "cpu"
    return TrainingArtifacts(
        scaler=prep.scaler,
        network=net,
        history=history,
        device=device,
    )


def _resolve_tau_schedule(cfg: Any) -> tuple[int, ...]:
    schedule = tuple(int(x) for x in (getattr(cfg, "tau_schedule", ()) or ()) if int(x) > 0)
    if schedule:
        return schedule
    lag = int(getattr(cfg, "lag", 0) or 0)
    if lag <= 0:
        raise ValueError("DeepTICA configuration must define a positive lag")
    return (lag,)


def _forward_to_numpy(net: nn.Module, data: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        try:
            outputs = net(data)  # type: ignore[misc]
        except Exception:
            tensor = torch.as_tensor(data, dtype=torch.float32)
            outputs = net(tensor)
        if isinstance(outputs, torch.Tensor):
            outputs = outputs.detach().cpu().numpy()
    return np.asarray(outputs, dtype=np.float64)


def _log_pair_diagnostics(
    diagnostics: dict[str, Any],
    n_shards: int,
    fallback_lag: int,
) -> tuple[int, float, list[int], int, int]:
    usable_pairs = int(diagnostics.get("usable_pairs", 0))
    coverage = float(diagnostics.get("pair_coverage", 0.0))
    short_shards = list(diagnostics.get("short_shards", []))
    total_possible = int(diagnostics.get("total_possible_pairs", 0))
    lag_used = int(diagnostics.get("lag_used", fallback_lag))

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
    checkpoint_path = _Path(checkpoint_dir) if checkpoint_dir else Path.cwd() / "checkpoints" / "deeptica" / run_stamp
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
    curriculum_cfg = config_cls(**cfg_kwargs)
    if curriculum_cfg.checkpoint_dir is not None:
        _Path(curriculum_cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    return curriculum_cfg


def _legacy_fit_model(
    net: nn.Module,
    cfg: Any,
    Z: np.ndarray,
    tau_schedule: tuple[int, ...],
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


def _compute_output_variance(outputs: np.ndarray) -> Optional[list[float]]:
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
        evals = np.clip(evals, 1e-12, None)
        inv_sqrt = evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T
        M = inv_sqrt @ Ct @ inv_sqrt.T
        eigs = np.linalg.eigvalsh((M + M.T) * 0.5)
        eigs = np.sort(eigs)[::-1]
        return [
            float(x)
            for x in eigs[: min(int(getattr(cfg, "n_out", 2)), eigs.size)]
        ]
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
        summary_dir.mkdir(parents=True, exist_ok=True)
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
