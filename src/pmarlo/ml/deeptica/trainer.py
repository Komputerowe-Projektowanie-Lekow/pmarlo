from __future__ import annotations

"""Curriculum trainer for DeepTICA style models."""

import csv
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, TypedDict, cast

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from pmarlo.features.deeptica.losses import VAMP2Loss

logger = logging.getLogger(__name__)


#inherit for the typeddict
class _TauBlock(TypedDict):
    tau: int
    epochs: List[int]
    train_loss_curve: List[float]
    train_score_curve: List[float]
    val_loss_curve: List[float]
    val_score_curve: List[float]
    learning_rate_curve: List[float]
    grad_norm_mean_curve: List[float]
    grad_norm_max_curve: List[float]
    diagnostics: Dict[str, object]


@torch.no_grad()
def _clone_state_dict(module: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {name: param.detach().cpu().clone() for name, param in module.state_dict().items()}


class _LaggedPairDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Dataset yielding time-lagged pairs for a fixed ``tau``."""

    def __init__(self, sequences: Sequence[np.ndarray], tau: int) -> None:
        self._raw_sequences = [np.asarray(seq, dtype=np.float32) for seq in sequences]
        if not self._raw_sequences:
            raise ValueError("sequences must contain at least one array")
        if any(seq.ndim != 2 for seq in self._raw_sequences):
            raise ValueError("each sequence must be two-dimensional")
        self._tensors = [torch.as_tensor(seq, dtype=torch.float32) for seq in self._raw_sequences]
        self._tau = 1
        self._pairs = torch.zeros((0, 3), dtype=torch.int64)
        self._pairs_per_shard: List[int] = []
        self._total_possible = 0
        self._short: List[int] = []
        self.set_tau(tau)

    @property
    def tau(self) -> int:
        return int(self._tau)

    def set_tau(self, tau: int) -> None:
        tau_int = int(tau)
        if tau_int <= 0:
            raise ValueError("tau must be positive")
        self._tau = tau_int
        pairs: List[torch.Tensor] = []
        counts: List[int] = []
        total_possible = 0
        short: List[int] = []
        for shard_idx, seq in enumerate(self._tensors):
            n_frames = int(seq.shape[0])
            possible = max(0, n_frames - tau_int)
            counts.append(possible)
            total_possible += possible
            if possible <= 0:
                if n_frames <= tau_int:
                    short.append(shard_idx)
                continue
            idx0 = torch.arange(0, possible, dtype=torch.int64)
            idx1 = idx0 + tau_int
            shard_ids = torch.full_like(idx0, shard_idx)
            pairs.append(torch.stack((shard_ids, idx0, idx1), dim=1))
        if pairs:
            self._pairs = torch.cat(pairs, dim=0)
        else:
            self._pairs = torch.zeros((0, 3), dtype=torch.int64)
        self._pairs_per_shard = counts
        self._total_possible = total_possible
        self._short = short

    def diagnostics(self) -> Dict[str, object]:
        coverage = 0.0
        if self._total_possible:
            coverage = float(len(self) / float(self._total_possible))
        return {
            "tau": self.tau,
            "usable_pairs": int(len(self)),
            "total_possible_pairs": int(self._total_possible),
            "pair_coverage": coverage,
            "pairs_per_shard": list(self._pairs_per_shard),
            "short_shards": list(self._short),
        }

    def __len__(self) -> int:
        return int(self._pairs.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        shard_idx, i, j = self._pairs[index].tolist()
        shard_tensor = self._tensors[int(shard_idx)]
        return shard_tensor[int(i)], shard_tensor[int(j)]


@dataclass(frozen=True)
class CurriculumConfig:
    """Hyperparameters governing the tau curriculum."""

    tau_schedule: Sequence[int] = (2,)
    val_tau: int = 0
    epochs_per_tau: int = 15
    warmup_epochs: int = 5
    batch_size: int = 256
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    val_fraction: float = 0.2
    shuffle: bool = True
    num_workers: int = 0
    device: str = "auto"
    grad_clip_norm: Optional[float] = 1.0
    log_every: int = 1
    checkpoint_dir: Optional[Path] = None
    vamp_eps: float = 1e-3
    vamp_eps_abs: float = 1e-6
    vamp_alpha: float = 0.15
    vamp_cond_reg: float = 1e-4
    seed: Optional[int] = None
    max_batches_per_epoch: Optional[int] = None

    def __post_init__(self) -> None:
        schedule = [int(t) for t in self.tau_schedule if int(t) > 0]
        if not schedule:
            schedule = [max(1, int(self.val_tau) or 2)]
        schedule.sort()
        object.__setattr__(self, "tau_schedule", tuple(schedule))
        val_tau = int(self.val_tau) if int(self.val_tau) > 0 else schedule[-1]
        object.__setattr__(self, "val_tau", val_tau)
        if int(self.epochs_per_tau) <= 0:
            raise ValueError("epochs_per_tau must be positive")
        if int(self.batch_size) <= 0:
            raise ValueError("batch_size must be positive")
        warmup = max(0, int(self.warmup_epochs))
        object.__setattr__(self, "warmup_epochs", warmup)
        frac = float(self.val_fraction)
        if not 0.0 < frac < 1.0:
            frac = min(max(frac, 1e-3), 0.9)
            object.__setattr__(self, "val_fraction", frac)
        if float(self.learning_rate) <= 0:
            raise ValueError("learning_rate must be positive")


class DeepTICACurriculumTrainer:
    """Train a model using a short→long tau curriculum with fixed validation lag."""

    def __init__(self, model: torch.nn.Module, cfg: CurriculumConfig) -> None:
        self.cfg = cfg
        self.model = model
        module = getattr(model, "net", model)
        if not isinstance(module, torch.nn.Module):
            raise TypeError("model must be a torch.nn.Module or expose a .net module")
        self.module: torch.nn.Module = module
        self.device = torch.device(self._resolve_device(cfg.device))
        self.module.to(self.device)
        if cfg.seed is not None:
            torch.manual_seed(int(cfg.seed))
            np.random.seed(int(cfg.seed))
        self.loss_fn = VAMP2Loss(
            eps=float(max(cfg.vamp_eps, 1e-9)),
            eps_abs=float(max(cfg.vamp_eps_abs, 0.0)),
            alpha=float(min(max(cfg.vamp_alpha, 0.0), 1.0)),
            cond_reg=float(max(cfg.vamp_cond_reg, 0.0)),
        ).to(self.device)
        params = list(self.module.parameters())
        self._trainable_parameters = [p for p in params if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            params,
            lr=float(cfg.learning_rate),
            weight_decay=float(max(cfg.weight_decay, 0.0)),
        )
        self._base_lr = float(cfg.learning_rate)
        self._current_lr = self._base_lr
        self._warmup_epochs = int(getattr(cfg, "warmup_epochs", 0))
        if self._warmup_epochs < 0:
            self._warmup_epochs = 0
        self._scheduler_total_epochs = 0
        self._scheduler_epoch = 0
        self._scheduler_warmup_epochs = 0
        self.history: Dict[str, object] = {}
        self._best_state: Optional[Dict[str, torch.Tensor]] = None
        self._best_epoch: int = -1
        self._best_tau: int = -1
        self._best_score: float = float("-inf")
        self._best_checkpoint_path: Optional[Path] = None
        self.cond_c00_curve: List[float] = []
        self.cond_ctt_curve: List[float] = []
        self.var_z0_curve: List[List[float]] = []
        self.var_zt_curve: List[List[float]] = []
        self.mean_z0_curve: List[List[float]] = []
        self.mean_zt_curve: List[List[float]] = []
        self.c0_eig_min_curve: List[float] = []
        self.c0_eig_max_curve: List[float] = []
        self.ctt_eig_min_curve: List[float] = []
        self.ctt_eig_max_curve: List[float] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(
        self,
        sequences: Sequence[np.ndarray],
        *,
        val_sequences: Optional[Sequence[np.ndarray]] = None,
    ) -> Dict[str, object]:
        """Train the wrapped model and return the history dictionary."""

        train_arrays = [np.asarray(seq, dtype=np.float32) for seq in sequences]
        if not train_arrays:
            raise ValueError("at least one training sequence is required")
        if any(arr.ndim != 2 for arr in train_arrays):
            raise ValueError("training sequences must all be 2-D arrays")
        if val_sequences is None:
            train_arrays, val_arrays = self._split_train_val(train_arrays)
        else:
            val_arrays = [np.asarray(seq, dtype=np.float32) for seq in val_sequences]
        if not val_arrays:
            raise ValueError("validation sequences are empty after splitting")

        val_dataset = _LaggedPairDataset(val_arrays, self.cfg.val_tau)
        if len(val_dataset) == 0:
            raise ValueError(
                "validation dataset is empty – ensure val_tau is compatible with sequence lengths"
            )
        val_loader = self._build_loader(val_dataset, shuffle=False)

        tau_blocks: List[tuple[int, _LaggedPairDataset, Dict[str, object]]] = []
        total_epochs_planned = 0
        for tau in self.cfg.tau_schedule:
            dataset = _LaggedPairDataset(train_arrays, tau)
            diag = dataset.diagnostics()
            tau_blocks.append((int(tau), dataset, diag))
            if len(dataset) > 0:
                total_epochs_planned += int(self.cfg.epochs_per_tau)

        self._initialize_scheduler(total_epochs_planned)

        per_tau_blocks: List[_TauBlock] = []
        overall_epochs: List[int] = []
        overall_train_loss: List[float] = []
        overall_train_score: List[float] = []
        overall_val_loss: List[float] = []
        overall_val_score: List[float] = []
        overall_learning_rate: List[float] = []
        overall_grad_norm_mean: List[float] = []
        overall_grad_norm_max: List[float] = []
        start_time = time.time()

        for tau, dataset, diag in tau_blocks:
            block: _TauBlock = {
                "tau": int(tau),
                "epochs": [],
                "train_loss_curve": [],
                "train_score_curve": [],
                "val_loss_curve": [],
                "val_score_curve": [],
                "learning_rate_curve": [],
                "grad_norm_mean_curve": [],
                "grad_norm_max_curve": [],
                "diagnostics": diag,
            }
            per_tau_blocks.append(block)
            if len(dataset) == 0:
                logger.warning(
                    "No lagged pairs available at tau=%d; skipping curriculum stage", tau
                )
                continue
            loader = self._build_loader(dataset, shuffle=self.cfg.shuffle)
            if loader is None:
                continue
            logger.info("Starting tau stage tau=%d (val_tau=%d)", tau, self.cfg.val_tau)
            for epoch_idx in range(int(self.cfg.epochs_per_tau)):
                current_lr = self._step_scheduler()
                train_metrics = self._train_one_epoch(loader)
                val_metrics = self._evaluate(val_loader)
                overall_epoch = len(overall_epochs) + 1
                overall_epochs.append(overall_epoch)
                overall_train_loss.append(train_metrics["loss"])
                overall_train_score.append(train_metrics["score"])
                overall_val_loss.append(val_metrics["loss"])
                overall_val_score.append(val_metrics["score"])
                overall_learning_rate.append(current_lr)
                overall_grad_norm_mean.append(train_metrics["grad_norm_mean"])
                overall_grad_norm_max.append(train_metrics["grad_norm_max"])
                cond_c00 = float(train_metrics.get("cond_c00", 0.0))
                cond_ctt = float(train_metrics.get("cond_ctt", 0.0))
                self.cond_c00_curve.append(cond_c00)
                self.cond_ctt_curve.append(cond_ctt)
                var_z0 = train_metrics.get("var_z0")
                var_zt = train_metrics.get("var_zt")
                mean_z0 = train_metrics.get("mean_z0")
                mean_zt = train_metrics.get("mean_zt")
                eig0_min = float(train_metrics.get("eig_c00_min", float("nan")))
                eig0_max = float(train_metrics.get("eig_c00_max", float("nan")))
                eigt_min = float(train_metrics.get("eig_ctt_min", float("nan")))
                eigt_max = float(train_metrics.get("eig_ctt_max", float("nan")))
                self.c0_eig_min_curve.append(eig0_min)
                self.c0_eig_max_curve.append(eig0_max)
                self.ctt_eig_min_curve.append(eigt_min)
                self.ctt_eig_max_curve.append(eigt_max)
                self.var_z0_curve.append([float(x) for x in var_z0] if isinstance(var_z0, list) else [])
                self.var_zt_curve.append([float(x) for x in var_zt] if isinstance(var_zt, list) else [])
                self.mean_z0_curve.append([float(x) for x in mean_z0] if isinstance(mean_z0, list) else [])
                self.mean_zt_curve.append([float(x) for x in mean_zt] if isinstance(mean_zt, list) else [])
                if cond_c00 > 1e6:
                    logger.warning(
                        "Condition number cond(C00)=%.3e exceeds stability threshold", cond_c00
                    )
                if cond_ctt > 1e6:
                    logger.warning(
                        "Condition number cond(Ctt)=%.3e exceeds stability threshold", cond_ctt
                    )
                block["epochs"].append(overall_epoch)
                block["train_loss_curve"].append(train_metrics["loss"])
                block["train_score_curve"].append(train_metrics["score"])
                block["val_loss_curve"].append(val_metrics["loss"])
                block["val_score_curve"].append(val_metrics["score"])
                block["learning_rate_curve"].append(current_lr)
                block["grad_norm_mean_curve"].append(train_metrics["grad_norm_mean"])
                block["grad_norm_max_curve"].append(train_metrics["grad_norm_max"])
                self._update_best(overall_epoch, tau, val_metrics["score"])
                if self.cfg.log_every > 0:
                    if (
                        (epoch_idx + 1) % int(self.cfg.log_every) == 0
                        or epoch_idx + 1 == int(self.cfg.epochs_per_tau)
                    ):
                        logger.info(
                            (
                                "tau=%d val_tau=%d epoch=%d/%d lr=%.6e "
                                "train_loss=%.6f val_score=%.6f "
                                "grad_norm_mean=%.6f grad_norm_max=%.6f"
                            ),
                            tau,
                            self.cfg.val_tau,
                            epoch_idx + 1,
                            int(self.cfg.epochs_per_tau),
                            current_lr,
                            train_metrics["loss"],
                            val_metrics["score"],
                            train_metrics["grad_norm_mean"],
                            train_metrics["grad_norm_max"],
                        )

        if self._best_state is not None:
            self.module.load_state_dict(self._best_state)

        history: Dict[str, object] = {
            "tau_schedule": [int(t) for t in self.cfg.tau_schedule],
            "val_tau": int(self.cfg.val_tau),
            "epochs_per_tau": int(self.cfg.epochs_per_tau),
            "loss_curve": overall_train_loss,
            "objective_curve": overall_train_score,
            "val_loss_curve": overall_val_loss,
            "val_score_curve": overall_val_score,
            "learning_rate_curve": overall_learning_rate,
            "grad_norm_mean_curve": overall_grad_norm_mean,
            "grad_norm_max_curve": overall_grad_norm_max,
            "epochs": overall_epochs,
            "per_tau": per_tau_blocks,
            "per_tau_objective_curve": {
                int(block["tau"]): block["val_score_curve"] for block in per_tau_blocks
            },
            "per_tau_learning_rate_curve": {
                int(block["tau"]): block["learning_rate_curve"] for block in per_tau_blocks
            },
            "per_tau_grad_norm_mean_curve": {
                int(block["tau"]): block["grad_norm_mean_curve"]
                for block in per_tau_blocks
            },
            "per_tau_grad_norm_max_curve": {
                int(block["tau"]): block["grad_norm_max_curve"]
                for block in per_tau_blocks
            },
            "pair_diagnostics": {
                int(block["tau"]): block["diagnostics"] for block in per_tau_blocks
            },
            "cond_c00_curve": [float(x) for x in self.cond_c00_curve],
            "cond_ctt_curve": [float(x) for x in self.cond_ctt_curve],
            "var_z0_curve": [list(row) for row in self.var_z0_curve],
            "var_zt_curve": [list(row) for row in self.var_zt_curve],
            "mean_z0_curve": [list(row) for row in self.mean_z0_curve],
            "mean_zt_curve": [list(row) for row in self.mean_zt_curve],
            "c0_eig_min_curve": [float(x) for x in self.c0_eig_min_curve],
            "c0_eig_max_curve": [float(x) for x in self.c0_eig_max_curve],
            "ctt_eig_min_curve": [float(x) for x in self.ctt_eig_min_curve],
            "ctt_eig_max_curve": [float(x) for x in self.ctt_eig_max_curve],
            "grad_norm_curve": list(overall_grad_norm_mean),
            "best_val_score": float(self._best_score) if self._best_score > float("-inf") else 0.0,
            "best_epoch": int(self._best_epoch) if self._best_epoch >= 0 else None,
            "best_tau": int(self._best_tau) if self._best_tau >= 0 else None,
            "wall_time_s": float(max(0.0, time.time() - start_time)),
        }

        self.grad_norm_curve = overall_grad_norm_mean

        csv_path = self._write_metrics_csv(history)
        if csv_path is not None:
            history["metrics_csv"] = str(csv_path)
        if self._best_checkpoint_path is not None:
            history["best_checkpoint"] = str(self._best_checkpoint_path)

        self.history = history
        self._attach_history_to_model(history)
        return history

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _initialize_scheduler(self, total_epochs: int) -> None:
        total = max(0, int(total_epochs))
        self._scheduler_total_epochs = total
        warmup = min(max(0, self._warmup_epochs), total)
        self._scheduler_warmup_epochs = warmup
        self._scheduler_epoch = 0
        self._set_learning_rate(self._base_lr)

    def _step_scheduler(self) -> float:
        if self._scheduler_total_epochs <= 0:
            lr = self._base_lr
        else:
            epoch = self._scheduler_epoch
            warmup = self._scheduler_warmup_epochs
            total = self._scheduler_total_epochs
            if warmup > 0 and epoch < warmup:
                factor = float(epoch + 1) / float(max(1, warmup))
            else:
                if total <= warmup:
                    factor = 1.0
                else:
                    progress = float(epoch + 1 - warmup) / float(max(1, total - warmup))
                    progress = max(0.0, min(1.0, progress))
                    factor = 0.5 * (1.0 + math.cos(math.pi * progress))
            lr = self._base_lr * max(0.0, factor)
        self._set_learning_rate(lr)
        self._scheduler_epoch += 1
        return self._current_lr

    def _set_learning_rate(self, lr: float) -> None:
        lr_float = float(lr)
        for group in self.optimizer.param_groups:
            group["lr"] = lr_float
        self._current_lr = lr_float

    @staticmethod
    def _grad_norm(
        parameters: Sequence[torch.nn.Parameter], norm_type: float = 2.0
    ) -> float:
        grads = [p.grad for p in parameters if p.grad is not None]
        if not grads:
            return 0.0
        stacked = torch.stack([torch.norm(g.detach(), norm_type) for g in grads])
        total = torch.norm(stacked, norm_type)
        return float(total.detach().cpu().item())

    def _build_loader(
        self, dataset: _LaggedPairDataset, *, shuffle: bool
    ) -> Optional[DataLoader[tuple[torch.Tensor, torch.Tensor]]]:
        if len(dataset) == 0:
            return None
        return DataLoader(
            dataset,
            batch_size=int(self.cfg.batch_size),
            shuffle=bool(shuffle),
            num_workers=int(self.cfg.num_workers),
            drop_last=False,
        )

    def _train_one_epoch(
        self, loader: DataLoader[tuple[torch.Tensor, torch.Tensor]]
    ) -> Dict[str, float]:
        self.module.train()
        total_loss = 0.0
        total_score = 0.0
        total_weight = 0
        grad_norms: List[float] = []
        cond_c00_sum = 0.0
        cond_ctt_sum = 0.0
        cond_weight = 0.0
        var_z0_sum: np.ndarray | None = None
        var_zt_sum: np.ndarray | None = None
        mean_z0_sum: np.ndarray | None = None
        mean_zt_sum: np.ndarray | None = None
        eig0_min = float("inf")
        eig0_max = 0.0
        eigt_min = float("inf")
        eigt_max = 0.0
        max_batches = (
            int(self.cfg.max_batches_per_epoch)
            if self.cfg.max_batches_per_epoch is not None
            else None
        )
        for batch_idx, (x_t, x_tau) in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            batch_size = int(x_t.shape[0])
            if batch_size == 0:
                continue
            weights = torch.full(
                (batch_size,),
                1.0 / float(batch_size),
                dtype=torch.float32,
                device=self.device,
            )
            x_t = x_t.to(self.device)
            x_tau = x_tau.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            out_t = self.module(x_t)
            out_tau = self.module(x_tau)
            loss, score = self.loss_fn(out_t, out_tau, weights)
            loss.backward()
            if self.cfg.grad_clip_norm is not None and self._trainable_parameters:
                torch.nn.utils.clip_grad_norm_(
                    self._trainable_parameters, float(self.cfg.grad_clip_norm)
                )
            grad_norm = self._grad_norm(self._trainable_parameters)
            grad_norms.append(grad_norm)
            self.optimizer.step()
            total_loss += float(loss.item()) * batch_size
            total_score += float(score.item()) * batch_size
            total_weight += batch_size
            metrics = getattr(self.loss_fn, "latest_metrics", {})
            cond_c00 = float(metrics.get("cond_C00", 0.0))
            cond_ctt = float(metrics.get("cond_Ctt", 0.0))
            cond_c00_sum += cond_c00 * batch_size
            cond_ctt_sum += cond_ctt * batch_size
            cond_weight += batch_size
            var_z0 = metrics.get("var_z0")
            if isinstance(var_z0, list) and var_z0:
                arr = np.asarray(var_z0, dtype=np.float64)
                var_z0_sum = arr * batch_size if var_z0_sum is None else var_z0_sum + arr * batch_size
            var_zt = metrics.get("var_zt")
            if isinstance(var_zt, list) and var_zt:
                arr = np.asarray(var_zt, dtype=np.float64)
                var_zt_sum = arr * batch_size if var_zt_sum is None else var_zt_sum + arr * batch_size
            mean_z0 = metrics.get("mean_z0")
            if isinstance(mean_z0, list) and mean_z0:
                arr = np.asarray(mean_z0, dtype=np.float64)
                mean_z0_sum = arr * batch_size if mean_z0_sum is None else mean_z0_sum + arr * batch_size
            mean_zt = metrics.get("mean_zt")
            if isinstance(mean_zt, list) and mean_zt:
                arr = np.asarray(mean_zt, dtype=np.float64)
                mean_zt_sum = arr * batch_size if mean_zt_sum is None else mean_zt_sum + arr * batch_size
            eig0_min = min(eig0_min, float(metrics.get("eig_C00_min", eig0_min)))
            eig0_max = max(eig0_max, float(metrics.get("eig_C00_max", eig0_max)))
            eigt_min = min(eigt_min, float(metrics.get("eig_Ctt_min", eigt_min)))
            eigt_max = max(eigt_max, float(metrics.get("eig_Ctt_max", eigt_max)))
        if total_weight == 0:
            return {"loss": 0.0, "score": 0.0, "grad_norm_mean": 0.0, "grad_norm_max": 0.0}
        grad_norm_mean = float(np.mean(grad_norms)) if grad_norms else 0.0
        grad_norm_max = float(np.max(grad_norms)) if grad_norms else 0.0
        agg: Dict[str, float | List[float]] = {
            "loss": total_loss / float(total_weight),
            "score": total_score / float(total_weight),
            "grad_norm_mean": grad_norm_mean,
            "grad_norm_max": grad_norm_max,
        }
        if cond_weight > 0:
            agg["cond_c00"] = cond_c00_sum / cond_weight
            agg["cond_ctt"] = cond_ctt_sum / cond_weight
        if var_z0_sum is not None and cond_weight > 0:
            agg["var_z0"] = (var_z0_sum / cond_weight).tolist()
        if var_zt_sum is not None and cond_weight > 0:
            agg["var_zt"] = (var_zt_sum / cond_weight).tolist()
        if mean_z0_sum is not None and cond_weight > 0:
            agg["mean_z0"] = (mean_z0_sum / cond_weight).tolist()
        if mean_zt_sum is not None and cond_weight > 0:
            agg["mean_zt"] = (mean_zt_sum / cond_weight).tolist()
        if eig0_min != float("inf"):
            agg["eig_c00_min"] = eig0_min
        if eig0_max != 0.0:
            agg["eig_c00_max"] = eig0_max
        if eigt_min != float("inf"):
            agg["eig_ctt_min"] = eigt_min
        if eigt_max != 0.0:
            agg["eig_ctt_max"] = eigt_max
        return agg  # type: ignore[return-value]

    def _evaluate(
        self, loader: DataLoader[tuple[torch.Tensor, torch.Tensor]]
    ) -> Dict[str, float]:
        self.module.eval()
        total_loss = 0.0
        total_score = 0.0
        total_weight = 0
        with torch.no_grad():
            for batch_idx, (x_t, x_tau) in enumerate(loader):
                batch_size = int(x_t.shape[0])
                if batch_size == 0:
                    continue
                weights = torch.full(
                    (batch_size,),
                    1.0 / float(batch_size),
                    dtype=torch.float32,
                    device=self.device,
                )
                out_t = self.module(x_t.to(self.device))
                out_tau = self.module(x_tau.to(self.device))
                loss, score = self.loss_fn(out_t, out_tau, weights)
                total_loss += float(loss.item()) * batch_size
                total_score += float(score.item()) * batch_size
                total_weight += batch_size
        if total_weight == 0:
            return {"loss": 0.0, "score": 0.0}
        return {
            "loss": total_loss / float(total_weight),
            "score": total_score / float(total_weight),
        }

    def _split_train_val(
        self, sequences: Sequence[np.ndarray]
    ) -> tuple[List[np.ndarray], List[np.ndarray]]:
        max_tau = max([*self.cfg.tau_schedule, int(self.cfg.val_tau)])
        train_arrays: List[np.ndarray] = []
        val_arrays: List[np.ndarray] = []
        for seq in sequences:
            n_frames = int(seq.shape[0])
            if n_frames <= max_tau + 1:
                logger.debug(
                    "Skipping sequence with %d frames; requires at least %d", n_frames, max_tau + 2
                )
                continue
            val_len = max(int(np.ceil(n_frames * float(self.cfg.val_fraction))), max_tau + 1)
            if val_len >= n_frames:
                val_len = max_tau + 1
            split = n_frames - val_len
            if split <= max_tau:
                split = max_tau + 1
            if split >= n_frames:
                continue
            train_arrays.append(seq[:split].copy())
            val_arrays.append(seq[split:].copy())
        if not train_arrays:
            raise ValueError("no training data remained after splitting by time")
        if not val_arrays:
            raise ValueError("no validation data remained after splitting by time")
        return train_arrays, val_arrays

    def _update_best(self, epoch: int, tau: int, val_score: float) -> None:
        if val_score <= self._best_score:
            return
        self._best_score = float(val_score)
        self._best_epoch = int(epoch)
        self._best_tau = int(tau)
        self._best_state = _clone_state_dict(self.module)
        if self.cfg.checkpoint_dir is not None:
            ckpt_dir = Path(self.cfg.checkpoint_dir)
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            path = ckpt_dir / "best_val_tau.pt"
            torch.save(
                {
                    "state_dict": self._best_state,
                    "epoch": int(epoch),
                    "tau": int(tau),
                    "val_tau": int(self.cfg.val_tau),
                },
                path,
            )
            self._best_checkpoint_path = path

    def _write_metrics_csv(self, history: Dict[str, object]) -> Optional[Path]:
        if self.cfg.checkpoint_dir is None:
            return None
        ckpt_dir = Path(self.cfg.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        csv_path = ckpt_dir / "curriculum_metrics.csv"
        with csv_path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "epoch",
                    "tau",
                    "train_loss",
                    "train_score",
                    "val_loss",
                    "val_score",
                    "learning_rate",
                    "grad_norm_mean",
                    "grad_norm_max",
                ]
            )
            per_tau = cast(List[_TauBlock], history.get("per_tau", []))
            #per tau is a proper List of blocks, prevents int(object) overload
            for block in per_tau:
                tau = int(block["tau"])
                epochs = block["epochs"]
                train_loss = block["train_loss_curve"]
                train_score = block["train_score_curve"]
                val_loss = block["val_loss_curve"]
                val_score = block["val_score_curve"]
                lr_curve = block.get("learning_rate_curve", [])
                grad_mean_curve = block.get("grad_norm_mean_curve", [])
                grad_max_curve = block.get("grad_norm_max_curve", [])
                for idx, epoch in enumerate(epochs):
                    writer.writerow(
                        [
                            int(epoch),
                            tau,
                            float(train_loss[idx]),
                            float(train_score[idx]),
                            float(val_loss[idx]),
                            float(val_score[idx]),
                            float(lr_curve[idx]) if idx < len(lr_curve) else 0.0,
                            float(grad_mean_curve[idx]) if idx < len(grad_mean_curve) else 0.0,
                            float(grad_max_curve[idx]) if idx < len(grad_max_curve) else 0.0,
                        ]
                    )
        return csv_path

    def _attach_history_to_model(self, history: Dict[str, object]) -> None:
        try:
            existing = getattr(self.model, "training_history", {})
            if isinstance(existing, dict):
                existing.update(history)
                setattr(self.model, "training_history", existing)
            else:
                setattr(self.model, "training_history", history)
        except AttributeError:
            setattr(self.model, "training_history", history)

    @staticmethod
    def _resolve_device(spec: str) -> str:
        if spec.lower() == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return spec
