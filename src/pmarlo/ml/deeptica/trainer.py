from __future__ import annotations

"""Curriculum trainer for DeepTICA style models."""

import csv
import logging
import math
import numbers
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    TypedDict,
    cast,
)

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from pmarlo import constants as const
from pmarlo.features.deeptica.losses import VAMP2Loss
from pmarlo.utils.path_utils import ensure_directory

logger = logging.getLogger(__name__)

MetricMapping = Mapping[str, object]


def prepare_batch(
    batch: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray | None]],
    *,
    device: torch.device | str = "cpu",
    use_weights: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Convert a batch of NumPy arrays into tensors on the requested device."""

    if not batch:
        raise ValueError("empty batch provided to prepare_batch")

    dev = torch.device(device) if not isinstance(device, torch.device) else device
    x_t_list: list[torch.Tensor] = []
    x_tau_list: list[torch.Tensor] = []
    weight_list: list[torch.Tensor] = []

    for item in batch:
        if len(item) != 3:
            raise ValueError("batch elements must be 3-tuples (x_t, x_tau, weights)")
        xt_arr, xtau_arr, weights = item
        xt_tensor = torch.as_tensor(np.asarray(xt_arr, dtype=np.float32), device=dev)
        xtau_tensor = torch.as_tensor(
            np.asarray(xtau_arr, dtype=np.float32), device=dev
        )
        x_t_list.append(xt_tensor)
        x_tau_list.append(xtau_tensor)
        if use_weights:
            if weights is None:
                raise ValueError(
                    "batch requires weights but a batch element provided None"
                )
            weight_tensor = torch.as_tensor(
                np.asarray(weights, dtype=np.float32), device=dev
            ).reshape(-1)
            weight_list.append(weight_tensor)

    x_t = torch.cat(x_t_list, dim=0) if len(x_t_list) > 1 else x_t_list[0]
    x_tau = torch.cat(x_tau_list, dim=0) if len(x_tau_list) > 1 else x_tau_list[0]

    total_frames = x_t.shape[0]
    if not use_weights:
        weights_tensor: torch.Tensor | None = None
    elif weight_list:
        weights_tensor = (
            torch.cat(weight_list, dim=0) if len(weight_list) > 1 else weight_list[0]
        )
        if weights_tensor.numel() != total_frames:
            raise ValueError("weights length does not match batch size")
    else:
        raise ValueError(
            "batch requires weights but none were provided in any batch element"
        )

    return x_t, x_tau, weights_tensor


def compute_loss_and_score(
    model: torch.nn.Module,
    loss_module: _LossModule,
    x_t: torch.Tensor,
    x_tau: torch.Tensor,
    weights: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward pass helper returning the loss/score pair."""

    z_t = model(x_t)
    z_tau = model(x_tau)
    loss, score = loss_module(z_t, z_tau, weights)
    return loss, score


def compute_grad_norm(parameters: Sequence[torch.nn.Parameter]) -> float:
    """Compute the L2 norm of gradients across the provided parameters."""

    total = 0.0
    for param in parameters:
        if param.grad is None:
            continue
        grad = param.grad.detach()
        total += float(torch.sum(grad * grad).item())
    return float(math.sqrt(total)) if total > 0.0 else 0.0


def make_metrics(
    *,
    loss: torch.Tensor,
    score: torch.Tensor,
    tau: int,
    optimizer: torch.optim.Optimizer,
    grad_norm: float,
) -> dict[str, float]:
    """Package scalar training metrics into a serialisable mapping."""

    if not optimizer.param_groups:
        raise ValueError(
            "optimizer has no parameter groups; cannot determine learning rate"
        )

    first_group = optimizer.param_groups[0]
    if "lr" not in first_group:
        raise ValueError(
            "optimizer parameter group 0 does not define a 'lr' value"
        )

    lr_value = first_group["lr"]
    if not isinstance(lr_value, numbers.Real):
        raise TypeError(
            "optimizer parameter group 0 contains a non-numeric learning rate"
        )

    lr = float(lr_value)

    return {
        "loss": float(loss.detach().cpu().item()),
        "vamp2": float(score.detach().cpu().item()),
        "tau": float(int(tau)),
        "learning_rate": lr,
        "grad_norm": float(grad_norm),
    }


def record_metrics(
    history: list[dict[str, float]],
    metrics: dict[str, float],
    *,
    model: torch.nn.Module | None = None,
) -> None:
    """Append metrics to the in-memory history and model hook if provided."""

    history.append(dict(metrics))
    if model is None:
        return
    steps = getattr(model, "training_history", None)
    if not isinstance(steps, dict):
        steps = {}
    step_list = list(steps.get("steps", []))
    step_list.append(dict(metrics))
    steps["steps"] = step_list
    setattr(model, "training_history", steps)


def checkpoint_if_better(
    *,
    model_net: torch.nn.Module,
    checkpoint_path: Path | str,
    metrics: Mapping[str, float],
    metric_name: str,
    best_score: float,
) -> float:
    """Persist model weights when the tracked metric improves."""

    score = float(metrics.get(metric_name, float("-inf")))
    if score <= best_score:
        return best_score
    path = Path(checkpoint_path)
    ensure_directory(path.parent)
    torch.save(model_net.state_dict(), path)
    return score


def _metric_scalar(metrics: MetricMapping, key: str, default: float = 0.0) -> float:
    """Extract a numeric scalar from a metrics mapping, guarding conversions."""

    value = metrics.get(key, default)
    if isinstance(value, numbers.Real):
        return float(value)
    if isinstance(value, np.generic):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return float(default)
    return float(default)


def _metric_vector(metrics: MetricMapping, key: str) -> list[float]:
    """Extract a sequence of floats from the metrics mapping when available."""

    value = metrics.get(key)
    if isinstance(value, np.ndarray):
        array = np.asarray(value, dtype=np.float64)
        return cast(list[float], array.tolist())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        result: list[float] = []
        for item in value:
            if isinstance(item, numbers.Real):
                result.append(float(item))
            elif isinstance(item, str):
                try:
                    result.append(float(item))
                except ValueError:
                    continue
        return result
    empty: list[float] = []
    return empty


# inherit for the typeddict
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


@dataclass
class _BatchOutcome:
    """Container holding results for a single optimisation step."""

    loss: float
    score: float
    batch_size: int
    grad_norm: float
    grad_norm_preclip: float  # Add pre-clip gradient norm
    metrics: Dict[str, object]


@dataclass
class _EpochAccumulator:
    """Accumulate metrics across batches for a full training epoch."""

    total_loss: float = 0.0
    total_score: float = 0.0
    total_weight: int = 0
    grad_norms: List[float] = field(default_factory=list)
    grad_norms_preclip: List[float] = field(default_factory=list)  # Add pre-clip norms
    cond_c00_sum: float = 0.0
    cond_ctt_sum: float = 0.0
    cond_weight: float = 0.0
    var_z0_sum: np.ndarray | None = None
    var_zt_sum: np.ndarray | None = None
    mean_z0_sum: np.ndarray | None = None
    mean_zt_sum: np.ndarray | None = None
    eig0_min: float = float("inf")
    eig0_max: float = 0.0
    eigt_min: float = float("inf")
    eigt_max: float = 0.0

    def update(self, outcome: _BatchOutcome) -> None:
        self.total_loss += outcome.loss * outcome.batch_size
        self.total_score += outcome.score * outcome.batch_size
        self.total_weight += outcome.batch_size
        self.grad_norms.append(outcome.grad_norm)
        self.grad_norms_preclip.append(outcome.grad_norm_preclip)  # Track pre-clip
        self._update_condition_metrics(outcome)

    def finalize(self) -> Dict[str, float | List[float]]:
        if self.total_weight == 0:
            return self._empty_epoch()
        metrics = self._base_epoch_metrics()
        self._append_condition_metrics_summary(metrics)
        self._append_eigenvalue_summary(metrics)
        return metrics

    def _empty_epoch(self) -> Dict[str, float | List[float]]:
        return {
            "loss": 0.0,
            "score": 0.0,
            "grad_norm_mean": 0.0,
            "grad_norm_max": 0.0,
        }

    def _grad_norm_stats(self) -> tuple[float, float, float, float]:
        """Return (postclip_mean, postclip_max, preclip_mean, preclip_max)."""
        if not self.grad_norms:
            return 0.0, 0.0, 0.0, 0.0
        postclip_mean = float(np.mean(self.grad_norms))
        postclip_max = float(np.max(self.grad_norms))
        preclip_mean = (
            float(np.mean(self.grad_norms_preclip)) if self.grad_norms_preclip else 0.0
        )
        preclip_max = (
            float(np.max(self.grad_norms_preclip)) if self.grad_norms_preclip else 0.0
        )
        return postclip_mean, postclip_max, preclip_mean, preclip_max

    def _base_epoch_metrics(self) -> Dict[str, float | List[float]]:
        grad_mean, grad_max, grad_preclip_mean, grad_preclip_max = (
            self._grad_norm_stats()
        )
        total_weight = float(self.total_weight)
        return {
            "loss": self.total_loss / total_weight,
            "score": self.total_score / total_weight,
            "grad_norm_mean": grad_mean,
            "grad_norm_max": grad_max,
            "grad_norm_preclip_mean": grad_preclip_mean,  # Add pre-clip stats
            "grad_norm_preclip_max": grad_preclip_max,
        }

    def _append_condition_metrics_summary(
        self, agg: Dict[str, float | List[float]]
    ) -> None:
        if self.cond_weight <= 0:
            return
        divisor = self.cond_weight
        agg["cond_c00"] = self.cond_c00_sum / divisor
        agg["cond_ctt"] = self.cond_ctt_sum / divisor
        if self.var_z0_sum is not None:
            agg["var_z0"] = (self.var_z0_sum / divisor).tolist()
        if self.var_zt_sum is not None:
            agg["var_zt"] = (self.var_zt_sum / divisor).tolist()
        if self.mean_z0_sum is not None:
            agg["mean_z0"] = (self.mean_z0_sum / divisor).tolist()
        if self.mean_zt_sum is not None:
            agg["mean_zt"] = (self.mean_zt_sum / divisor).tolist()

    def _append_eigenvalue_summary(self, agg: Dict[str, float | List[float]]) -> None:
        if self.eig0_min != float("inf"):
            agg["eig_c00_min"] = self.eig0_min
        if self.eig0_max != 0.0:
            agg["eig_c00_max"] = self.eig0_max
        if self.eigt_min != float("inf"):
            agg["eig_ctt_min"] = self.eigt_min
        if self.eigt_max != 0.0:
            agg["eig_ctt_max"] = self.eigt_max

    def _update_condition_metrics(self, outcome: _BatchOutcome) -> None:
        metrics: MetricMapping = outcome.metrics
        batch_size = float(outcome.batch_size)
        cond_c00 = _metric_scalar(metrics, "cond_C00")
        cond_ctt = _metric_scalar(metrics, "cond_Ctt")
        self.cond_c00_sum += cond_c00 * batch_size
        self.cond_ctt_sum += cond_ctt * batch_size
        self.cond_weight += batch_size
        self.var_z0_sum = self._accumulate_vector(
            self.var_z0_sum, metrics.get("var_z0"), batch_size
        )
        self.var_zt_sum = self._accumulate_vector(
            self.var_zt_sum, metrics.get("var_zt"), batch_size
        )
        self.mean_z0_sum = self._accumulate_vector(
            self.mean_z0_sum, metrics.get("mean_z0"), batch_size
        )
        self.mean_zt_sum = self._accumulate_vector(
            self.mean_zt_sum, metrics.get("mean_zt"), batch_size
        )
        self.eig0_min = min(
            self.eig0_min, _metric_scalar(metrics, "eig_C00_min", self.eig0_min)
        )
        self.eig0_max = max(
            self.eig0_max, _metric_scalar(metrics, "eig_C00_max", self.eig0_max)
        )
        self.eigt_min = min(
            self.eigt_min, _metric_scalar(metrics, "eig_Ctt_min", self.eigt_min)
        )
        self.eigt_max = max(
            self.eigt_max, _metric_scalar(metrics, "eig_Ctt_max", self.eigt_max)
        )

    @staticmethod
    def _accumulate_vector(
        accumulator: np.ndarray | None, values: object, weight: float
    ) -> np.ndarray | None:
        if not isinstance(values, list) or not values:
            return accumulator
        arr = np.asarray(values, dtype=np.float64) * weight
        if accumulator is None:
            return arr
        return accumulator + arr


@dataclass
class _CurriculumMetrics:
    """Running collections of curriculum-wide metrics."""

    per_tau_blocks: List[_TauBlock] = field(default_factory=list)
    overall_epochs: List[int] = field(default_factory=list)
    overall_train_loss: List[float] = field(default_factory=list)
    overall_train_score: List[float] = field(default_factory=list)
    overall_val_loss: List[float] = field(default_factory=list)
    overall_val_score: List[float] = field(default_factory=list)
    overall_learning_rate: List[float] = field(default_factory=list)
    overall_grad_norm_mean: List[float] = field(default_factory=list)
    overall_grad_norm_max: List[float] = field(default_factory=list)

    def add_tau_block(self, tau: int, diagnostics: Dict[str, object]) -> _TauBlock:
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
            "diagnostics": diagnostics,
        }
        self.per_tau_blocks.append(block)
        return block

    def next_epoch_index(self) -> int:
        return len(self.overall_epochs) + 1

    def record_epoch(
        self,
        block: _TauBlock,
        overall_epoch: int,
        train_metrics: MetricMapping,
        val_metrics: MetricMapping,
        lr: float,
    ) -> None:
        train_loss = _metric_scalar(train_metrics, "loss")
        train_score = _metric_scalar(train_metrics, "score")
        grad_norm_mean = _metric_scalar(train_metrics, "grad_norm_mean")
        grad_norm_max = _metric_scalar(train_metrics, "grad_norm_max")
        val_loss = _metric_scalar(val_metrics, "loss")
        val_score = _metric_scalar(val_metrics, "score")
        block["epochs"].append(overall_epoch)
        block["train_loss_curve"].append(train_loss)
        block["train_score_curve"].append(train_score)
        block["val_loss_curve"].append(val_loss)
        block["val_score_curve"].append(val_score)
        block["learning_rate_curve"].append(float(lr))
        block["grad_norm_mean_curve"].append(grad_norm_mean)
        block["grad_norm_max_curve"].append(grad_norm_max)
        self.overall_epochs.append(overall_epoch)
        self.overall_train_loss.append(train_loss)
        self.overall_train_score.append(train_score)
        self.overall_val_loss.append(val_loss)
        self.overall_val_score.append(val_score)
        self.overall_learning_rate.append(float(lr))
        self.overall_grad_norm_mean.append(grad_norm_mean)
        self.overall_grad_norm_max.append(grad_norm_max)


@torch.no_grad()
def _clone_state_dict(module: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {
        name: param.detach().cpu().clone()
        for name, param in module.state_dict().items()
    }


class _LaggedPairDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Dataset yielding time-lagged pairs for a fixed ``tau``."""

    def __init__(self, sequences: Sequence[np.ndarray], tau: int) -> None:
        self._raw_sequences = [np.asarray(seq, dtype=np.float32) for seq in sequences]
        if not self._raw_sequences:
            raise ValueError("sequences must contain at least one array")
        if any(seq.ndim != 2 for seq in self._raw_sequences):
            raise ValueError("each sequence must be two-dimensional")
        self._tensors = [
            torch.as_tensor(seq, dtype=torch.float32) for seq in self._raw_sequences
        ]
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
    learning_rate: float = const.DEEPTICA_DEFAULT_LEARNING_RATE
    weight_decay: float = const.DEEPTICA_DEFAULT_WEIGHT_DECAY
    val_fraction: float = 0.2
    shuffle: bool = True
    num_workers: int = 0
    device: str = "auto"
    grad_clip_norm: Optional[float] = 1.0
    log_every: int = 1
    checkpoint_dir: Optional[Path] = None
    vamp_eps: float = const.DEEPTICA_DEFAULT_VAMP_EPS
    vamp_eps_abs: float = const.DEEPTICA_DEFAULT_VAMP_EPS_ABS
    vamp_alpha: float = 0.15
    vamp_cond_reg: float = const.DEEPTICA_DEFAULT_VAMP_COND_REG
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
            frac = min(
                max(frac, const.DEEPTICA_MIN_BATCH_FRACTION),
                const.DEEPTICA_MAX_BATCH_FRACTION,
            )
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
            eps=float(max(cfg.vamp_eps, const.DEEPTICA_MIN_VAMP_EPS)),
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

        train_arrays, val_arrays = self._prepare_train_val(sequences, val_sequences)
        val_loader = self._build_validation_loader(val_arrays)
        tau_blocks, total_epochs_planned = self._prepare_tau_blocks(train_arrays)
        self._initialize_scheduler(total_epochs_planned)
        start_time = time.time()
        metrics = self._run_curriculum(tau_blocks, val_loader)

        if self._best_state is not None:
            self.module.load_state_dict(self._best_state)

        history = self._build_history(metrics, start_time)
        self.grad_norm_curve = metrics.overall_grad_norm_mean

        csv_path = self._write_metrics_csv(history)
        if csv_path is not None:
            history["metrics_csv"] = str(csv_path)
        if self._best_checkpoint_path is not None:
            history["best_checkpoint"] = str(self._best_checkpoint_path)

        # Mark training as complete in the progress file
        self._finalize_realtime_metrics(history)

        self.history = history
        self._attach_history_to_model(history)
        return history

    def _prepare_train_val(
        self,
        sequences: Sequence[np.ndarray],
        val_sequences: Optional[Sequence[np.ndarray]],
    ) -> tuple[List[np.ndarray], List[np.ndarray]]:
        train_arrays = [np.asarray(seq, dtype=np.float32) for seq in sequences]
        if not train_arrays:
            raise ValueError("at least one training sequence is required")
        if any(arr.ndim != 2 for arr in train_arrays):
            raise ValueError("training sequences must all be 2-D arrays")
        if val_sequences is None:
            train_arrays, val_arrays = self._split_train_val(train_arrays)
        else:
            val_arrays = [np.asarray(seq, dtype=np.float32) for seq in val_sequences]
            if any(arr.ndim != 2 for arr in val_arrays):
                raise ValueError("validation sequences must all be 2-D arrays")
        if not val_arrays:
            raise ValueError("validation sequences are empty after splitting")
        return train_arrays, val_arrays

    def _build_validation_loader(
        self, val_arrays: Sequence[np.ndarray]
    ) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
        val_dataset = _LaggedPairDataset(val_arrays, self.cfg.val_tau)
        if len(val_dataset) == 0:
            raise ValueError(
                "validation dataset is empty – ensure val_tau is compatible with sequence lengths"
            )
        loader = self._build_loader(val_dataset, shuffle=False)
        if loader is None:
            raise ValueError("validation dataset could not be constructed")
        return loader

    def _prepare_tau_blocks(
        self, train_arrays: Sequence[np.ndarray]
    ) -> tuple[List[tuple[int, _LaggedPairDataset, Dict[str, object]]], int]:
        tau_blocks: List[tuple[int, _LaggedPairDataset, Dict[str, object]]] = []
        total_epochs_planned = 0
        for tau in self.cfg.tau_schedule:
            dataset = _LaggedPairDataset(train_arrays, tau)
            diag = dataset.diagnostics()
            tau_blocks.append((int(tau), dataset, diag))
            if len(dataset) > 0:
                total_epochs_planned += int(self.cfg.epochs_per_tau)
        return tau_blocks, total_epochs_planned

    def _run_curriculum(
        self,
        tau_blocks: Sequence[tuple[int, _LaggedPairDataset, Dict[str, object]]],
        val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    ) -> _CurriculumMetrics:
        metrics = _CurriculumMetrics()
        for tau, dataset, diag in tau_blocks:
            block = metrics.add_tau_block(int(tau), diag)
            if len(dataset) == 0:
                logger.warning(
                    "No lagged pairs available at tau=%d; skipping curriculum stage",
                    tau,
                )
                continue
            loader = self._build_loader(dataset, shuffle=self.cfg.shuffle)
            if loader is None:
                continue
            self._run_tau_stage(int(tau), loader, val_loader, block, metrics)
        return metrics

    def _run_tau_stage(
        self,
        tau: int,
        loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
        val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
        block: _TauBlock,
        metrics: _CurriculumMetrics,
    ) -> None:
        logger.info("Starting tau stage tau=%d (val_tau=%d)", tau, self.cfg.val_tau)
        for epoch_idx in range(int(self.cfg.epochs_per_tau)):
            current_lr = self._step_scheduler()
            train_metrics = self._train_one_epoch(loader)
            val_metrics = self._evaluate(val_loader)
            overall_epoch = metrics.next_epoch_index()
            metrics.record_epoch(
                block, overall_epoch, train_metrics, val_metrics, current_lr
            )
            self._record_condition_metrics(train_metrics)
            self._update_best(overall_epoch, tau, _metric_scalar(val_metrics, "score"))
            self._maybe_log_tau_progress(
                tau, epoch_idx, current_lr, train_metrics, val_metrics
            )
            # Write real-time progress metrics after each epoch
            self._write_realtime_metrics(
                overall_epoch, tau, train_metrics, val_metrics, current_lr
            )

    def _record_condition_metrics(self, train_metrics: MetricMapping) -> None:
        cond_c00 = _metric_scalar(train_metrics, "cond_c00")
        cond_ctt = _metric_scalar(train_metrics, "cond_ctt")
        self.cond_c00_curve.append(cond_c00)
        self.cond_ctt_curve.append(cond_ctt)
        eig0_min = _metric_scalar(train_metrics, "eig_c00_min", float("nan"))
        eig0_max = _metric_scalar(train_metrics, "eig_c00_max", float("nan"))
        eigt_min = _metric_scalar(train_metrics, "eig_ctt_min", float("nan"))
        eigt_max = _metric_scalar(train_metrics, "eig_ctt_max", float("nan"))
        self.c0_eig_min_curve.append(eig0_min)
        self.c0_eig_max_curve.append(eig0_max)
        self.ctt_eig_min_curve.append(eigt_min)
        self.ctt_eig_max_curve.append(eigt_max)
        self.var_z0_curve.append(_metric_vector(train_metrics, "var_z0"))
        self.var_zt_curve.append(_metric_vector(train_metrics, "var_zt"))
        self.mean_z0_curve.append(_metric_vector(train_metrics, "mean_z0"))
        self.mean_zt_curve.append(_metric_vector(train_metrics, "mean_zt"))
        if cond_c00 > const.DEEPTICA_CONDITION_NUMBER_WARN:
            logger.warning(
                "Condition number cond(C00)=%.3e exceeds stability threshold",
                cond_c00,
            )
        if cond_ctt > const.DEEPTICA_CONDITION_NUMBER_WARN:
            logger.warning(
                "Condition number cond(Ctt)=%.3e exceeds stability threshold",
                cond_ctt,
            )

    def _maybe_log_tau_progress(
        self,
        tau: int,
        epoch_idx: int,
        current_lr: float,
        train_metrics: MetricMapping,
        val_metrics: MetricMapping,
    ) -> None:
        log_every = int(self.cfg.log_every)
        if log_every <= 0:
            return
        epochs_per_tau = int(self.cfg.epochs_per_tau)
        epoch_num = epoch_idx + 1
        if epoch_num % log_every != 0 and epoch_num != epochs_per_tau:
            return
        # Enhanced logging with pre-clip gradient norms
        grad_preclip_mean = _metric_scalar(train_metrics, "grad_norm_preclip_mean", 0.0)
        grad_preclip_max = _metric_scalar(train_metrics, "grad_norm_preclip_max", 0.0)
        clip_threshold = (
            self.cfg.grad_clip_norm
            if self.cfg.grad_clip_norm is not None
            else float("inf")
        )

        logger.info(
            (
                "tau=%d val_tau=%d epoch=%d/%d lr=%.6e "
                "train_loss=%.6f val_score=%.6f "
                "grad_preclip=(%.3f/%.3f) grad_postclip=(%.3f/%.3f) clip_at=%.1f"
            ),
            tau,
            self.cfg.val_tau,
            epoch_num,
            epochs_per_tau,
            current_lr,
            _metric_scalar(train_metrics, "loss"),
            _metric_scalar(val_metrics, "score"),
            grad_preclip_mean,
            grad_preclip_max,
            _metric_scalar(train_metrics, "grad_norm_mean"),
            _metric_scalar(train_metrics, "grad_norm_max"),
            clip_threshold,
        )

    def _build_history(
        self, metrics: _CurriculumMetrics, start_time: float
    ) -> Dict[str, object]:
        history: Dict[str, object] = {
            "tau_schedule": [int(t) for t in self.cfg.tau_schedule],
            "val_tau": int(self.cfg.val_tau),
            "epochs_per_tau": int(self.cfg.epochs_per_tau),
            "loss_curve": list(metrics.overall_train_loss),
            "objective_curve": list(metrics.overall_train_score),
            "val_loss_curve": list(metrics.overall_val_loss),
            "val_score_curve": list(metrics.overall_val_score),
            "learning_rate_curve": list(metrics.overall_learning_rate),
            "grad_norm_mean_curve": list(metrics.overall_grad_norm_mean),
            "grad_norm_max_curve": list(metrics.overall_grad_norm_max),
            "epochs": list(metrics.overall_epochs),
            "per_tau": metrics.per_tau_blocks,
            "per_tau_objective_curve": {
                int(block["tau"]): block["val_score_curve"]
                for block in metrics.per_tau_blocks
            },
            "per_tau_learning_rate_curve": {
                int(block["tau"]): block["learning_rate_curve"]
                for block in metrics.per_tau_blocks
            },
            "per_tau_grad_norm_mean_curve": {
                int(block["tau"]): block["grad_norm_mean_curve"]
                for block in metrics.per_tau_blocks
            },
            "per_tau_grad_norm_max_curve": {
                int(block["tau"]): block["grad_norm_max_curve"]
                for block in metrics.per_tau_blocks
            },
            "pair_diagnostics": {
                int(block["tau"]): block["diagnostics"]
                for block in metrics.per_tau_blocks
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
            "grad_norm_curve": list(metrics.overall_grad_norm_mean),
            "best_val_score": (
                float(self._best_score) if self._best_score > float("-inf") else 0.0
            ),
            "best_epoch": int(self._best_epoch) if self._best_epoch >= 0 else None,
            "best_tau": int(self._best_tau) if self._best_tau >= 0 else None,
            "wall_time_s": float(max(0.0, time.time() - start_time)),
        }
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
    ) -> Dict[str, object]:
        self.module.train()
        accumulator = _EpochAccumulator()
        for outcome in self._iterate_training_batches(loader):
            accumulator.update(outcome)
        return cast(Dict[str, object], accumulator.finalize())

    def _iterate_training_batches(
        self, loader: DataLoader[tuple[torch.Tensor, torch.Tensor]]
    ) -> Iterator[_BatchOutcome]:
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

            # Compute gradient norm BEFORE clipping to get true gradient magnitude
            grad_norm_preclip = self._grad_norm(self._trainable_parameters)

            # Apply gradient clipping if configured
            if self.cfg.grad_clip_norm is not None and self._trainable_parameters:
                torch.nn.utils.clip_grad_norm_(
                    self._trainable_parameters, float(self.cfg.grad_clip_norm)
                )

            # Compute gradient norm AFTER clipping (for diagnostics)
            grad_norm = self._grad_norm(self._trainable_parameters)

            self.optimizer.step()
            metrics_obj = getattr(self.loss_fn, "latest_metrics", {})
            metrics = metrics_obj if isinstance(metrics_obj, dict) else {}
            yield _BatchOutcome(
                loss=float(loss.item()),
                score=float(score.item()),
                batch_size=batch_size,
                grad_norm=grad_norm,
                grad_norm_preclip=grad_norm_preclip,
                metrics=cast(Dict[str, object], metrics),
            )

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
                    "Skipping sequence with %d frames; requires at least %d",
                    n_frames,
                    max_tau + 2,
                )
                continue
            val_len = max(
                int(np.ceil(n_frames * float(self.cfg.val_fraction))), max_tau + 1
            )
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
            ensure_directory(ckpt_dir)
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

    def _write_realtime_metrics(
        self,
        epoch: int,
        tau: int,
        train_metrics: MetricMapping,
        val_metrics: MetricMapping,
        lr: float,
    ) -> None:
        """Write real-time training progress to JSON file for live monitoring."""
        if self.cfg.checkpoint_dir is None:
            return

        import json

        ckpt_dir = Path(self.cfg.checkpoint_dir)
        ensure_directory(ckpt_dir)
        progress_path = ckpt_dir / "training_progress.json"

        # Build current epoch data
        epoch_data = {
            "epoch": int(epoch),
            "tau": int(tau),
            "val_tau": int(self.cfg.val_tau),
            "train_loss": float(_metric_scalar(train_metrics, "loss")),
            "train_score": float(_metric_scalar(train_metrics, "score")),
            "val_loss": float(_metric_scalar(val_metrics, "loss")),
            "val_score": float(_metric_scalar(val_metrics, "score")),
            "learning_rate": float(lr),
            "grad_norm_mean": float(_metric_scalar(train_metrics, "grad_norm_mean")),
            "grad_norm_max": float(_metric_scalar(train_metrics, "grad_norm_max")),
            "best_val_score": (
                float(self._best_score) if self._best_score > float("-inf") else 0.0
            ),
            "best_epoch": int(self._best_epoch) if self._best_epoch >= 0 else None,
        }

        # Read existing data if available
        history_data = []
        if progress_path.exists():
            try:
                with progress_path.open("r") as f:
                    content = json.load(f)
                    if isinstance(content, dict) and "epochs" in content:
                        history_data = content["epochs"]
                    elif isinstance(content, list):
                        history_data = content
            except Exception:
                history_data = []

        # Append current epoch
        history_data.append(epoch_data)

        # Write updated progress
        progress = {
            "status": "training",
            "current_epoch": int(epoch),
            "total_epochs_planned": int(self._scheduler_total_epochs),
            "epochs": history_data,
        }

        with progress_path.open("w") as f:
            json.dump(progress, f, indent=2)

    def _finalize_realtime_metrics(self, history: Dict[str, object]) -> None:
        """Mark training as complete with final summary in progress file."""
        if self.cfg.checkpoint_dir is None:
            return

        import json

        ckpt_dir = Path(self.cfg.checkpoint_dir)
        progress_path = ckpt_dir / "training_progress.json"

        if not progress_path.exists():
            return

        try:
            with progress_path.open("r") as f:
                progress = json.load(f)
        except Exception:
            return

        # Update status to complete
        progress["status"] = "completed"
        progress["wall_time_s"] = _coerce_float(history.get("wall_time_s", 0.0))
        progress["best_val_score"] = _coerce_float(history.get("best_val_score", 0.0))
        progress["best_epoch"] = history.get("best_epoch")
        progress["best_tau"] = history.get("best_tau")

        with progress_path.open("w") as f:
            json.dump(progress, f, indent=2)

    def _write_metrics_csv(self, history: Dict[str, object]) -> Optional[Path]:
        if self.cfg.checkpoint_dir is None:
            return None
        ckpt_dir = Path(self.cfg.checkpoint_dir)
        ensure_directory(ckpt_dir)
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
            # per tau is a proper List of blocks, prevents int(object) overload
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
                            (
                                float(grad_mean_curve[idx])
                                if idx < len(grad_mean_curve)
                                else 0.0
                            ),
                            (
                                float(grad_max_curve[idx])
                                if idx < len(grad_max_curve)
                                else 0.0
                            ),
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
        resolved = spec.strip()
        if resolved.lower() == "auto":
            raise ValueError(
                "Automatic device selection is no longer supported; specify an explicit device"
            )
        if resolved.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA device requested but torch reports no available CUDA runtime"
            )
        return resolved


class _LossModule(Protocol):
    def __call__(
        self,
        z_t: torch.Tensor,
        z_tau: torch.Tensor,
        weights: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...


def _coerce_float(value: object, default: float = 0.0) -> float:
    if isinstance(value, numbers.Real):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return default
