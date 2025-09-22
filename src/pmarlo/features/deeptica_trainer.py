"""DeepTICA trainer integrating VAMP-2 optimisation and curriculum support."""

from __future__ import annotations

import importlib.util
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

__all__ = ["TrainerConfig", "DeepTICATrainer"]

logger = logging.getLogger(__name__)

_TORCH_SPEC = importlib.util.find_spec("torch")
_MLCOLVAR_SPEC = importlib.util.find_spec("mlcolvar")

if _TORCH_SPEC is not None and _MLCOLVAR_SPEC is not None:  # pragma: no cover
    import torch
    from torch.nn.utils import clip_grad_norm_, clip_grad_value_
else:  # pragma: no cover - executed in minimal test environment
    torch = None  # type: ignore[assignment]


@dataclass(frozen=True)
class TrainerConfig:
    tau_steps: int
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    use_weights: bool = True
    tau_schedule: Tuple[int, ...] = ()
    grad_clip_norm: Optional[float] = 1.0
    grad_clip_mode: str = "norm"
    grad_clip_value: Optional[float] = None
    grad_norm_warn: Optional[float] = None
    log_every: int = 25
    checkpoint_dir: Optional[Path] = None
    checkpoint_metric: str = "vamp2"
    device: str = "auto"
    scheduler: str = "none"  # "none" | "cosine"
    scheduler_warmup_steps: int = 0
    scheduler_total_steps: Optional[int] = None
    max_steps: Optional[int] = None
    vamp_eps: float = 1e-3
    vamp_eps_abs: float = 1e-6
    vamp_alpha: float = 0.15
    vamp_cond_reg: float = 1e-4


if torch is None:

    class DeepTICATrainer:
        """Placeholder trainer used when PyTorch is not available."""

        def __init__(self, model: object, cfg: TrainerConfig) -> None:
            if cfg.tau_steps <= 0:
                raise ValueError("tau_steps must be positive")
            self.model = model
            self.cfg = cfg
            logger.warning(
                "DeepTICA trainer instantiated without PyTorch; only stub behaviour is available."
            )

        def step(
            self,
            batch: List[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]],
        ) -> Dict[str, float]:
            raise NotImplementedError(
                "DeepTICA training requires PyTorch; install pmarlo[mlcv] to enable"
            )

        def evaluate(
            self,
            batch: List[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]],
        ) -> Dict[str, float]:
            raise NotImplementedError(
                "DeepTICA evaluation requires PyTorch; install pmarlo[mlcv] to enable"
            )

else:

    from .deeptica.losses import VAMP2Loss

    class DeepTICATrainer:
        """Optimises a DeepTICA model using a tau-curriculum and VAMP-2 loss."""

        def __init__(self, model, cfg: TrainerConfig) -> None:
            if cfg.tau_steps <= 0:
                raise ValueError("tau_steps must be positive")
            self.model = model
            self.cfg = cfg
            self._stub_mode = not hasattr(self.model, "net")

            device_str = self._resolve_device(cfg.device)
            self.device = torch.device(device_str)

            if self._stub_mode:
                logger.warning(
                    "DeepTICA trainer received model without 'net'; stub behaviour only."
                )
                self.loss_module = None
                self.optimizer = None
                self.scheduler = None
                self.curriculum = (
                    list(cfg.tau_schedule) if cfg.tau_schedule else [int(cfg.tau_steps)]
                )
                self.curriculum_index = 0
                self.history = []
                self.global_step = 0
                self.best_score = float("-inf")
                self.checkpoint_dir = None
                self.best_checkpoint_path = None
                return

            self.model.net.to(self.device)
            self.model.net.train()
            self.loss_module = VAMP2Loss(
                eps=float(max(1e-9, cfg.vamp_eps)),
                eps_abs=float(max(0.0, cfg.vamp_eps_abs)),
                alpha=float(min(max(cfg.vamp_alpha, 0.0), 1.0)),
                cond_reg=float(max(0.0, cfg.vamp_cond_reg)),
            ).to(self.device)

            weight_decay = float(cfg.weight_decay)
            if weight_decay <= 0.0:
                weight_decay = 1e-4
            self.optimizer = torch.optim.AdamW(
                self.model.net.parameters(),
                lr=float(cfg.learning_rate),
                weight_decay=weight_decay,
            )

            self.scheduler = self._make_scheduler()

            self.curriculum: List[int] = (
                list(cfg.tau_schedule) if cfg.tau_schedule else [int(cfg.tau_steps)]
            )
            self.curriculum_index = 0

            self.history: List[Dict[str, float]] = []
            self.global_step: int = 0
            self.best_score: float = float("-inf")
            self.checkpoint_dir = (
                Path(cfg.checkpoint_dir) if cfg.checkpoint_dir else None
            )
            if self.checkpoint_dir:
                self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
                self.best_checkpoint_path = self.checkpoint_dir / "best_deeptica.pt"
            else:
                self.best_checkpoint_path = None

        # ------------------------------------------------------------------
        # Public API
        # ------------------------------------------------------------------
        def current_tau(self) -> int:
            return int(self.curriculum[self.curriculum_index])

        def advance_tau(self) -> bool:
            if self.curriculum_index + 1 >= len(self.curriculum):
                return False
            self.curriculum_index += 1
            self.best_score = float("-inf")
            logger.info("Advanced tau curriculum to %s", self.current_tau())
            return True

        def step(
            self,
            batch: List[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]],
        ) -> Dict[str, float]:
            if getattr(self, "_stub_mode", False):
                raise NotImplementedError(
                    "DeepTICA training requires a model exposing a 'net' attribute"
                )
            tensors = self._prepare_batch(batch)
            if tensors is None:
                logger.debug("Received empty batch; skipping optimisation step")
                return {"loss": 0.0, "vamp2": 0.0, "tau": float(self.current_tau())}
            x_t, x_tau, weights = tensors

            self.model.net.train()
            self.optimizer.zero_grad()
            loss, score = self._compute_loss_and_score(x_t, x_tau, weights)
            loss.backward()

            grad_norm = self._compute_grad_norm(self.model.net.parameters())

            clip_mode = str(getattr(self.cfg, "grad_clip_mode", "norm")).lower()
            if clip_mode == "value":
                clip_value = getattr(self.cfg, "grad_clip_value", None)
                if clip_value is not None:
                    clip_grad_value_(
                        self.model.net.parameters(), float(clip_value)
                    )
            else:
                if self.cfg.grad_clip_norm is not None:
                    grad_norm = float(
                        clip_grad_norm_(
                            self.model.net.parameters(),
                            max_norm=float(self.cfg.grad_clip_norm),
                        )
                    )

            warn_thresh = getattr(self.cfg, "grad_norm_warn", None)
            if warn_thresh is not None and grad_norm is not None:
                if float(grad_norm) > float(warn_thresh):
                    logger.warning(
                        "Gradient norm %.3f exceeded warning threshold %.3f",
                        float(grad_norm),
                        float(warn_thresh),
                    )
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

            metrics = {
                "loss": float(loss.item()),
                "vamp2": float(score.item()),
                "tau": float(self.current_tau()),
                "lr": float(self.optimizer.param_groups[0]["lr"]),
            }
            if grad_norm is not None:
                metrics["grad_norm"] = float(grad_norm)
            self._record_metrics(metrics)
            self._maybe_checkpoint(metrics)
            self.global_step += 1

            if self.global_step % max(1, self.cfg.log_every) == 0:
                logger.info(
                    "DeepTICA step=%d tau=%s loss=%.6f vamp2=%.6f",
                    self.global_step,
                    self.current_tau(),
                    metrics["loss"],
                    metrics["vamp2"],
                )
            return metrics

        def evaluate(
            self,
            batch: List[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]],
        ) -> Dict[str, float]:
            if getattr(self, "_stub_mode", False):
                raise NotImplementedError(
                    "DeepTICA evaluation requires a model exposing a 'net' attribute"
                )
            tensors = self._prepare_batch(batch)
            if tensors is None:
                return {"loss": 0.0, "vamp2": 0.0, "tau": float(self.current_tau())}
            x_t, x_tau, weights = tensors
            self.model.net.eval()
            with torch.no_grad():
                loss, score = self._compute_loss_and_score(x_t, x_tau, weights)
            return {
                "loss": float(loss.item()),
                "vamp2": float(score.item()),
                "tau": float(self.current_tau()),
            }

        # ------------------------------------------------------------------
        # Internal helpers
        # ------------------------------------------------------------------
        def _resolve_device(self, device_spec: str) -> str:
            if device_spec.lower() == "auto":
                return "cuda" if torch.cuda.is_available() else "cpu"
            return device_spec

        def _make_scheduler(self):
            if self.cfg.scheduler.lower() != "cosine":
                return None
            warmup = int(max(0, self.cfg.scheduler_warmup_steps))
            total_steps = self.cfg.scheduler_total_steps
            if total_steps is None:
                raise ValueError(
                    "scheduler_total_steps must be provided when scheduler='cosine'"
                )
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=max(1, int(total_steps)),
                eta_min=0.0,
            )

        def _prepare_batch(
            self,
            batch: List[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]],
        ) -> Optional[Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]]:
            if not batch:
                return None

            x_t = []
            x_tau = []
            weights = [] if self.cfg.use_weights else None
            for item in batch:
                try:
                    x0, x1, w = item
                except ValueError:
                    logger.debug("Skipping malformed batch item: %s", item)
                    continue
                x_t.append(torch.from_numpy(np.asarray(x0)).float())
                x_tau.append(torch.from_numpy(np.asarray(x1)).float())
                if weights is not None:
                    weights.append(
                        torch.from_numpy(
                            np.asarray(np.ones_like(x0) if w is None else w)
                        ).float()
                    )

            if not x_t:
                return None

            x_t_cat = torch.cat(x_t, dim=0).to(self.device)
            x_tau_cat = torch.cat(x_tau, dim=0).to(self.device)
            w_cat = (
                torch.cat(weights, dim=0).to(self.device) if weights is not None else None
            )
            return x_t_cat, x_tau_cat, w_cat

        def _compute_loss_and_score(
            self,
            x_t: torch.Tensor,
            x_tau: torch.Tensor,
            weights: Optional[torch.Tensor],
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            if weights is not None and weights.shape != x_t.shape:
                weights = weights.expand_as(x_t)
            z_t = self.model.net(x_t)
            z_tau = self.model.net(x_tau)
            loss, score = self.loss_module(z_t, z_tau, weights)
            return loss, score

        def _compute_grad_norm(self, parameters: Iterable[torch.nn.Parameter]):
            total_norm = 0.0
            for p in parameters:
                if p.grad is None:
                    continue
                param_norm = p.grad.data.norm(2).item()
                total_norm += param_norm**2
            return float(total_norm**0.5)

        def _record_metrics(self, metrics: Dict[str, float]) -> None:
            self.history.append(dict(metrics))

        def _maybe_checkpoint(self, metrics: Dict[str, float]) -> None:
            if getattr(self, "_stub_mode", False):
                return
            if self.best_checkpoint_path is None:
                return
            metric_name = str(self.cfg.checkpoint_metric)
            score = metrics.get(metric_name)
            if score is None:
                return
            if float(score) > self.best_score:
                self.best_score = float(score)
                torch.save(self.model.net.state_dict(), self.best_checkpoint_path)

