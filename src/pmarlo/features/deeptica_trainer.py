from __future__ import annotations

"""DeepTICA trainer integrating VAMP-2 optimisation and curriculum support."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

__all__ = ["TrainerConfig", "DeepTICATrainer"]


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainerConfig:
    tau_steps: int
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    use_weights: bool = True
    tau_schedule: Tuple[int, ...] = ()
    grad_clip_norm: Optional[float] = 1.0
    log_every: int = 25
    checkpoint_dir: Optional[Path] = None
    checkpoint_metric: str = "vamp2"
    device: str = "auto"
    scheduler: str = "none"  # "none" | "cosine"


class DeepTICATrainer:
    """Optimises a DeepTICA model using a tau-curriculum and VAMP-2 loss."""

    def __init__(self, model, cfg: TrainerConfig) -> None:
        if cfg.tau_steps <= 0:
            raise ValueError("tau_steps must be positive")
        self.model = model
        self.cfg = cfg

        device_str = self._resolve_device(cfg.device)
        self.device = torch.device(device_str)
        self.model.net.to(self.device)
        self.model.net.train()

        self.optimizer = torch.optim.AdamW(
            self.model.net.parameters(),
            lr=float(cfg.learning_rate),
            weight_decay=float(cfg.weight_decay),
        )

        self.scheduler = self._make_scheduler()

        self.curriculum: List[int] = (
            list(cfg.tau_schedule) if cfg.tau_schedule else [int(cfg.tau_steps)]
        )
        self.curriculum_index = 0

        self.history: List[Dict[str, float]] = []
        self.global_step: int = 0
        self.best_score: float = float("-inf")
        self.checkpoint_dir = Path(cfg.checkpoint_dir) if cfg.checkpoint_dir else None
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
        tensors = self._prepare_batch(batch)
        if tensors is None:
            logger.debug("Received empty batch; skipping optimisation step")
            return {"loss": 0.0, "vamp2": 0.0, "tau": float(self.current_tau())}
        x_t, x_tau, weights = tensors

        self.model.net.train()
        self.optimizer.zero_grad()
        loss, score = self._compute_loss_and_score(x_t, x_tau, weights)
        loss.backward()
        if self.cfg.grad_clip_norm is not None:
            clip_grad_norm_(
                self.model.net.parameters(), max_norm=float(self.cfg.grad_clip_norm)
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
        t_max = max(1, int(1000))
        return torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=t_max)

    def _prepare_batch(
        self,
        batch: List[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]],
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        x_parts: List[torch.Tensor] = []
        y_parts: List[torch.Tensor] = []
        w_parts: List[torch.Tensor] = []

        for x_t, x_tau, weights in batch:
            x_arr = torch.as_tensor(x_t, dtype=torch.float32, device=self.device)
            y_arr = torch.as_tensor(x_tau, dtype=torch.float32, device=self.device)
            if x_arr.ndim != 2 or y_arr.ndim != 2:
                raise ValueError("Batch entries must be 2-D arrays")
            if x_arr.shape != y_arr.shape:
                raise ValueError("x_t and x_tau must have matching shapes")
            x_parts.append(x_arr)
            y_parts.append(y_arr)

            if self.cfg.use_weights:
                if weights is None:
                    w = torch.ones(x_arr.shape[0], dtype=torch.float32, device=self.device)
                else:
                    w = torch.as_tensor(weights, dtype=torch.float32, device=self.device)
                    if w.ndim != 1 or w.shape[0] != x_arr.shape[0]:
                        raise ValueError("weights must be 1-D and align with batch length")
                w_parts.append(w)

        if not x_parts:
            return None

        x_cat = torch.cat(x_parts, dim=0)
        y_cat = torch.cat(y_parts, dim=0)

        if self.cfg.use_weights:
            w_cat = torch.cat(w_parts, dim=0)
        else:
            w_cat = torch.ones(x_cat.shape[0], dtype=torch.float32, device=self.device)

        total = torch.clamp(w_cat.sum(), min=1e-12)
        w_cat = (w_cat / total).to(torch.float32)
        return x_cat, y_cat, w_cat

    def _compute_loss_and_score(
        self,
        x_t: torch.Tensor,
        x_tau: torch.Tensor,
        weights: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        out_t = self.model.net(x_t)
        out_tau = self.model.net(x_tau)
        score = self._vamp2_score(out_t, out_tau, weights)
        loss = -score
        return loss, score.detach()

    def _vamp2_score(
        self,
        x_t: torch.Tensor,
        x_tau: torch.Tensor,
        weights: torch.Tensor,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        weights = weights.reshape(-1, 1)
        weights = torch.clamp(weights, min=1e-12)
        mean_t = torch.sum(x_t * weights, dim=0, keepdim=True)
        mean_tau = torch.sum(x_tau * weights, dim=0, keepdim=True)
        x_center = x_t - mean_t
        y_center = x_tau - mean_tau

        w_sqrt = torch.sqrt(weights)
        x_weighted = x_center * w_sqrt
        y_weighted = y_center * w_sqrt

        dim = x_t.shape[1]
        eye = torch.eye(dim, device=self.device, dtype=torch.float32)
        C00 = x_weighted.T @ x_weighted + eps * eye
        Ctt = y_weighted.T @ y_weighted + eps * eye
        C0tau = x_weighted.T @ y_weighted

        L0 = torch.linalg.cholesky(C00)
        Lt = torch.linalg.cholesky(Ctt)
        S = torch.linalg.solve(L0, C0tau)
        S = torch.linalg.solve(Lt, S.T).T

        svals = torch.linalg.svdvals(S)
        k = min(int(getattr(self.model.cfg, "n_out", svals.shape[0])), svals.shape[0])
        return torch.sum((svals[:k]) ** 2)

    def _record_metrics(self, metrics: Dict[str, float]) -> None:
        self.history.append(metrics)
        hist = self.model.training_history
        hist.setdefault("steps", []).append(metrics)

    def _maybe_checkpoint(self, metrics: Dict[str, float]) -> None:
        score = metrics.get(self.cfg.checkpoint_metric)
        if score is None or score <= self.best_score:
            return
        self.best_score = float(score)
        if not self.best_checkpoint_path:
            return
        ckpt = {
            "model_state": self.model.net.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "step": self.global_step,
            "tau": self.current_tau(),
            "score": self.best_score,
        }
        torch.save(ckpt, self.best_checkpoint_path)
        logger.info(
            "Saved DeepTICA checkpoint to %s (score %.6f)",
            self.best_checkpoint_path,
            self.best_score,
        )
