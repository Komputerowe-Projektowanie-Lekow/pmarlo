from __future__ import annotations

"""Numerically stable VAMP-2 loss utilities for Deep-TICA training."""

from typing import Optional, Tuple

import torch
from torch import Tensor, nn

from pmarlo import constants as const


class VAMP2Loss(nn.Module):
    """Compute a scale-invariant, regularised VAMP-2 score."""

    def __init__(
        self,
        eps: float = const.DEEPTICA_DEFAULT_VAMP_EPS,
        *,
        eps_abs: float = const.DEEPTICA_DEFAULT_VAMP_EPS_ABS,
        alpha: float = 0.15,
        cond_reg: float = const.DEEPTICA_DEFAULT_VAMP_COND_REG,
        jitter: float = const.DEEPTICA_DEFAULT_JITTER,
        max_cholesky_retries: int = 5,
        jitter_growth: float = 10.0,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        self.eps = float(eps)
        self.eps_abs = float(max(eps_abs, 0.0))
        self.alpha = float(min(max(alpha, 0.0), 1.0))
        self.cond_reg = float(max(cond_reg, 0.0))
        self.jitter = float(max(jitter, 0.0))
        self.jitter_growth = float(max(jitter_growth, 1.0))
        self.max_cholesky_retries = int(max(1, max_cholesky_retries))
        self.target_dtype = dtype
        self.register_buffer("_eye", torch.empty(0, dtype=dtype), persistent=False)
        self._latest_metrics: dict[str, float | list[float]] = {}

    def forward(
        self,
        z0: Tensor,
        zt: Tensor,
        weights: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        if z0.ndim != 2 or zt.ndim != 2:
            raise ValueError("VAMP2Loss expects 2-D activations")
        if z0.shape != zt.shape:
            raise ValueError("z0 and zt must share the same shape")
        if z0.shape[0] == 0:
            raise ValueError("VAMP2Loss received empty batch")

        device = z0.device
        dtype = self.target_dtype
        z0 = z0.to(dtype=dtype)
        zt = zt.to(dtype=dtype)

        if weights is None:
            w = torch.full(
                (z0.shape[0],), 1.0 / float(z0.shape[0]), device=device, dtype=dtype
            )
        else:
            w = weights.reshape(-1).to(device=device, dtype=dtype)
            w = torch.clamp(w, min=0.0)
            total = torch.clamp(w.sum(), min=const.NUMERIC_MIN_POSITIVE)
            w = w / total

        w_col = w.reshape(-1, 1)
        mean0 = torch.sum(z0 * w_col, dim=0, keepdim=True)
        meant = torch.sum(zt * w_col, dim=0, keepdim=True)

        stacked = torch.cat((z0, zt), dim=1)
        cov = torch.cov(stacked.T, correction=0, aweights=w)
        dim = z0.shape[-1]
        C00 = cov[:dim, :dim]
        Ctt = cov[dim:, dim:]
        C0t = cov[:dim, dim:]

        diag_c00 = torch.diagonal(C00, dim1=-2, dim2=-1)
        diag_ctt = torch.diagonal(Ctt, dim1=-2, dim2=-1)

        eye = self._identity_like(C00, device)
        dim = C00.shape[-1]
        trace_floor = torch.tensor(
            const.NUMERIC_MIN_POSITIVE, device=device, dtype=dtype
        )
        tr0 = torch.clamp(torch.trace(C00), min=trace_floor)
        trt = torch.clamp(torch.trace(Ctt), min=trace_floor)
        mu0 = tr0 / float(max(1, dim))
        mut = trt / float(max(1, dim))
        diag_mean0 = torch.clamp(
            torch.diagonal(C00, dim1=-2, dim2=-1).mean(), min=trace_floor
        )
        diag_meant = torch.clamp(
            torch.diagonal(Ctt, dim1=-2, dim2=-1).mean(), min=trace_floor
        )
        ridge0 = torch.maximum(mu0 * self.eps, diag_mean0 * self.eps_abs)
        ridget = torch.maximum(mut * self.eps, diag_meant * self.eps_abs)
        alpha = self.alpha
        C00 = (1.0 - alpha) * C00 + (alpha * mu0 + ridge0) * eye
        Ctt = (1.0 - alpha) * Ctt + (alpha * mut + ridget) * eye
        C00 = (C00 + C00.T) * 0.5
        Ctt = (Ctt + Ctt.T) * 0.5

        eig0 = torch.linalg.eigvalsh(C00)
        eigt = torch.linalg.eigvalsh(Ctt)

        min0 = torch.clamp(eig0.min(), min=trace_floor)
        mint = torch.clamp(eigt.min(), min=trace_floor)
        max0 = torch.clamp(eig0.max(), min=trace_floor)
        maxt = torch.clamp(eigt.max(), min=trace_floor)
        cond_c00 = max0 / min0
        cond_ctt = maxt / mint

        L0, C00 = self._stable_cholesky(C00, eye)
        Lt, Ctt = self._stable_cholesky(Ctt, eye)

        left = torch.linalg.solve_triangular(L0, C0t, upper=False)
        right = torch.linalg.solve_triangular(
            Lt, left.transpose(-1, -2), upper=False
        )
        K = right.transpose(-1, -2)

        score = torch.sum(K * K)

        penalty = torch.tensor(0.0, device=device, dtype=dtype)
        if self.cond_reg > 0.0:
            cond_term = torch.log(torch.clamp(cond_c00, min=1.0)) + torch.log(
                torch.clamp(cond_ctt, min=1.0)
            )
            penalty = penalty + self.cond_reg * cond_term

        loss = -score + penalty
        self._latest_metrics = {
            "cond_C00": float(cond_c00.detach().cpu().item()),
            "cond_Ctt": float(cond_ctt.detach().cpu().item()),
            "var_z0": [float(x) for x in diag_c00.detach().cpu().tolist()],
            "var_zt": [float(x) for x in diag_ctt.detach().cpu().tolist()],
            "mean_z0": [float(x) for x in mean0.detach().cpu().reshape(-1).tolist()],
            "mean_zt": [float(x) for x in meant.detach().cpu().reshape(-1).tolist()],
            "eig_C00_min": float(min0.detach().cpu().item()),
            "eig_C00_max": float(max0.detach().cpu().item()),
            "eig_Ctt_min": float(mint.detach().cpu().item()),
            "eig_Ctt_max": float(maxt.detach().cpu().item()),
        }
        return loss, score.detach()

    @property
    def latest_metrics(self) -> dict[str, float | list[float]]:
        return dict(self._latest_metrics)

    def _identity_like(self, mat: Tensor, device: torch.device) -> Tensor:
        dim = mat.shape[-1]
        eye = self._eye  # type: ignore[has-type]
        if eye.numel() != dim * dim or eye.device != device:
            eye = torch.eye(dim, device=device, dtype=self.target_dtype)
            self._eye = eye
        return eye

    def _stable_cholesky(self, mat: Tensor, eye: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute a Cholesky factor with adaptive jitter for stability."""

        base = mat
        total_jitter = 0.0
        jitter = self.jitter

        for attempt in range(self.max_cholesky_retries):
            if total_jitter > 0.0:
                updated = base + eye * total_jitter
            else:
                updated = base

            chol, info = torch.linalg.cholesky_ex(updated, upper=False)

            if info.ndim == 0:
                success = int(info.item()) == 0
            else:
                success = not torch.any(info > 0)

            if success:
                return chol, updated

            if attempt + 1 >= self.max_cholesky_retries:
                break

            step = jitter if jitter > 0.0 else const.NUMERIC_MIN_POSITIVE
            total_jitter += step
            jitter = max(jitter, const.NUMERIC_MIN_POSITIVE) * self.jitter_growth

        raise RuntimeError("Cholesky decomposition failed after retries")
