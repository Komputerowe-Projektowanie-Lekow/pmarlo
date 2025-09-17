from __future__ import annotations

"""Numerically stable VAMP-2 loss utilities for Deep-TICA training."""

from typing import Optional, Tuple

import torch
from torch import Tensor, nn


class VAMP2Loss(nn.Module):
    """Compute a scale-invariant, regularised VAMP-2 score."""

    def __init__(self, eps: float = 1e-6, dtype: torch.dtype = torch.float64) -> None:
        super().__init__()
        self.eps = float(eps)
        self.target_dtype = dtype
        self.register_buffer("_eye", torch.empty(0, dtype=dtype), persistent=False)

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
            w = torch.full((z0.shape[0], 1), 1.0 / float(z0.shape[0]), device=device, dtype=dtype)
        else:
            w = weights.reshape(-1, 1).to(device=device, dtype=dtype)
            w = torch.clamp(w, min=0.0)
            total = torch.clamp(w.sum(), min=1e-12)
            w = w / total

        mean0 = torch.sum(z0 * w, dim=0, keepdim=True)
        meant = torch.sum(zt * w, dim=0, keepdim=True)
        z0_c = z0 - mean0
        zt_c = zt - meant

        C00 = z0_c.T @ (z0_c * w)
        Ctt = zt_c.T @ (zt_c * w)
        C0t = z0_c.T @ (zt_c * w)

        eye = self._identity_like(C00, device)
        C00 = C00 + self.eps * eye
        Ctt = Ctt + self.eps * eye

        L0 = torch.linalg.cholesky(C00, upper=False)
        Lt = torch.linalg.cholesky(Ctt, upper=False)

        try:
            left = torch.linalg.solve_triangular(L0, C0t, upper=False)
            right = torch.linalg.solve_triangular(Lt, left.transpose(-1, -2), upper=False)
        except AttributeError:  # pragma: no cover - legacy torch fallback
            left = torch.triangular_solve(C0t, L0, upper=False)[0]
            right = torch.triangular_solve(left.transpose(-1, -2), Lt, upper=False)[0]
        K = right.transpose(-1, -2)

        score = torch.sum(K * K)
        loss = -score
        return loss, score.detach()

    def _identity_like(self, mat: Tensor, device: torch.device) -> Tensor:
        dim = mat.shape[-1]
        eye = self._eye
        if eye.numel() != dim * dim or eye.device != device:
            eye = torch.eye(dim, device=device, dtype=self.target_dtype)
            self._eye = eye
        return eye
