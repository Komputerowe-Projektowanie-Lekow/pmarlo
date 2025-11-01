import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.special import polygamma


def beta_to_kT(beta: float) -> float:
    """Convert inverse temperature beta=1/(kT) to kT."""
    if beta <= 0:
        raise ValueError("beta must be > 0")
    return 1.0 / beta


def fes_uncertainty_sd_kT(
    bin_counts: np.ndarray,
    alpha: float = 1e-6,
    kT: float = 1.0,
) -> np.ndarray:
    """
    SD[F] where F = -kT * log p_i and p has a Dirichlet posterior with counts n_i and pseudocount alpha.
    SD[log p_i] = sqrt(trigamma(n_i+alpha) + trigamma(N + alpha_0)),
    so SD[F] = kT * SD[log p_i].
    """
    n = np.asarray(bin_counts, dtype=float)
    if np.any(n < 0):
        raise ValueError("bin_counts must be non-negative")
    N = float(n.sum())
    K = n.size
    a0 = alpha * K
    tri_bins = polygamma(1, n + alpha)
    tri_tot = polygamma(1, N + a0)
    sd_logp = np.sqrt(tri_bins + tri_tot)
    return kT * sd_logp


def mark_bins_for_smoothing(
    bin_counts: np.ndarray,
    target_sd_kT: float = 0.5,
    alpha: float = 1e-6,
    kT: float = 1.0,
):
    """
    Returns (mask, sd_map):
      mask[i]=True iff SD[F_i] > target_sd_kT.
    """
    sd = fes_uncertainty_sd_kT(bin_counts, alpha=alpha, kT=kT)
    return (sd > float(target_sd_kT)), sd


def adaptive_bandwidth(
    ess_map: np.ndarray,
    h0: float = 1.2,
    ess_ref: float = 50.0,
    h_min: float = 0.4,
    h_max: float = 3.0,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    h = h0 * sqrt(ess_ref / max(ESS_i, eps)), clipped to [h_min, h_max].
    """
    ess = np.maximum(np.asarray(ess_map, dtype=float), eps)
    h = h0 * np.sqrt(ess_ref / ess)
    return np.clip(h, h_min, h_max)


def _blend_two(
    F: np.ndarray, F_lo: np.ndarray, F_hi: np.ndarray, w: np.ndarray
) -> np.ndarray:
    """Linear blend per element: (1-w)*F_lo + w*F_hi, where w in [0,1]."""
    return (1.0 - w) * F_lo + w * F_hi


def smooth_F_with_adaptive_gaussian(
    F: np.ndarray,
    h_map: np.ndarray,
    apply_mask: np.ndarray | None = None,
    sigma_grid: tuple[float, ...] = (0.5, 1.0, 2.0, 3.0),
) -> np.ndarray:
    """
    Approximate per-bin sigma by blending a small bank of global Gaussian blurs.
    - Compute blurred versions at a few sigma values.
    - For each cell, linearly interpolate between the two nearest sigmas using h_map.
    - If apply_mask is provided, only replace where mask==True; elsewhere keep F.
    """
    F = np.asarray(F, dtype=float)
    h_map = np.asarray(h_map, dtype=float)
    if F.shape != h_map.shape:
        raise ValueError("F and h_map must have the same shape")
    sigmas = tuple(float(s) for s in sigma_grid)
    bank = [gaussian_filter(F, sigma=s, mode="nearest") for s in sigmas]

    h = np.clip(h_map, sigmas[0], sigmas[-1])
    idx_hi = np.searchsorted(sigmas, h, side="right")
    idx_hi = np.clip(idx_hi, 1, len(sigmas) - 1)
    idx_lo = idx_hi - 1
    s_lo = np.take(sigmas, idx_lo)
    s_hi = np.take(sigmas, idx_hi)
    w = (h - s_lo) / np.maximum(s_hi - s_lo, 1e-12)

    bank_stack = np.stack(bank)
    F_lo = np.take(bank_stack, idx_lo, axis=0)
    F_hi = np.take(bank_stack, idx_hi, axis=0)

    F_out = _blend_two(F, F_lo, F_hi, w)
    if apply_mask is not None:
        F_mix = F.copy()
        F_mix = np.where(apply_mask, F_out, F_mix)
        return F_mix
    return F_out
