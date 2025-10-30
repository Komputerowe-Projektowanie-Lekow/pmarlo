# ruff: noqa: D401
"""Chapman–Kolmogorov guardrail configuration and decision logic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Tuple

import numpy as np

Mode = Literal["absolute", "ess_adjusted"]


@dataclass(frozen=True)
class CKConfig:
    """Configuration for CK guardrail evaluation."""

    mode: Mode = "ess_adjusted"
    absolute: float = 0.15
    min_pass_fraction: float = 0.8
    per_lag_cap: float = 0.35
    k_steps: Tuple[int, ...] = (2, 3, 4)
    sigma_mult: float = 3.0


@dataclass(frozen=True)
class CKDecision:
    """Outcome of a CK guardrail check."""

    pass_fraction: float
    per_lag: Dict[int, Dict[str, float]]
    passed: bool
    reason: str


def ck_error(P_tau: np.ndarray, P_k_tau: np.ndarray, k: int) -> float:
    """Compute RMS matrix error between :math:`P(\tau)^k` and :math:`P(k\tau)`."""

    if P_tau.shape != P_k_tau.shape or P_tau.shape[0] != P_tau.shape[1]:
        raise ValueError(
            "P_tau and P_k_tau must be square matrices of identical shape."
        )
    P_comp = np.linalg.matrix_power(P_tau, k)
    diff = P_comp - P_k_tau
    return float(np.sqrt(np.mean(diff * diff)))


def _multinomial_rms_se(P: np.ndarray, counts: np.ndarray) -> float:
    """Approximate RMS sampling error using per-row multinomial standard errors."""

    n = P.shape[0]
    if counts.shape[0] != n:
        raise ValueError("counts length must equal number of states.")
    sesq_rows = []
    for i in range(n):
        Ni_raw = float(counts[i])
        Ni = Ni_raw if np.isfinite(Ni_raw) and Ni_raw > 0.0 else 1.0
        pi = P[i]
        sesq_rows.append(np.sum(pi * (1.0 - pi) / Ni) / n)
    return float(np.sqrt(np.mean(sesq_rows)))


def _ess_adjusted_threshold(noise_rms: float, cap: float, sigma_mult: float) -> float:
    """Compute ESS-adjusted CK threshold clipped by the configured cap."""

    return float(min(cap, sigma_mult * noise_rms))


def decide_ck(
    P_taus: Dict[int, np.ndarray],
    P_ktaus: Dict[int, np.ndarray],
    row_counts_by_lag: Dict[int, np.ndarray],
    cfg: CKConfig,
) -> CKDecision:
    """Evaluate CK guardrail compliance for the provided lag matrices."""

    per_lag: Dict[int, Dict[str, float]] = {}
    total = 0
    passes = 0

    for k in cfg.k_steps:
        if k not in P_taus or k not in P_ktaus or k not in row_counts_by_lag:
            continue
        total += 1
        P_tau = P_taus[k]
        Pk = P_ktaus[k]
        err = ck_error(P_tau, Pk, k)

        if cfg.mode == "absolute":
            thr = cfg.absolute
            noise = float("nan")
        elif cfg.mode == "ess_adjusted":
            noise = _multinomial_rms_se(Pk, row_counts_by_lag[k])
            thr = _ess_adjusted_threshold(noise, cfg.per_lag_cap, cfg.sigma_mult)
        else:  # pragma: no cover - guarded by type hints
            raise ValueError(f"Unknown CK mode: {cfg.mode}")

        ok = err <= thr
        if ok:
            passes += 1
        per_lag[k] = {
            "error": err,
            "threshold": thr,
            "noise_rms": noise,
            "pass": float(ok),
        }

    pass_fraction = (passes / total) if total > 0 else 0.0
    passed = pass_fraction >= cfg.min_pass_fraction
    reason = (
        f"CK guardrail {'PASSED' if passed else 'FAILED'}: "
        f"{passes}/{total} lags within threshold (pass_fraction={pass_fraction:.2f}, "
        f"mode={cfg.mode}, cap={cfg.per_lag_cap})."
    )
    return CKDecision(
        pass_fraction=pass_fraction, per_lag=per_lag, passed=passed, reason=reason
    )
