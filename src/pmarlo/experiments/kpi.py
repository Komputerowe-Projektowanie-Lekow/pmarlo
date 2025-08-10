"""
KPI utilities for PMARLO experiments.

This module defines domain-specific KPIs and helpers to compute and persist
benchmark JSON files for experiments. The JSON layout is fixed-order to ensure
stable downstream parsing and comparisons across runs.
"""

from __future__ import annotations

import json
import math
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _ordered(obj: Dict[str, Any]) -> "OrderedDict[str, Any]":
    """Create an OrderedDict preserving the current insertion order."""
    return OrderedDict(obj.items())


def default_kpi_metrics(
    conformational_coverage: Optional[float] = None,
    transition_matrix_accuracy: Optional[float] = None,
    replica_exchange_success_rate: Optional[float] = None,
    runtime_seconds: Optional[float] = None,
    memory_mb: Optional[float] = None,
) -> "OrderedDict[str, Optional[float]]":
    """
    Construct KPI metrics dictionary in the required key order.
    """
    return _ordered(
        {
            "conformational_coverage": _finite_or_none(conformational_coverage),
            "transition_matrix_accuracy": _finite_or_none(transition_matrix_accuracy),
            "replica_exchange_success_rate": _finite_or_none(
                replica_exchange_success_rate
            ),
            "runtime_seconds": _finite_or_none(runtime_seconds),
            "memory_mb": _finite_or_none(memory_mb),
        }
    )


def build_benchmark_record(
    *,
    algorithm: str,
    experiment_id: str,
    input_parameters: Dict[str, Any],
    kpi_metrics: Dict[str, Optional[float]],
    notes: str = "",
    errors: Optional[List[str]] = None,
) -> "OrderedDict[str, Any]":
    """
    Build a benchmark record with fixed top-level field order.
    """
    return _ordered(
        {
            "algorithm": algorithm,
            "experiment_id": experiment_id,
            "input_parameters": input_parameters,
            "kpi_metrics": kpi_metrics,
            "notes": notes,
            "errors": errors or [],
        }
    )


def write_benchmark_json(
    output_dir: str | Path, record: "OrderedDict[str, Any]"
) -> Path:
    """
    Persist a benchmark record to benchmark.json under the run directory.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "benchmark.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    return out_path


def compute_conformational_coverage(
    discrete_states: Iterable[int] | List[int] | None,
    total_num_states: Optional[int],
) -> Optional[float]:
    """
    Estimate conformational coverage as unique visited states divided by
    total defined states. Returns None if not enough information.
    """
    try:
        if (
            not discrete_states
            or not isinstance(total_num_states, int)
            or total_num_states <= 0
        ):
            return None
        visited = {int(s) for s in discrete_states}
        coverage = len(visited) / float(total_num_states)
        return _clamp01(coverage)
    except Exception:
        return None


def compute_transition_matrix_accuracy(transition_matrix: Any) -> Optional[float]:
    """
    Proxy for MSM quality: 1 - mean absolute deviation of row sums from 1.0.
    Returns value in [0, 1], or None if matrix is unavailable.
    """
    try:
        # Local import to avoid hard dependency at module import time
        import numpy as np

        if transition_matrix is None:
            return None
        mat = np.asarray(transition_matrix, dtype=float)
        if mat.ndim != 2 or mat.shape[0] == 0:
            return None
        row_sums = mat.sum(axis=1)
        mad = float(np.mean(np.abs(row_sums - 1.0)))
        # Convert to accuracy-like score in [0, 1]
        score = 1.0 - mad
        return _clamp01(score)
    except Exception:
        return None


def compute_replica_exchange_success_rate(
    stats: Optional[Dict[str, Any]],
) -> Optional[float]:
    """
    Compute acceptance rate from replica exchange statistics dict.
    """
    try:
        if not stats:
            return None
        if "overall_acceptance_rate" in stats:
            return _clamp01(float(stats["overall_acceptance_rate"]))
        acc = float(stats.get("total_exchanges_accepted", 0.0))
        att = float(stats.get("total_exchange_attempts", 0.0))
        if att <= 0:
            return None
        return _clamp01(acc / att)
    except Exception:
        return None


class RuntimeMemoryTracker:
    """
    Lightweight runtime and memory tracker.

    Usage:
        with RuntimeMemoryTracker() as t:
            ... run work ...
        seconds = t.runtime_seconds
        memory_mb = t.max_rss_mb
    """

    def __init__(self) -> None:
        self._start_time: float = 0.0
        self._end_time: float = 0.0
        self.runtime_seconds: Optional[float] = None
        self.max_rss_mb: Optional[float] = None

    def __enter__(self) -> "RuntimeMemoryTracker":
        self._start_time = time.perf_counter()
        self.max_rss_mb = _get_process_rss_mb()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._end_time = time.perf_counter()
        self.runtime_seconds = self._end_time - self._start_time
        # Capture end RSS; take max(start, end) to be conservative
        end_rss = _get_process_rss_mb()
        if end_rss is None and self.max_rss_mb is None:
            self.max_rss_mb = None
        elif end_rss is None:
            # Keep start value
            pass
        elif self.max_rss_mb is None:
            self.max_rss_mb = end_rss
        else:
            self.max_rss_mb = max(self.max_rss_mb, end_rss)


def _get_process_rss_mb() -> Optional[float]:
    try:
        import psutil  # Optional dependency

        proc = psutil.Process()
        rss = float(proc.memory_info().rss)
        return rss / (1024.0 * 1024.0)
    except Exception:
        return None


def _clamp01(x: Optional[float]) -> Optional[float]:
    if x is None or math.isnan(x) or math.isinf(x):
        return None
    return max(0.0, min(1.0, float(x)))


def _finite_or_none(x: Optional[float]) -> Optional[float]:
    try:
        if x is None:
            return None
        xf = float(x)
        if math.isnan(xf) or math.isinf(xf):
            return None
        return xf
    except Exception:
        return None


# ---------- Additional KPI helpers for richer decisions ----------


def compute_row_stochasticity_mad(transition_matrix: Any) -> Optional[float]:
    """Mean absolute deviation of row sums from 1.0 (lower is better)."""
    try:
        import numpy as np

        if transition_matrix is None:
            return None
        mat = np.asarray(transition_matrix, dtype=float)
        if mat.ndim != 2 or mat.shape[0] == 0:
            return None
        row_sums = mat.sum(axis=1)
        mad = float(np.mean(np.abs(row_sums - 1.0)))
        return mad
    except Exception:
        return None


def compute_spectral_gap(transition_matrix: Any) -> Optional[float]:
    """
    Spectral gap 1 - lambda_2 for transition matrix (higher often indicates
    faster mixing / clearer metastability separation). Returns None if not computable.
    """
    try:
        import numpy as np

        if transition_matrix is None:
            return None
        mat = np.asarray(transition_matrix, dtype=float)
        if mat.ndim != 2 or mat.shape[0] == 0:
            return None
        eigenvals = np.linalg.eigvals(mat)
        eigenvals = np.real(eigenvals)
        eigenvals.sort()
        # For stochastic matrices, the largest eigenvalue is ~1.0
        # lambda_2 is the second largest eigenvalue
        lam2 = eigenvals[-2] if len(eigenvals) >= 2 else np.nan
        if np.isnan(lam2):
            return None
        return float(max(0.0, 1.0 - lam2))
    except Exception:
        return None


def compute_stationary_entropy(stationary_distribution: Any) -> Optional[float]:
    """Shannon entropy of stationary distribution in nats (higher → more spread)."""
    try:
        import numpy as np

        if stationary_distribution is None:
            return None
        pi = np.asarray(stationary_distribution, dtype=float)
        if pi.ndim != 1 or pi.size == 0:
            return None
        pi_safe = pi / max(np.sum(pi), 1e-12)
        pi_safe = np.clip(pi_safe, 1e-12, 1.0)
        return float(-np.sum(pi_safe * np.log(pi_safe)))
    except Exception:
        return None


def compute_frames_per_second(
    num_frames: Optional[int], runtime_seconds: Optional[float]
) -> Optional[float]:
    try:
        if not num_frames or not runtime_seconds or runtime_seconds <= 0:
            return None
        return float(num_frames) / float(runtime_seconds)
    except Exception:
        return None


def compute_wall_clock_per_step(
    runtime_seconds: Optional[float], steps: Optional[int]
) -> Optional[float]:
    try:
        if runtime_seconds is None or steps is None or steps <= 0:
            return None
        return float(runtime_seconds) / float(steps)
    except Exception:
        return None


# --- MSM quality diagnostics ---


def compute_detailed_balance_mad(
    transition_matrix: Any, stationary_distribution: Any
) -> Optional[float]:
    """
    Mean absolute deviation from detailed balance: average |pi_i T_ij - pi_j T_ji|.
    Lower is better; 0 implies perfect reversibility under pi.
    """
    try:
        import numpy as np

        if transition_matrix is None or stationary_distribution is None:
            return None
        T = np.asarray(transition_matrix, dtype=float)
        pi = np.asarray(stationary_distribution, dtype=float)
        if T.ndim != 2 or T.shape[0] == 0 or pi.ndim != 1:
            return None
        if T.shape[0] != T.shape[1] or T.shape[0] != pi.size:
            return None
        # Normalize pi defensively
        pi_sum = float(np.sum(pi))
        if pi_sum <= 0:
            return None
        pi = pi / pi_sum
        # Flow matrices
        F = pi[:, None] * T
        diff = np.abs(F - F.T)
        # Normalize by total flow to make scale-independent
        denom = float(np.sum(F))
        if denom <= 0:
            return None
        return float(np.mean(diff) / denom)
    except Exception:
        return None


def compute_its_convergence_score(
    implied_timescales: Any,
) -> Optional[float]:
    """
    Convergence score for implied timescales: lower is better.
    Computes the average relative absolute slope of timescales vs lag time
    across available non-NaN points for the first few timescales.
    """
    try:
        import numpy as np

        if not isinstance(implied_timescales, dict):
            return None
        lag_times = np.array(implied_timescales.get("lag_times"))
        timescales = np.array(implied_timescales.get("timescales"))
        if lag_times.size == 0 or timescales.size == 0:
            return None
        n_series = min(3, timescales.shape[1])  # focus on first few slowest
        scores = []
        for i in range(n_series):
            y = timescales[:, i]
            mask = np.isfinite(y) & np.isfinite(lag_times)
            if np.count_nonzero(mask) < 3:
                continue
            x = lag_times[mask].astype(float)
            y = y[mask].astype(float)
            # Fit simple line y = a x + b
            A = np.vstack([x, np.ones_like(x)]).T
            a, b = np.linalg.lstsq(A, y, rcond=None)[0]
            mean_y = float(np.mean(np.abs(y))) or 1.0
            rel_slope = float(abs(a)) / mean_y
            scores.append(rel_slope)
        if not scores:
            return None
        return float(np.mean(scores))
    except Exception:
        return None
