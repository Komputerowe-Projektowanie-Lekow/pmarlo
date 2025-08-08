"""
Benchmarking utilities for experiments.

Provides:
- Environment capture for reproducibility
- Baseline and trend persistence
- Threshold-based regression/improvement comparison
"""

from __future__ import annotations

import json
import platform as _platform
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def get_environment_info() -> Dict[str, Any]:
    """Capture environment details for reproducibility."""
    # CPU info via platform; psutil for more details if available
    cpu_info = _platform.processor() or _platform.machine()
    gpu_info: Optional[str] = None
    try:
        # Attempt to detect CUDA device via OpenMM if available
        import openmm
        from openmm import Platform

        try:
            Platform.getPlatformByName("CUDA")
            gpu_info = "CUDA available"
        except Exception:
            gpu_info = None
    except Exception:
        gpu_info = None

    # OS and Python version
    os_name = f"{_platform.system()} {_platform.release()}"
    python_version = sys.version.split()[0]

    # Optional psutil memory and cpu details
    try:
        import psutil

        cpu_count = psutil.cpu_count(logical=True)
        cpu_info = f"{cpu_info} ({cpu_count} CPUs)" if cpu_info else f"{cpu_count} CPUs"
    except Exception:
        pass

    # OpenMM platform (best-effort)
    openmm_platform_name: Optional[str] = None
    try:
        from openmm import Platform

        # Prefer Reference/CPU/CUDA availability order is environment-dependent
        for name in ("CUDA", "CPU", "Reference", "OpenCL"):
            try:
                Platform.getPlatformByName(name)
                openmm_platform_name = name
                break
            except Exception:
                continue
    except Exception:
        openmm_platform_name = None

    return {
        "platform": openmm_platform_name or "unknown",
        "cpu_info": cpu_info or "unknown",
        "gpu_info": gpu_info,
        "os": os_name,
        "python_version": python_version,
    }


def _safe_read_json(path: Path) -> Optional[Any]:
    try:
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        # Malformed JSON – stop processing per requirements
        return {"error": f"Parse error reading {str(path)}"}


def _safe_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def initialize_baseline_if_missing(
    dir_root: Path, baseline_object: Dict[str, Any]
) -> None:
    """Create baseline.json at dir_root if missing."""
    baseline_path = dir_root / "baseline.json"
    if not baseline_path.exists():
        _safe_write_json(baseline_path, baseline_object)


def update_trend(
    dir_root: Path, run_object: Dict[str, Any], max_entries: int = 20
) -> None:
    """Append run to trend.json (oldest to newest)."""
    trend_path = dir_root / "trend.json"
    existing = _safe_read_json(trend_path)
    if isinstance(existing, dict) and "error" in existing:
        return  # stop on parse error
    if not isinstance(existing, list):
        trend: List[Dict[str, Any]] = []
    else:
        trend = existing
    trend.append(run_object)
    if len(trend) > max_entries:
        trend = trend[-max_entries:]
    _safe_write_json(trend_path, trend)


def compute_threshold_comparison(
    previous: Dict[str, Any],
    current: Dict[str, Any],
    *,
    fps_regression_pct: float = 5.0,
    seconds_per_step_regression_pct: float = 5.0,
    spectral_gap_regression_pct: float = 5.0,
    transition_matrix_accuracy_regression_pct: float = 5.0,
) -> Dict[str, Dict[str, Any]]:
    """
    Compare key metrics: fps and seconds_per_step.
    Returns a dict with deltas and flags.
    """

    def _pct_change(old: float, new: float) -> float:
        if old == 0:
            return 0.0
        return (new - old) / old * 100.0

    comparison: Dict[str, Dict[str, Any]] = {}

    # Extract metrics from input_parameters for both records
    prev_in = previous.get("input_parameters", {}) if isinstance(previous, dict) else {}
    curr_in = current.get("input_parameters", {}) if isinstance(current, dict) else {}

    # Frames per second
    old_fps = prev_in.get("frames_per_second")
    new_fps = curr_in.get("frames_per_second")
    if isinstance(old_fps, (int, float)) and isinstance(new_fps, (int, float)):
        pct = _pct_change(old_fps, new_fps)
        comparison["fps"] = {
            "delta": new_fps - old_fps,
            "percent_change": pct,
            "regression": pct < -fps_regression_pct,
            "improvement": pct > fps_regression_pct,
            "threshold_exceeded": abs(pct) > fps_regression_pct,
        }

    # Seconds per step (lower is better)
    old_sps = prev_in.get("seconds_per_step")
    new_sps = curr_in.get("seconds_per_step")
    if isinstance(old_sps, (int, float)) and isinstance(new_sps, (int, float)):
        pct = _pct_change(old_sps, new_sps)
        comparison["seconds_per_step"] = {
            "delta": new_sps - old_sps,
            "percent_change": pct,
            "regression": pct > seconds_per_step_regression_pct,  # increase is bad
            "improvement": pct < -seconds_per_step_regression_pct,  # decrease is good
            "threshold_exceeded": abs(pct) > seconds_per_step_regression_pct,
        }

    return comparison


def build_baseline_object(
    *,
    input_parameters: Dict[str, Any],
    results: Dict[str, Any],
    min_spectral_gap: float = 0.5,
    max_seconds_per_step: float = 0.08,
) -> Dict[str, Any]:
    """Create a baseline.json-compatible object with success_criteria booleans."""
    spectral_gap = input_parameters.get("spectral_gap")
    seconds_per_step = input_parameters.get("seconds_per_step")

    success_criteria = {
        "min_spectral_gap": bool(
            isinstance(spectral_gap, (int, float)) and spectral_gap >= min_spectral_gap
        ),
        "max_seconds_per_step": bool(
            isinstance(seconds_per_step, (int, float))
            and seconds_per_step <= max_seconds_per_step
        ),
    }

    return {
        "input_parameters": input_parameters,
        "results": results,
        "success_criteria": success_criteria,
    }


def build_msm_baseline_object(
    *,
    input_parameters: Dict[str, Any],
    results: Dict[str, Any],
    min_transition_matrix_accuracy: float = 0.85,
    min_conformational_coverage: float = 0.8,
) -> Dict[str, Any]:
    """
    Create a baseline object for MSM runs with MSM-specific success criteria.

    Drops seconds_per_step and focuses on model quality metrics.
    """
    tma = results.get("transition_matrix_accuracy")
    cov = results.get("conformational_coverage")

    success_criteria = {
        "min_transition_matrix_accuracy": bool(
            isinstance(tma, (int, float)) and tma >= min_transition_matrix_accuracy
        ),
        "min_conformational_coverage": bool(
            isinstance(cov, (int, float)) and cov >= min_conformational_coverage
        ),
    }

    return {
        "input_parameters": input_parameters,
        "results": results,
        "success_criteria": success_criteria,
    }


def build_remd_baseline_object(
    *,
    input_parameters: Dict[str, Any],
    results: Dict[str, Any],
    min_acceptance_rate: float = 0.2,
    max_seconds_per_step: float = 0.08,
) -> Dict[str, Any]:
    """
    Create a baseline object for REMD runs with REMD-specific success criteria.

    - min_acceptance_rate: minimum overall exchange acceptance rate
    - max_seconds_per_step: throughput guardrail
    """
    acc = input_parameters.get("overall_acceptance_rate") or results.get(
        "replica_exchange_success_rate"
    )
    sps = input_parameters.get("seconds_per_step")

    success_criteria = {
        "min_overall_acceptance_rate": bool(
            isinstance(acc, (int, float)) and acc >= min_acceptance_rate
        ),
        "max_seconds_per_step": bool(
            isinstance(sps, (int, float)) and sps <= max_seconds_per_step
        ),
    }

    return {
        "input_parameters": input_parameters,
        "results": results,
        "success_criteria": success_criteria,
    }
