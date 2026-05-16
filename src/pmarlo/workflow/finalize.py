"""Lightweight MSM/FES finalisation pipeline for precomputed projections."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, MutableMapping, Sequence

import numpy as np

from ..analysis import (
    compute_analysis_debug,
    compute_diagnostics,
    compute_weighted_fes,
    prepare_msm_discretization,
)
from ..reweight import AnalysisReweightMode, Reweighter

DatasetLike = MutableMapping[str, Any]


@dataclass(slots=True)
class AnalysisConfig:
    """Configuration for the analysis finalisation pipeline."""

    temperature_ref_K: float = 300.0
    lag_time: int = 1
    n_microstates: int = 150
    cluster_mode: str = "kmeans"
    reweight: str = AnalysisReweightMode.MBAR
    fes_bins: int = 64
    fes_split: str | None = "train"
    fes_method: str = "kde"
    fes_bandwidth: str | float = "scott"
    fes_min_count_per_bin: int = 1
    apply_whitening: bool = True
    collect_debug_data: bool = False

    def __post_init__(self) -> None:
        if self.lag_time < 1:
            raise ValueError("lag_time must be >= 1")
        if self.n_microstates < 2:
            raise ValueError("n_microstates must be >= 2")
        if self.temperature_ref_K <= 0:
            raise ValueError("temperature_ref_K must be positive")


def _format_debug_warning(entry: object) -> str:
    """Canonicalise analysis debug warnings for reporting."""

    if isinstance(entry, Mapping):
        code = str(entry.get("code", "ANALYSIS_DEBUG_WARNING"))
        message = entry.get("message")
        if message:
            return f"{code}: {message}"
        return code
    return str(entry)


def _normalise_reweight_mode(mode: str | None) -> str:
    if mode is None:
        return AnalysisReweightMode.NONE
    return AnalysisReweightMode.normalise(mode)


def _validate_debug_inputs(dataset: DatasetLike, cfg: AnalysisConfig) -> None:
    if not cfg.collect_debug_data:
        return

    raw_dtrajs = dataset.get("dtrajs")
    if not isinstance(raw_dtrajs, Sequence) or not raw_dtrajs:
        raise ValueError(
            "collect_debug_data requires 'dtrajs' sequences in the dataset"
        )
    if not any(np.asarray(traj).size > int(cfg.lag_time) for traj in raw_dtrajs):
        raise ValueError(
            "collect_debug_data requires at least one trajectory longer than lag"
        )


def _stationary_distribution(transition_matrix: np.ndarray) -> np.ndarray:
    transition = np.asarray(transition_matrix, dtype=np.float64)
    if transition.shape[0] == 0:
        return np.asarray([], dtype=np.float64)

    eigvals, eigvecs = np.linalg.eig(transition.T)
    idx = int(np.argmin(np.abs(eigvals - 1.0)))
    pi = np.real(eigvecs[:, idx]).astype(np.float64, copy=False)
    if np.sum(pi) < 0:
        pi = -pi
    pi = np.clip(pi, 0.0, None)
    total = float(np.sum(pi))
    if total <= 0 or not np.isfinite(total):
        raise ValueError(
            "Unable to compute stationary distribution from transition matrix"
        )
    return pi / total


def finalize_dataset(dataset: DatasetLike, cfg: AnalysisConfig) -> Dict[str, Any]:
    """Run MSM discretisation and FES estimation with optional reweighting."""

    if not isinstance(dataset, MutableMapping):
        raise ValueError("Dataset must be a mutable mapping of splits and metadata")
    _validate_debug_inputs(dataset, cfg)

    reweight_mode = _normalise_reweight_mode(cfg.reweight)
    weights: Dict[str, np.ndarray] | None = None
    effective_mode = AnalysisReweightMode.NONE

    if reweight_mode != AnalysisReweightMode.NONE:
        reweighter = Reweighter(cfg.temperature_ref_K)
        weights = reweighter.apply(dataset, mode=reweight_mode)
        effective_mode = reweight_mode

    msm = prepare_msm_discretization(
        dataset,
        cluster_mode=cfg.cluster_mode,
        n_microstates=cfg.n_microstates,
        lag_time=cfg.lag_time,
        frame_weights=weights,
        random_state=None,
        apply_whitening=cfg.apply_whitening,
    )

    counts = np.asarray(msm.counts, dtype=np.float64)
    pi = _stationary_distribution(msm.transition_matrix)
    fes_weights = weights.get(cfg.fes_split) if weights is not None else None

    fes = compute_weighted_fes(
        dataset,
        split=cfg.fes_split,
        weights=fes_weights,
        bins=cfg.fes_bins,
        temperature_K=cfg.temperature_ref_K,
        method=cfg.fes_method,
        bandwidth=cfg.fes_bandwidth,
        min_count_per_bin=cfg.fes_min_count_per_bin,
        apply_whitening=cfg.apply_whitening,
    )

    diagnostics = compute_diagnostics(dataset, diag_mass=msm.diag_mass)

    result: Dict[str, Any] = {
        "msm": msm,
        "transition_matrix": msm.transition_matrix,
        "counts": counts,
        "stationary_distribution": pi,
        "lag_time": msm.lag_time,
        "diag_mass": msm.diag_mass,
        "reweight_mode": effective_mode,
        "fes": fes,
        "diagnostics": diagnostics,
    }
    if weights is not None:
        result["frame_weights"] = weights
    if diagnostics.get("warnings"):
        result["warnings"] = diagnostics["warnings"]

    if cfg.collect_debug_data:
        debug_data = compute_analysis_debug(dataset, lag=cfg.lag_time)
        result["analysis_debug"] = debug_data

        debug_warnings = debug_data.summary.get("warnings", [])
        if debug_warnings:
            formatted = [_format_debug_warning(item) for item in debug_warnings]
            result.setdefault("warnings", [])
            result["warnings"].extend(formatted)
    return result
