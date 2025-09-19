"""Lightweight MSM/FES finalisation pipeline for precomputed projections."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, MutableMapping

import numpy as np

from ..analysis import compute_weighted_fes, prepare_msm_discretization, compute_diagnostics
from ..reweight import AnalysisReweightMode, Reweighter

DatasetLike = MutableMapping[str, Any] | Mapping[str, Any]


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


def _normalise_reweight_mode(mode: str | None) -> str:
    if mode is None:
        return AnalysisReweightMode.NONE
    return AnalysisReweightMode.normalise(mode)


def finalize_dataset(dataset: DatasetLike, cfg: AnalysisConfig) -> Dict[str, Any]:
    """Run MSM discretisation and FES estimation with optional reweighting."""

    if not isinstance(dataset, (MutableMapping, dict)):
        raise ValueError("Dataset must be a mutable mapping of splits and metadata")

    reweight_mode = _normalise_reweight_mode(cfg.reweight)
    weights: Dict[str, np.ndarray] | None = None
    effective_mode = AnalysisReweightMode.NONE

    if reweight_mode != AnalysisReweightMode.NONE:
        reweighter = Reweighter(cfg.temperature_ref_K)
        try:
            weights = reweighter.apply(dataset, mode=reweight_mode)
            effective_mode = reweight_mode
        except Exception:
            weights = None
            effective_mode = AnalysisReweightMode.NONE
    else:
        effective_mode = AnalysisReweightMode.NONE

    msm = prepare_msm_discretization(
        dataset,
        cluster_mode=cfg.cluster_mode,
        n_microstates=cfg.n_microstates,
        lag_time=cfg.lag_time,
        frame_weights=weights,
        random_state=None,
        apply_whitening=bool(cfg.apply_whitening),
    )

    counts = np.asarray(msm.counts, dtype=np.float64)
    row_sums = counts.sum(axis=1)
    total = float(np.sum(row_sums))
    if total > 0 and counts.shape[0] > 0:
        pi = row_sums / total
    elif counts.shape[0] > 0:
        pi = np.full((counts.shape[0],), 1.0 / counts.shape[0], dtype=np.float64)
    else:
        pi = np.asarray([], dtype=np.float64)

    fes = compute_weighted_fes(
        dataset,
        split=cfg.fes_split,
        weights=None,
        bins=cfg.fes_bins,
        temperature_K=cfg.temperature_ref_K,
        method=cfg.fes_method,
        bandwidth=cfg.fes_bandwidth,
        min_count_per_bin=cfg.fes_min_count_per_bin,
        apply_whitening=bool(cfg.apply_whitening),
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
    return result
