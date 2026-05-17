"""Benchmark utilities for comparing MD sampling strategies."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, List, Union

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["run_benchmark"]

if TYPE_CHECKING:
    import mdtraj


def run_benchmark(
    trajectories: Union["mdtraj.Trajectory", List["mdtraj.Trajectory"]],
    feature_type: str = "phi_psi",
    n_bins: int = 36,
    temperature_K: float = 300.0,
) -> Dict:
    """Compute sampling-quality metrics for one or more trajectories.

    Covers the three key questions for an adaptive-sampling benchmark:

    * **Coverage** – fraction of the 2-D feature space visited.
    * **Transitions** – number of times the first CV coordinate changes sign
      (a proxy for basin-crossing events).
    * **FES** – 2-D free-energy surface estimate in kJ/mol.

    Parameters
    ----------
    trajectories:
        One trajectory or a list of trajectories to combine.
    feature_type:
        Feature type passed to :func:`pmarlo.features.featurize_trajectory`.
        Use ``"phi_psi"`` for alanine dipeptide.
    n_bins:
        Number of histogram bins per axis.
    temperature_K:
        Temperature used when converting populations to free energies.

    Returns
    -------
    dict with keys:
        ``n_frames``, ``n_features``, ``feature_type``,
        ``coverage_2d`` (float 0–1), ``n_transitions`` (int),
        ``fes_2d`` (np.ndarray of shape ``(n_bins, n_bins)``).
    """
    import mdtraj

    from pmarlo.features import featurize_trajectory

    if not isinstance(trajectories, list):
        trajectories = [trajectories]

    combined: mdtraj.Trajectory = mdtraj.join(trajectories)
    feats = featurize_trajectory(combined, feature_type)

    results: Dict = {
        "n_frames": len(combined),
        "n_features": feats.shape[1],
        "feature_type": feature_type,
    }

    if feats.shape[1] >= 2:
        hist, xedges, yedges = np.histogram2d(
            feats[:, 0],
            feats[:, 1],
            bins=n_bins,
            range=[[-np.pi, np.pi], [-np.pi, np.pi]],
        )
        results["coverage_2d"] = float((hist > 0).sum() / hist.size)
        results["hist_edges"] = (xedges, yedges)

        kB_kJ = 8.314e-3  # kJ mol⁻¹ K⁻¹
        RT = kB_kJ * temperature_K
        prob = hist / hist.sum()
        fes = -RT * np.log(np.where(prob > 0, prob, np.nan))
        fes -= np.nanmin(fes)
        results["fes_2d"] = fes

    if feats.shape[1] >= 1:
        signs = np.sign(feats[:, 0])
        results["n_transitions"] = int(np.sum(np.abs(np.diff(signs)) > 0))

    logger.info(
        "Benchmark: %d frames | coverage=%.3f | transitions=%d",
        results["n_frames"],
        results.get("coverage_2d", float("nan")),
        results.get("n_transitions", 0),
    )
    return results
