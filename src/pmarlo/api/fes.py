"""Free energy surface generation utilities."""
from __future__ import annotations

import logging

from typing import Any, Mapping, Tuple, Sequence, Optional, Dict

import numpy as np

from .features import _fes_pair_from_phi_psi_maps

from ..markov_state_model.free_energy import FESResult, generate_2d_fes as _generate_2d_fes
from ..markov_state_model.picker import (
        pick_frames_around_minima
)

logger = logging.getLogger("pmarlo")

def _fes_highest_variance_pair(X: np.ndarray) -> Tuple[int, int] | None:
    """Return indices of the highest-variance CV columns.

    Constant (zero-variance) columns are ignored. If fewer than two
    non-constant columns remain, the lone surviving index is paired with
    itself. ``None`` is returned when ``X`` has no columns.
    """

    if X.shape[1] < 1:
        return None
    variances = np.var(X, axis=0)
    non_const = np.where(variances > 0)[0]
    if non_const.size == 0:
        logger.debug("[fes] No non-constant columns found for variance-based pairing")
        return None
    order = non_const[np.argsort(variances[non_const])[::-1]]
    if order.size == 1:
        idx = int(order[0])
        logger.debug("[fes] Only one non-constant column (idx=%d); pairing with itself", idx)
        return idx, idx
    i, j = int(order[0]), int(order[1])
    logger.debug("[fes] Highest variance pair: idx=(%d, %d)", i, j)
    return i, j


def _fes_periodic_pair_flags(
    periodic: np.ndarray, i_idx: int, j_idx: int
) -> Tuple[bool, bool]:
    pi = bool(periodic[i_idx]) if len(periodic) > i_idx else False
    pj = bool(periodic[j_idx]) if len(periodic) > j_idx else False
    return pi, pj

def _fes_pair_from_requested(
    cols: Sequence[str], requested: Optional[Tuple[str, str]]
) -> Tuple[int, int] | None:
    if requested is None:
        return None
    a, b = requested
    if a not in cols or b not in cols:
        raise ValueError(
            (
                f"Requested FES pair {requested} not found. Available columns "
                f"include: {cols[:12]} ..."
            )
        )
    return cols.index(a), cols.index(b)

def select_fes_pair(
    X: np.ndarray,
    cols: Sequence[str],
    periodic: np.ndarray,
    requested: Optional[Tuple[str, str]] = None,
    ensure: bool = True,
) -> Tuple[int, int, bool, bool]:
    """Select a pair of CV columns for FES.

    Preference order:
    1) If requested is provided, return those indices (or raise if missing).
    2) Pair phi:resN with psi:resN where available (lowest residue index).
    3) Fallback: highest-variance distinct pair if ensure=True.
    """

    # 1) Requested
    pair = _fes_pair_from_requested(cols, requested)
    if pair is not None:
        i, j = pair
        pi, pj = _fes_periodic_pair_flags(periodic, i, j)
        logger.info("[fes] Using requested pair: '%s' × '%s'", cols[i], cols[j])
        return i, j, pi, pj

    # 2) Residue-aware phi/psi pairing
    pair_phi_psi = _fes_pair_from_phi_psi_maps(cols)
    if pair_phi_psi is not None:
        i, j, rid = pair_phi_psi
        pi, pj = _fes_periodic_pair_flags(periodic, i, j)
        logger.info("[fes] φ/ψ pair selected: phi_res=%d, psi_res=%d", rid, rid)
        return i, j, pi, pj

    # 3) Highest-variance fallback
    if ensure:
        hv = _fes_highest_variance_pair(X)
        if hv is not None:
            i, j = hv
            pi, pj = _fes_periodic_pair_flags(periodic, i, j)
            logger.info("[fes] Variance-based pair: '%s' × '%s'", cols[i], cols[j])
            return i, j, pi, pj
        if X.shape[1] > 0:
            # Fold: use first axis for both coordinates
            logger.warning("[fes] All columns constant; folding first axis onto itself")
            pi, pj = _fes_periodic_pair_flags(periodic, 0, 0)
            return 0, 0, pi, pj

    raise RuntimeError("No suitable FES pair could be selected.")

def generate_free_energy_surface(
    cv1: np.ndarray,
    cv2: np.ndarray,
    bins: Tuple[int, int] = (100, 100),
    temperature: float = 300.0,
    periodic: Tuple[bool, bool] = (False, False),
    smooth: bool = False,
    inpaint: bool = False,
    min_count: int = 1,
    kde_bw_deg: Tuple[float, float] = (20.0, 20.0),
    config: Any | None = None,
    fes_smoothing_mode: str | None = None,
    fes_target_sd_kT: float | None = None,
    fes_alpha: float | None = None,
    fes_h0: float | None = None,
    fes_ess_ref: float | None = None,
    fes_h_min: float | None = None,
    fes_h_max: float | None = None,
    grid_strategy: str = "adaptive",
) -> FESResult:
    """Generate a 2D free-energy surface.

    Parameters
    ----------
    cv1, cv2
        Collective variable samples.
    bins
        Number of histogram bins in ``(x, y)``.
    temperature
        Simulation temperature in Kelvin.
    periodic
        Flags indicating whether each dimension is periodic.
    smooth
        If ``True``, smooth the density with a periodic KDE.
    inpaint
        If ``True``, fill empty bins using the KDE estimate.
    min_count
        Histogram bins with fewer samples are marked as empty unless ``inpaint``
        is ``True``.
    kde_bw_deg
        Bandwidth in degrees for the periodic KDE when smoothing or inpainting.
    config
        Optional configuration object supplying ``fes_*`` smoothing parameters.
    fes_smoothing_mode, fes_target_sd_kT, fes_alpha, fes_h0, fes_ess_ref,
    fes_h_min, fes_h_max
        Overrides for the corresponding smoothing options. ``None`` leaves the
        value unchanged relative to ``config`` or the defaults.
    grid_strategy
        Strategy for grid extent selection: "fixed" uses full data range,
        "adaptive" crops to [q1, q99] percentiles and adjusts bin counts
        to target finite_bins_fraction >= 0.6. Default is "adaptive".

    Returns
    -------
    FESResult
        Dataclass containing the free-energy surface and bin edges.
    """

    logger.info(
        "[fes] Generating 2D FES: n_samples=%d, bins=%s, T=%.1fK, periodic=%s, grid_strategy=%s",
        len(cv1), bins, temperature, periodic, grid_strategy
    )

    # Build processing options description
    opts = []
    if smooth:
        opts.append(f"smooth(bw={kde_bw_deg})")
    if inpaint:
        opts.append("inpaint")
    if fes_smoothing_mode:
        opts.append(f"mode={fes_smoothing_mode}")
    if opts:
        logger.info("[fes] Processing: %s", ", ".join(opts))

    config_payload: dict[str, Any] = {}
    if isinstance(config, Mapping):
        config_payload.update(config)
    elif config is not None:
        config_payload["__base_config__"] = config

    overrides = {
        "fes_smoothing_mode": fes_smoothing_mode,
        "fes_target_sd_kT": fes_target_sd_kT,
        "fes_alpha": fes_alpha,
        "fes_h0": fes_h0,
        "fes_ess_ref": fes_ess_ref,
        "fes_h_min": fes_h_min,
        "fes_h_max": fes_h_max,
    }
    for key, value in overrides.items():
        if value is not None:
            config_payload[key] = value

    config_arg: Any | None = config_payload if config_payload else config

    out = _generate_2d_fes(
        cv1,
        cv2,
        bins=bins,
        temperature=temperature,
        periodic=periodic,
        smooth=smooth,
        inpaint=inpaint,
        min_count=min_count,
        kde_bw_deg=kde_bw_deg,
        config=config_arg,
        grid_strategy=grid_strategy,
    )

    # Log summary statistics about the generated FES
    if hasattr(out, 'F') and out.F is not None:
        F_finite = out.F[np.isfinite(out.F)]
        if F_finite.size > 0:
            empty_frac = out.metadata.get("empty_bins_fraction", 0.0) if hasattr(out, 'metadata') else 0.0
            logger.info(
                "[fes] FES complete: F_min=%.2f, F_max=%.2f kJ/mol, %d/%d bins filled (%.1f%% empty)",
                np.min(F_finite), np.max(F_finite), F_finite.size, out.F.size, empty_frac * 100
            )
        else:
            logger.warning("[fes] FES generated but no finite energy values found")

    return out

def generate_fes_and_pick_minima(
    X: np.ndarray,
    cols: Sequence[str],
    periodic: np.ndarray,
    requested_pair: Optional[Tuple[str, str]] = None,
    bins: Tuple[int, int] = (60, 60),
    temperature: float = 300.0,
    smooth: bool = True,
    min_count: int = 1,
    kde_bw_deg: Tuple[float, float] = (20.0, 20.0),
    deltaF_kJmol: float = 3.0,
) -> Dict[str, Any]:
    """High-level helper to generate a 2D FES on selected pair and pick minima.

    Returns dict with keys: i, j, names, periodic_flags, fes (dict), minima (dict).
    """
    logger.info("[fes] Starting FES generation and minima detection workflow")

    i, j, per_i, per_j = select_fes_pair(
        X, cols, periodic, requested=requested_pair, ensure=True
    )
    cv1 = X[:, i].reshape(-1)
    cv2 = X[:, j].reshape(-1)

    # Convert angles to degrees when labeling suggests dihedrals
    name_i = cols[i]
    name_j = cols[j]
    converted = []
    if name_i.startswith("phi") or name_i.startswith("psi"):
        cv1 = np.degrees(cv1)
        converted.append(name_i)
    if name_j.startswith("phi") or name_j.startswith("psi"):
        cv2 = np.degrees(cv2)
        converted.append(name_j)
    if converted:
        logger.debug("[fes] Converted %s from radians to degrees", ", ".join(converted))

    if np.allclose(cv1, cv2):
        raise RuntimeError(
            "Selected FES pair are identical; aborting to avoid diagonal artifact."
        )

    fes = generate_free_energy_surface(
        cv1,
        cv2,
        bins=bins,
        temperature=temperature,
        periodic=(per_i, per_j),
        smooth=smooth,
        min_count=min_count,
        kde_bw_deg=kde_bw_deg,
    )

    logger.info("[fes] Picking minima with ΔF threshold=%.2f kJ/mol", deltaF_kJmol)
    minima = pick_frames_around_minima(
        cv1, cv2, fes.F, fes.xedges, fes.yedges, deltaF_kJmol=deltaF_kJmol
    )

    # Log minima detection results
    if isinstance(minima, dict):
        n_minima = len(minima.get("minima_indices", []))
        if n_minima > 0:
            logger.info("[fes] Found %d minima on the FES", n_minima)
        else:
            logger.warning("[fes] No minima detected on the FES")

    return {
        "i": int(i),
        "j": int(j),
        "names": (name_i, name_j),
        "periodic_flags": (bool(per_i), bool(per_j)),
        "fes": fes,
        "minima": minima,
    }