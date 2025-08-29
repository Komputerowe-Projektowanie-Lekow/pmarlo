from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, Sequence

import mdtraj as md  # type: ignore
import numpy as np
from numpy.typing import NDArray

from .surfaces import _kT_kJ_per_mol, periodic_kde_2d

logger = logging.getLogger("pmarlo")


@dataclass
class RamachandranResult:
    """Result of a Ramachandran free-energy surface calculation."""

    F: NDArray[np.float64]
    phi_edges: NDArray[np.float64]
    psi_edges: NDArray[np.float64]
    counts: NDArray[np.float64]
    mask: NDArray[np.bool_]
    finite_fraction: float
    temperature: float

    @property
    def output_shape(self) -> tuple[int, int]:
        """Grid shape of the Ramachandran surface."""
        return (int(self.F.shape[0]), int(self.F.shape[1]))


def compute_ramachandran(  # noqa: C901
    traj: md.Trajectory,
    selection: int | str | Sequence[int] | None = None,
) -> NDArray[np.float64]:
    """Compute φ/ψ angles for a single residue in degrees.

    Parameters
    ----------
    traj
        MD trajectory.
    selection
        Residue index, selection string, or sequence of residue indices. If
        ``None``, the central residue with both φ and ψ defined is used.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_frames, 2)`` containing wrapped ``(φ, ψ)`` angles
        in degrees.
    """

    phi, phi_idx = md.compute_phi(traj)
    psi, psi_idx = md.compute_psi(traj)
    if phi.size == 0 or psi.size == 0:
        raise ValueError("No phi/psi angles could be computed from trajectory")

    top = traj.topology
    phi_res = [top.atom(int(f[1])).residue.index for f in phi_idx]
    psi_res = [top.atom(int(f[2])).residue.index for f in psi_idx]
    common_res = sorted(set(phi_res).intersection(psi_res))
    if not common_res:
        raise ValueError("No residues with both phi and psi angles")

    chosen: int
    if selection is None:
        chosen = common_res[len(common_res) // 2]
    else:
        if isinstance(selection, str):
            sel_res = {
                top.atom(i).residue.index for i in traj.topology.select(selection)
            }
        elif isinstance(selection, Iterable):
            sel_res = set(int(r) for r in selection)
        else:
            sel_res = {int(selection)}
        overlap = [r for r in common_res if r in sel_res]
        if not overlap:
            raise ValueError("Selection does not include residues with phi and psi")
        chosen = overlap[0]

    phi_col = phi_res.index(chosen)
    psi_col = psi_res.index(chosen)

    angles = np.stack([phi[:, phi_col], psi[:, psi_col]], axis=1).astype(np.float64)
    angles_deg = np.degrees(angles)
    angles_deg = ((angles_deg + 180.0) % 360.0) - 180.0
    angles_deg[angles_deg <= -180.0] += 360.0

    # Self-check for nearly identical φ/ψ streams
    if angles_deg.shape[0] > 1:
        corr = float(np.corrcoef(angles_deg[:, 0], angles_deg[:, 1])[0, 1])
    else:
        corr = 0.0
    if abs(corr) > 0.99:
        logger.warning("phi/psi nearly identical; check residue mapping")
        for alt in common_res:
            if alt == chosen:
                continue
            alt_phi_col = phi_res.index(alt)
            alt_psi_col = psi_res.index(alt)
            alt_angles = np.stack(
                [phi[:, alt_phi_col], psi[:, alt_psi_col]], axis=1
            ).astype(np.float64)
            alt_deg = np.degrees(alt_angles)
            alt_deg = ((alt_deg + 180.0) % 360.0) - 180.0
            alt_deg[alt_deg <= -180.0] += 360.0
            if alt_deg.shape[0] > 1:
                alt_corr = float(np.corrcoef(alt_deg[:, 0], alt_deg[:, 1])[0, 1])
            else:
                alt_corr = 0.0
            if abs(alt_corr) <= 0.99:
                angles_deg = alt_deg
                phi_col = alt_phi_col
                psi_col = alt_psi_col
                chosen = alt
                break

    logger.info("phi_res=%d, psi_res=%d", phi_res[phi_col], psi_res[psi_col])
    return angles_deg.astype(np.float64, copy=False)


def periodic_hist2d(
    phi: NDArray[np.float64],
    psi: NDArray[np.float64],
    bins: tuple[int, int] = (42, 42),
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Compute a periodic 2D histogram."""

    x: NDArray[np.float64] = np.asarray(phi, dtype=np.float64).ravel()
    y: NDArray[np.float64] = np.asarray(psi, dtype=np.float64).ravel()
    if x.shape != y.shape:
        raise ValueError("phi and psi must have the same shape")

    bx, by = bins
    x_edges = np.linspace(-180.0, 180.0, bx + 2)
    y_edges = np.linspace(-180.0, 180.0, by + 2)
    H_raw, _, _ = np.histogram2d(x, y, bins=(x_edges, y_edges))
    H: NDArray[np.float64] = np.asarray(H_raw, dtype=np.float64)
    H[0, :] += H[-1, :]
    H = H[:-1, :]
    H[:, 0] += H[:, -1]
    H = H[:, :-1]
    H = H.astype(np.float64, copy=False)
    xout: NDArray[np.float64] = x_edges[:-1].astype(np.float64, copy=False)
    yout: NDArray[np.float64] = y_edges[:-1].astype(np.float64, copy=False)
    return (H, xout, yout)


def compute_ramachandran_fes(
    traj: md.Trajectory,
    selection: int | str | Sequence[int] | None = None,
    bins: tuple[int, int] = (42, 42),
    temperature: float = 300.0,
    min_count: int = 5,
    smooth: bool = False,
    inpaint: bool = False,
    kde_bw_deg: tuple[float, float] = (20.0, 20.0),
    stride: int | None = None,
    tau: float | None = None,
) -> RamachandranResult:
    """Compute a Ramachandran free-energy surface.

    Parameters
    ----------
    traj
        MD trajectory.
    selection
        Residue selection passed to :func:`compute_ramachandran`.
    bins
        Histogram bins in ``(φ, ψ)``.
    temperature
        Temperature in Kelvin.
    min_count
        Minimum count to consider a bin populated.
    smooth
        If ``True``, use periodic Gaussian KDE to smooth the density.
    inpaint
        If ``True``, fill empty bins using the KDE estimate.
    kde_bw_deg
        Bandwidth in degrees for the KDE when smoothing or inpainting.
    stride
        Use every ``stride``-th frame. Defaults to ``1`` (use all frames).
    tau
        Correlation time in frames. Retained for compatibility but no longer
        alters the frame stride.
    """

    stride = max(1, int(stride or 1))
    angles = compute_ramachandran(traj, selection)[::stride]
    H, xedges, yedges = periodic_hist2d(angles[:, 0], angles[:, 1], bins=bins)

    mask = H < float(min_count)
    total: float = float(np.sum(H))
    if total == 0:
        raise ValueError("Histogram is empty; check input trajectory and selection")
    bin_area = np.diff(xedges)[0] * np.diff(yedges)[0]
    density = H / (total * bin_area)

    kde_density = np.zeros_like(density)
    if smooth or inpaint:
        bw_rad = (np.radians(kde_bw_deg[0]), np.radians(kde_bw_deg[1]))
        kde_density = periodic_kde_2d(
            np.radians(angles[:, 0]),
            np.radians(angles[:, 1]),
            bw=bw_rad,
            gridsize=bins,
        )
    if smooth:
        density = kde_density
    if inpaint:
        density[mask] = kde_density[mask]
    density /= density.sum() * bin_area

    masked_fraction = float(mask.sum()) / mask.size
    logger.info("Ramachandran FES masked bins: %.1f%%", masked_fraction * 100.0)
    if masked_fraction > 0.30:
        logger.warning(
            "More than 30%% of Ramachandran bins are empty (%.1f%%)",
            masked_fraction * 100.0,
        )

    kT = _kT_kJ_per_mol(temperature)
    tiny = np.finfo(float).tiny
    F = np.where(density > tiny, -kT * np.log(density), np.inf)
    if not inpaint:
        F = np.where(mask, np.nan, F)
    if np.any(np.isfinite(F)):
        F -= np.nanmin(F)
    finite_bins = int(np.isfinite(F).sum())
    result = RamachandranResult(
        F=F,
        phi_edges=xedges,
        psi_edges=yedges,
        counts=H,
        mask=mask,
        finite_fraction=float(finite_bins) / np.prod(F.shape),
        temperature=float(temperature),
    )
    logger.info(
        "Ramachandran FES finite bins: %d/%d (%.1f%%)",
        finite_bins,
        np.prod(result.output_shape),
        result.finite_fraction * 100.0,
    )
    return result
