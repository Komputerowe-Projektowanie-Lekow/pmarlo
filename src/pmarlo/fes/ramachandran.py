from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, Sequence

import mdtraj as md  # type: ignore
import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter

from .surfaces import _kT_kJ_per_mol

logger = logging.getLogger("pmarlo")


@dataclass
class RamachandranResult:
    """Result of a Ramachandran free-energy surface calculation."""

    F: NDArray[np.float64]
    phi_edges: NDArray[np.float64]
    psi_edges: NDArray[np.float64]
    counts: NDArray[np.float64]
    finite_fraction: float
    temperature: float


def compute_ramachandran(
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
    return angles_deg.astype(np.float64, copy=False)


def periodic_hist2d(
    phi: NDArray[np.float64],
    psi: NDArray[np.float64],
    bins: tuple[int, int] = (42, 42),
    smoothing: float | None = 1.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Compute a periodic 2D histogram with optional Gaussian smoothing."""

    x = np.asarray(phi, dtype=float).ravel()
    y = np.asarray(psi, dtype=float).ravel()
    if x.shape != y.shape:
        raise ValueError("phi and psi must have the same shape")

    bx, by = bins
    x_edges = np.linspace(-180.0, 180.0, bx + 2)
    y_edges = np.linspace(-180.0, 180.0, by + 2)
    H_raw, _, _ = np.histogram2d(x, y, bins=(x_edges, y_edges))
    H: NDArray[np.float64] = np.asarray(H_raw, dtype=float)

    H[0, :] += H[-1, :]
    H = H[:-1, :]
    H[:, 0] += H[:, -1]
    H = H[:, :-1]

    if smoothing and smoothing > 0:
        H = gaussian_filter(H, sigma=float(smoothing), mode="wrap")

    return H, x_edges[:-1], y_edges[:-1]


def compute_ramachandran_fes(
    traj: md.Trajectory,
    selection: int | str | Sequence[int] | None = None,
    bins: tuple[int, int] = (42, 42),
    temperature: float = 300.0,
    min_count: int = 5,
    smoothing: float | None = 1.0,
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
    smoothing
        Standard deviation for Gaussian smoothing. ``None`` or ``0`` disables
        smoothing.
    stride
        Use every ``stride``-th frame. If ``None``, determined from ``tau``.
    tau
        Correlation time in frames. If provided and ``stride`` is ``None``, the
        stride defaults to ``max(1, int(tau / 2))``.
    """

    if stride is None:
        stride = max(1, int(tau / 2)) if tau is not None else 1
    stride = max(1, int(stride))

    angles = compute_ramachandran(traj, selection)[::stride]
    H, xedges, yedges = periodic_hist2d(
        angles[:, 0], angles[:, 1], bins=bins, smoothing=smoothing
    )

    total: float = float(np.sum(H))
    if total == 0:
        raise ValueError("Histogram is empty; check input trajectory and selection")
    p = H / total
    kT = _kT_kJ_per_mol(temperature)
    tiny = np.finfo(float).tiny
    F = np.full_like(H, np.inf, dtype=float)
    mask = H >= float(min_count)
    F[mask] = -kT * np.log(np.clip(p[mask], tiny, None))
    if np.any(np.isfinite(F)):
        F -= np.nanmin(F)
    finite_fraction = float(np.isfinite(F).sum()) / F.size
    logger.info(
        "Ramachandran FES finite bins: %d/%d (%.1f%%)",
        np.isfinite(F).sum(),
        F.size,
        finite_fraction * 100.0,
    )
    return RamachandranResult(
        F=F,
        phi_edges=xedges,
        psi_edges=yedges,
        counts=H,
        finite_fraction=finite_fraction,
        temperature=float(temperature),
    )
