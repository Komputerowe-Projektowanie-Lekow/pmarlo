from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter


@dataclass
class PMFResult:
    """Result of a one-dimensional potential of mean force calculation."""

    F: NDArray[np.float64]
    edges: NDArray[np.float64]
    counts: NDArray[np.float64]
    periodic: bool
    temperature: float

    @property
    def output_shape(self) -> tuple[int, ...]:
        """Shape of the PMF array."""
        return tuple(int(n) for n in self.F.shape)


@dataclass
class FESResult:
    """Result of a two-dimensional free-energy surface calculation.

    Parameters
    ----------
    F
        Free-energy surface values in kJ/mol.
    xedges, yedges
        Bin edges along the x and y axes.
    levels_kJmol
        Optional contour levels used for plotting.
    metadata
        Free-form dictionary for additional information such as ``counts`` or
        ``temperature``. The field ensures that the dataclass remains easily
        serialisable.
    """

    F: NDArray[np.float64]
    xedges: NDArray[np.float64]
    yedges: NDArray[np.float64]
    levels_kJmol: NDArray[np.float64] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def output_shape(self) -> tuple[int, int]:
        """Shape of the free-energy surface grid."""
        return (int(self.F.shape[0]), int(self.F.shape[1]))

    def __getitem__(self, key: str) -> Any:  # pragma: no cover - compatibility shim
        """Dictionary-style access with deprecation warning.

        Historically :class:`FESResult` behaved like a mapping. To preserve
        backwards compatibility, this method allows ``fes["F"]``-style access
        while emitting a :class:`DeprecationWarning`.
        """

        warnings.warn(
            "Dictionary-style access to FESResult is deprecated; use attributes "
            "instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        mapping = {
            "F": self.F,
            "xedges": self.xedges,
            "yedges": self.yedges,
            "levels_kJmol": self.levels_kJmol,
        }
        if key in mapping:
            return mapping[key]
        raise KeyError(key)


def _kT_kJ_per_mol(temperature_kelvin: float) -> float:
    from scipy import constants

    # Cast to float because scipy.constants may be typed as Any
    return float(constants.k * temperature_kelvin * constants.Avogadro / 1000.0)


logger = logging.getLogger(__name__)


def periodic_kde_2d(
    theta_x: np.ndarray,
    theta_y: np.ndarray,
    bw: Tuple[float, float] = (0.35, 0.35),
    gridsize: Tuple[int, int] = (42, 42),
) -> NDArray[np.float64]:
    """Kernel density estimate on a 2D torus.

    Parameters
    ----------
    theta_x, theta_y
        Angles in radians of equal shape.
    bw
        Bandwidth (standard deviations) along x and y in radians.
    gridsize
        Number of grid points along x and y.
    """

    x: NDArray[np.float64] = np.asarray(theta_x, dtype=np.float64).reshape(-1)
    y: NDArray[np.float64] = np.asarray(theta_y, dtype=np.float64).reshape(-1)
    if x.size == 0 or y.size == 0:
        raise ValueError("theta_x and theta_y must not be empty")
    if x.shape != y.shape:
        raise ValueError("theta_x and theta_y must have the same shape")

    sx, sy = float(bw[0]), float(bw[1])
    if sx <= 0 or sy <= 0:
        raise ValueError("bandwidth components must be positive")
    gx, gy = int(gridsize[0]), int(gridsize[1])
    if gx <= 0 or gy <= 0:
        raise ValueError("gridsize must be positive")

    x_grid: NDArray[np.float64] = np.linspace(-np.pi, np.pi, gx, endpoint=False).astype(
        np.float64, copy=False
    )
    y_grid: NDArray[np.float64] = np.linspace(-np.pi, np.pi, gy, endpoint=False).astype(
        np.float64, copy=False
    )
    X, Y = np.meshgrid(x_grid, y_grid, indexing="ij")
    X = X.astype(np.float64, copy=False)
    Y = Y.astype(np.float64, copy=False)
    density: NDArray[np.float64] = np.zeros_like(X, dtype=np.float64)

    shifts = (-2 * np.pi, 0.0, 2 * np.pi)
    for dx in shifts:
        for dy in shifts:
            diff_x = X[..., None] - (x + dx)[None, None, :]
            diff_y = Y[..., None] - (y + dy)[None, None, :]
            density += np.exp(-0.5 * ((diff_x / sx) ** 2 + (diff_y / sy) ** 2)).sum(
                axis=-1
            )

    norm = 2 * np.pi * sx * sy * x.size
    density /= norm
    return density.astype(np.float64, copy=False)


def generate_1d_pmf(
    cv: np.ndarray,
    bins: int = 100,
    temperature: float = 300.0,
    periodic: bool = False,
    range_: Optional[Tuple[float, float]] = None,
    smoothing_sigma: Optional[float] = None,
) -> PMFResult:
    """Generate a one-dimensional potential of mean force (PMF).

    Parameters
    ----------
    cv
        Collective variable samples.
    bins
        Number of histogram bins; must be positive.
    temperature
        Simulation temperature in Kelvin; must be positive.
    periodic
        Whether the CV is periodic.
    range_
        Optional histogram range as ``(min, max)``.
    smoothing_sigma
        Standard deviation for Gaussian smoothing; must be non-negative.
    """

    cv = np.asarray(cv, dtype=float).reshape(-1)
    if cv.size == 0:
        raise ValueError("cv array must not be empty")
    if bins <= 0:
        raise ValueError("bins must be positive")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if smoothing_sigma is not None and smoothing_sigma < 0:
        raise ValueError("smoothing_sigma must be non-negative")

    if range_ is None:
        hist_range = (float(np.min(cv)), float(np.max(cv)))
    else:
        hist_range = (float(range_[0]), float(range_[1]))
    if not np.isfinite(hist_range).all() or hist_range[0] >= hist_range[1]:
        raise ValueError("range_ must be finite with min < max")

    H, edges = np.histogram(cv, bins=bins, range=hist_range, density=True)
    if smoothing_sigma and smoothing_sigma > 0:
        H = gaussian_filter(
            H, sigma=float(smoothing_sigma), mode="wrap" if periodic else "reflect"
        )
    kT = _kT_kJ_per_mol(temperature)
    tiny = np.finfo(float).tiny
    H_clipped = np.clip(H, tiny, None)
    F = np.where(H > 0, -kT * np.log(H_clipped), np.inf)
    if np.any(np.isfinite(F)):
        F -= np.nanmin(F)
    return PMFResult(
        F=F, edges=edges, counts=H, periodic=periodic, temperature=temperature
    )


def generate_2d_fes(  # noqa: C901
    cv1: np.ndarray,
    cv2: np.ndarray,
    bins: Tuple[int, int] = (100, 100),
    temperature: float = 300.0,
    periodic: Tuple[bool, bool] = (False, False),
    ranges: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
    smooth: bool = False,
    inpaint: bool = False,
    min_count: int = 1,
    kde_bw_deg: Tuple[float, float] = (20.0, 20.0),
    epsilon: float = 1e-6,
) -> FESResult:
    """Generate a two-dimensional free-energy surface (FES)."""

    x: NDArray[np.float64] = (
        np.asarray(cv1, dtype=np.float64).reshape(-1).astype(np.float64, copy=False)
    )
    y: NDArray[np.float64] = (
        np.asarray(cv2, dtype=np.float64).reshape(-1).astype(np.float64, copy=False)
    )
    if x.size == 0 or y.size == 0:
        raise ValueError("cv1 and cv2 must not be empty")
    if x.shape != y.shape:
        raise ValueError("cv1 and cv2 must have the same shape")
    if len(bins) != 2 or any(b <= 0 for b in bins):
        raise ValueError("bins must be a tuple of two positive integers")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if len(periodic) != 2:
        raise ValueError("periodic must be a tuple of two booleans")
    if min_count < 0:
        raise ValueError("min_count must be non-negative")

    if ranges is None:
        xr = (float(np.min(x)), float(np.max(x)))
        yr = (float(np.min(y)), float(np.max(y)))
    else:
        if len(ranges) != 2 or any(len(r) != 2 for r in ranges):
            raise ValueError("ranges must be ((xmin, xmax), (ymin, ymax))")
        xr = (float(ranges[0][0]), float(ranges[0][1]))
        yr = (float(ranges[1][0]), float(ranges[1][1]))
    if not np.isfinite(xr + yr).all() or xr[0] >= xr[1] or yr[0] >= yr[1]:
        raise ValueError("ranges must be finite with min < max for both axes")

    # Wrap periodic coordinates into the specified range
    if periodic[0]:
        x = (((x - xr[0]) % (xr[1] - xr[0])) + xr[0]).astype(np.float64, copy=False)
    if periodic[1]:
        y = (((y - yr[0]) % (yr[1] - yr[0])) + yr[0]).astype(np.float64, copy=False)

    # Build edges with an extra bin to allow for periodic wrapping
    bx, by = bins
    x_edges: NDArray[np.float64] = np.linspace(xr[0], xr[1], bx + 1).astype(
        np.float64, copy=False
    )
    y_edges: NDArray[np.float64] = np.linspace(yr[0], yr[1], by + 1).astype(
        np.float64, copy=False
    )
    if periodic[0]:
        dx = x_edges[1] - x_edges[0]
        x_hist_edges = np.concatenate([x_edges, [x_edges[-1] + dx]])
    else:
        x_hist_edges = x_edges
    if periodic[1]:
        dy = y_edges[1] - y_edges[0]
        y_hist_edges = np.concatenate([y_edges, [y_edges[-1] + dy]])
    else:
        y_hist_edges = y_edges

    H_counts, _, _ = np.histogram2d(x, y, bins=(x_hist_edges, y_hist_edges))
    if periodic[0]:
        H_counts[0, :] += H_counts[-1, :]
        H_counts = H_counts[:-1, :]
    if periodic[1]:
        H_counts[:, 0] += H_counts[:, -1]
        H_counts = H_counts[:, :-1]

    xedges = x_edges
    yedges = y_edges
    bin_area = np.diff(xedges)[0] * np.diff(yedges)[0]
    H_density: NDArray[np.float64] = (H_counts / (H_counts.sum() * bin_area)).astype(
        np.float64, copy=False
    )
    mask: NDArray[np.bool_] = H_counts < min_count

    kde_density: NDArray[np.float64] = np.zeros_like(H_density, dtype=np.float64)
    if smooth or inpaint:
        if all(periodic):
            bw_rad = (np.radians(kde_bw_deg[0]), np.radians(kde_bw_deg[1]))
            kde_density = periodic_kde_2d(
                np.radians(x), np.radians(y), bw=bw_rad, gridsize=bins
            )
        else:
            mode = tuple("wrap" if p else "reflect" for p in periodic)
            kde_density = gaussian_filter(H_density, sigma=0.6, mode=mode).astype(
                np.float64, copy=False
            )
            kde_density /= kde_density.sum() * bin_area

    density: NDArray[np.float64] = H_density.astype(np.float64, copy=False)
    if smooth:
        density = kde_density
    if inpaint:
        density[mask] = kde_density[mask]
    density /= density.sum() * bin_area

    if inpaint:
        final_mask: NDArray[np.bool_] = np.zeros_like(mask, dtype=bool)
    else:
        final_mask = mask
    kT = _kT_kJ_per_mol(temperature)
    tiny = np.finfo(float).tiny
    F: NDArray[np.float64] = np.where(density > tiny, -kT * np.log(density), np.inf)
    if not inpaint:
        F = np.where(final_mask, np.nan, F)
    if np.any(np.isfinite(F)):
        F -= np.nanmin(F)
    metadata = {
        "counts": density,
        "periodic": periodic,
        "temperature": temperature,
        "mask": final_mask,
    }
    result = FESResult(F=F, xedges=xedges, yedges=yedges, metadata=metadata)

    masked_fraction = float(final_mask.sum()) / np.prod(result.output_shape)
    logger.info("FES masked fraction=%0.3f", masked_fraction)
    if masked_fraction > 0.30:
        logger.warning(
            "More than 30%% of bins are empty (%.1f%%)", masked_fraction * 100
        )

    return result
