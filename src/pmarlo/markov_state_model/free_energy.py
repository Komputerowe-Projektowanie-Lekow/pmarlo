from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Any, ClassVar, Optional, Tuple

import numpy as np

from pmarlo import constants as const
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


@dataclass(init=False)
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

    version: ClassVar[str] = "2.0"
    F: NDArray[np.float64]
    xedges: NDArray[np.float64]
    yedges: NDArray[np.float64]
    levels_kJmol: NDArray[np.float64] | None
    metadata: dict[str, Any]
    counts: NDArray[np.float64] | None
    cv1_name: str | None
    cv2_name: str | None
    temperature: float | None

    def __init__(
        self,
        F: NDArray[np.float64] | None = None,
        *,
        free_energy: NDArray[np.float64] | None = None,
        xedges: NDArray[np.float64],
        yedges: NDArray[np.float64],
        levels_kJmol: NDArray[np.float64] | None = None,
        metadata: dict[str, Any] | None = None,
        counts: NDArray[np.float64] | None = None,
        cv1_name: str | None = None,
        cv2_name: str | None = None,
        temperature: float | None = None,
    ) -> None:
        if F is None and free_energy is None:
            raise TypeError(
                "FESResult requires either 'F' or 'free_energy' to be provided"
            )
        if F is not None and free_energy is not None:
            warnings.warn(
                "Both 'F' and 'free_energy' were provided; using 'F'",
                RuntimeWarning,
                stacklevel=2,
            )
        array_F = F if F is not None else free_energy
        self.F = np.asarray(array_F, dtype=np.float64)
        self.xedges = np.asarray(xedges, dtype=np.float64)
        self.yedges = np.asarray(yedges, dtype=np.float64)
        self.levels_kJmol = (
            None if levels_kJmol is None else np.asarray(levels_kJmol, dtype=np.float64)
        )

        meta: dict[str, Any] = dict(metadata or {})

        counts_value = counts if counts is not None else meta.get("counts")
        self.counts = (
            None if counts_value is None else np.asarray(counts_value, dtype=np.float64)
        )
        if self.counts is not None:
            meta["counts"] = self.counts

        self.cv1_name = cv1_name if cv1_name is not None else meta.get("cv1_name")
        self.cv2_name = cv2_name if cv2_name is not None else meta.get("cv2_name")
        if self.cv1_name is not None:
            meta.setdefault("cv1_name", self.cv1_name)
        if self.cv2_name is not None:
            meta.setdefault("cv2_name", self.cv2_name)

        temp_val = temperature if temperature is not None else meta.get("temperature")
        self.temperature = None if temp_val is None else float(temp_val)
        if self.temperature is not None:
            meta["temperature"] = self.temperature

        self.metadata = meta

    @property
    def output_shape(self) -> tuple[int, int]:
        """Shape of the free-energy surface grid."""
        return (int(self.F.shape[0]), int(self.F.shape[1]))

    @property
    def free_energy(self) -> NDArray[np.float64]:  # pragma: no cover - alias
        """Alias for the free-energy surface array for legacy consumers."""

        return self.F

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

    def to_dict(self, metadata_only: bool = False) -> dict[str, Any]:
        """Serialize the FES result to a JSON-friendly dictionary."""

        def _serialize(value: Any) -> Any:
            if isinstance(value, np.ndarray):
                if metadata_only:
                    return {"shape": list(value.shape), "dtype": str(value.dtype)}
                return value.tolist()
            return value

        payload: dict[str, Any] = {"version": self.version}

        primary_arrays = {
            "free_energy": self.F,
            "xedges": self.xedges,
            "yedges": self.yedges,
        }
        payload.update(
            {key: _serialize(value) for key, value in primary_arrays.items()}
        )

        optional_arrays = {
            key: value
            for key, value in {
                "levels_kJmol": self.levels_kJmol,
                "counts": self.counts,
            }.items()
            if value is not None
        }
        payload.update(
            {key: _serialize(value) for key, value in optional_arrays.items()}
        )

        optional_scalars = {
            "temperature": (
                float(self.temperature) if self.temperature is not None else None
            ),
            "cv1_name": self.cv1_name,
            "cv2_name": self.cv2_name,
        }
        payload.update(
            {key: value for key, value in optional_scalars.items() if value is not None}
        )

        excluded_keys = {"counts", "temperature", "cv1_name", "cv2_name"}
        metadata = {
            key: _serialize(value)
            for key, value in self.metadata.items()
            if key not in excluded_keys
        }
        if metadata:
            payload["metadata"] = metadata

        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FESResult":
        """Reconstruct an :class:`FESResult` from serialized metadata."""

        raw = dict(data)
        version = raw.pop("version", cls.version)
        if version not in {"1.0", "2.0"}:
            raise ValueError(f"Version mismatch: {version} != {cls.version}")

        def _restore(value: Any) -> Any:
            if isinstance(value, dict) and {"shape", "dtype"}.issubset(value.keys()):
                shape = tuple(int(x) for x in value["shape"])
                dtype = np.dtype(value.get("dtype", "float64"))
                return np.zeros(shape, dtype=dtype)
            if isinstance(value, list):
                return np.asarray(value)
            return value

        metadata_extra = raw.pop("metadata", {}) or {}
        counts = raw.pop("counts", None)
        cv1_name = raw.pop("cv1_name", None) or metadata_extra.get("cv1_name")
        cv2_name = raw.pop("cv2_name", None) or metadata_extra.get("cv2_name")
        temperature = raw.pop("temperature", None)
        if temperature is None:
            temperature = metadata_extra.get("temperature")

        levels = raw.pop("levels_kJmol", None)
        restored = cls(
            F=_restore(raw.pop("free_energy")),
            xedges=_restore(raw.pop("xedges")),
            yedges=_restore(raw.pop("yedges")),
            levels_kJmol=None if levels is None else _restore(levels),
            counts=None if counts is None else _restore(counts),
            metadata={k: _restore(v) for k, v in metadata_extra.items()},
            cv1_name=cv1_name,
            cv2_name=cv2_name,
            temperature=temperature,
        )
        return restored


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
    epsilon: float = const.NUMERIC_ABSOLUTE_TOLERANCE,
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
        # Adaptive percentile clipping to reduce outlier-driven empty bins
        try:
            if not any(periodic):
                x_p1, x_p99 = np.percentile(x, [1.0, 99.0])
                y_p1, y_p99 = np.percentile(y, [1.0, 99.0])
                if np.isfinite([x_p1, x_p99]).all() and x_p99 > x_p1:
                    xr = (float(x_p1), float(x_p99))
                else:
                    xr = (float(np.min(x)), float(np.max(x)))
                if np.isfinite([y_p1, y_p99]).all() and y_p99 > y_p1:
                    yr = (float(y_p1), float(y_p99))
                else:
                    yr = (float(np.min(y)), float(np.max(y)))
                # Clip samples into the selected range to keep edge bins populated
                x = np.clip(x, xr[0], xr[1]).astype(np.float64, copy=False)
                y = np.clip(y, yr[0], yr[1]).astype(np.float64, copy=False)
            else:
                xr = (float(np.min(x)), float(np.max(x)))
                yr = (float(np.min(y)), float(np.max(y)))
        except Exception:
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
    # Derive bins per axis from Freedman–Diaconis rule or sqrt(N) after percentile trimming
    bx_req, by_req = bins
    n_pts = int(x.shape[0])
    min_bins = 40
    max_bins = 512
    # sqrt rule baseline
    sqrt_bins = max(min_bins, int(np.sqrt(max(1, n_pts))))

    # Freedman–Diaconis per axis
    def _fd_bins(arr: NDArray[np.float64], lo: float, hi: float) -> int:
        try:
            q75, q25 = np.percentile(arr, [75.0, 25.0])
            iqr = float(q75 - q25)
            if iqr <= 0:
                return 0
            h = 2.0 * iqr / (max(1.0, n_pts) ** (1.0 / 3.0))
            if h <= 0:
                return 0
            nb = int((float(hi) - float(lo)) / float(h))
            return int(np.clip(nb, min_bins, max_bins))
        except Exception:
            return 0

    bx_fd = _fd_bins(x, xr[0], xr[1])
    by_fd = _fd_bins(y, yr[0], yr[1])
    # Choose the better of requested, FD, and sqrt rule
    bx = max(int(bx_req), int(bx_fd or 0), int(sqrt_bins))
    by = max(int(by_req), int(by_fd or 0), int(sqrt_bins))
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
    grid_shape = H_density.shape
    # Adaptive smoothing/inpainting decision based on occupancy
    total_bins = float(H_density.size)
    occupied = float(np.count_nonzero(H_counts >= max(1, min_count)))
    occ_frac = occupied / max(1.0, total_bins)
    empty_frac_initial = 1.0 - occ_frac
    adaptive = False
    # Auto-enable inpainting when more than 30% of bins are empty
    inpaint_flag = bool(inpaint or (empty_frac_initial > 0.30))
    smooth_flag = bool(smooth)
    # Compute Gaussian smoothing sigma from median bin width (1.3× per axis → ~1.3 bins)
    dx = np.median(np.diff(xedges)) if xedges.size > 1 else 1.0
    dy = np.median(np.diff(yedges)) if yedges.size > 1 else 1.0
    # Convert data-units sigma to grid sigma (bins): divide by bin width
    sigma_x_bins = float(1.3 * (dx / max(dx, np.finfo(float).eps)))
    sigma_y_bins = float(1.3 * (dy / max(dy, np.finfo(float).eps)))
    sigma_g = (sigma_x_bins, sigma_y_bins)
    if empty_frac_initial > 0.40:
        adaptive = True
        smooth_flag = True  # allow smooth density for readability
    if smooth_flag or inpaint_flag:
        if all(periodic):
            bw_rad = (np.radians(kde_bw_deg[0]), np.radians(kde_bw_deg[1]))
            kde_density = periodic_kde_2d(
                np.radians(x), np.radians(y), bw=bw_rad, gridsize=grid_shape
            )
        else:
            mode = tuple("wrap" if p else "reflect" for p in periodic)
            kde_density = gaussian_filter(
                H_density,
                sigma=sigma_g,
                mode=mode,
            ).astype(np.float64, copy=False)
            kde_density /= kde_density.sum() * bin_area

    density: NDArray[np.float64] = H_density.astype(np.float64, copy=False)
    if smooth_flag:
        density = kde_density
    if inpaint_flag:
        density[mask] = kde_density[mask]
    density /= density.sum() * bin_area

    if inpaint_flag:
        final_mask: NDArray[np.bool_] = np.zeros_like(mask, dtype=bool)
    else:
        final_mask = mask
    kT = _kT_kJ_per_mol(temperature)
    tiny = np.finfo(float).tiny
    # Avoid RuntimeWarning: divide by zero encountered in log by clipping first
    # and only assigning +inf where true zeros occurred. Using errstate keeps
    # logs clean without changing semantics.
    with np.errstate(divide="ignore", invalid="ignore"):
        F_safe = -kT * np.log(np.clip(density, tiny, None))
    F: NDArray[np.float64] = np.where(density > tiny, F_safe, np.inf)
    if not inpaint:
        F = np.where(final_mask, np.nan, F)
    if np.any(np.isfinite(F)):
        F -= np.nanmin(F)
    # Fraction of empty (below min_count) bins
    empty_bins_fraction = float(np.count_nonzero(final_mask)) / np.prod(H_density.shape)
    metadata = {
        "counts": density,
        "periodic": periodic,
        "temperature": temperature,
        "mask": final_mask,
        "empty_bins_fraction": empty_bins_fraction,
        "adaptive": {
            "initial_empty_fraction": float(empty_frac_initial),
            "bins": (int(bx), int(by)),
            "sigma_used": (
                (float(sigma_g[0]), float(sigma_g[1]))
                if (smooth_flag or inpaint_flag) and not all(periodic)
                else None
            ),
            "inpaint": bool(inpaint_flag),
            "smooth": bool(smooth_flag),
        },
    }
    if empty_bins_fraction > 0.60:
        metadata["sparse_warning"] = (
            f"Sparse FES ({empty_bins_fraction*100.0:.1f}% empty bins)"
        )
    if adaptive and empty_bins_fraction < empty_frac_initial:
        metadata["sparse_banner"] = "sparse data → adaptive smoothing"
    result = FESResult(F=F, xedges=xedges, yedges=yedges, metadata=metadata)

    masked_fraction = float(final_mask.sum()) / np.prod(result.output_shape)
    logger.info("FES masked fraction=%0.3f", masked_fraction)
    if masked_fraction > 0.30:
        logger.warning(
            "More than 30%% of bins are empty (%.1f%%)", masked_fraction * 100
        )

    return result
