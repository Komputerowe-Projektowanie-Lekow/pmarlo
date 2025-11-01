from __future__ import annotations

import logging
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter
from scipy.stats import iqr
from scipy.stats.mstats import mquantiles

from pmarlo import constants as const
from pmarlo.utils.thermodynamics import kT_kJ_per_mol

from .fes_smoothing import (
    adaptive_bandwidth,
    beta_to_kT,
    mark_bins_for_smoothing,
    smooth_F_with_adaptive_gaussian,
)


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
        """Alias for the free-energy surface array for backward-compatible consumers."""

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


def free_energy_from_density(
    density: NDArray[np.float64],
    temperature: float,
    *,
    mask: NDArray[np.bool_] | None = None,
    inpaint: bool = False,
    tiny: float | None = None,
) -> NDArray[np.float64]:
    """Convert a normalised probability density into a free-energy surface.

    Parameters
    ----------
    density
        Array containing non-negative probability densities. The array is not
        modified in-place.
    temperature
        Simulation temperature in Kelvin used to compute :math:`kT`.
    mask
        Optional boolean mask identifying bins that should be reported as NaN
        (typically empty histogram bins). The mask is ignored when ``inpaint``
        is ``True`` because callers have already filled those bins.
    inpaint
        If ``True`` skip applying ``mask`` so that bins filled via inpainting
        remain finite.
    tiny
        Optional floor used to guard against ``log(0)``. Defaults to the
        machine-dependent ``np.finfo(float).tiny``.
    """

    if temperature <= 0:
        raise ValueError("temperature must be positive when computing free energy")

    density_arr = np.array(density, dtype=np.float64, copy=False)
    tiny_val = float(tiny if tiny is not None else np.finfo(np.float64).tiny)
    kT = kT_kJ_per_mol(float(temperature))

    # Avoid RuntimeWarning: divide by zero encountered in log by clipping first
    # and only assigning +inf where true zeros occurred. Using errstate keeps
    # logs clean without changing semantics.
    with np.errstate(divide="ignore", invalid="ignore"):
        log_density = np.log(np.clip(density_arr, tiny_val, None))
        free_energy = -kT * log_density

    result = np.where(density_arr > tiny_val, free_energy, np.inf)
    if mask is not None:
        mask_arr = np.array(mask, dtype=bool, copy=False)
        if not inpaint:
            result = np.where(mask_arr, np.nan, result)

    if np.any(np.isfinite(result)):
        result = result - np.nanmin(result)

    return result


logger = logging.getLogger(__name__)


def _wrap_periodic(angle: np.ndarray) -> np.ndarray:
    """Wrap angular differences into ``[-pi, pi)`` for toroidal kernels."""

    return ((angle + np.pi) % (2.0 * np.pi)) - np.pi


def periodic_kde_2d(
    theta_x: np.ndarray,
    theta_y: np.ndarray,
    bw: Tuple[float, float] = (0.35, 0.35),
    gridsize: Tuple[int, int] = (42, 42),
) -> NDArray[np.float64]:
    """Kernel density estimate on a 2D torus using a wrapped Gaussian mixture."""

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

    # Broadcast over samples (last axis) for vectorised Gaussian evaluation.
    dx = _wrap_periodic(X[..., np.newaxis] - x[np.newaxis, np.newaxis, :])
    dy = _wrap_periodic(Y[..., np.newaxis] - y[np.newaxis, np.newaxis, :])
    inv_cov = (dx / sx) ** 2 + (dy / sy) ** 2
    kernel = np.exp(-0.5 * inv_cov)
    norm = float(x.size) * (2.0 * np.pi * sx * sy)
    if norm <= 0.0:
        raise ValueError("normalisation constant must be positive")
    density = kernel.sum(axis=-1) / norm
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
    F = free_energy_from_density(H, temperature)
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
    config: Any | None = None,
    grid_strategy: str = "adaptive",
) -> FESResult:
    """Generate a two-dimensional free-energy surface (FES).

    Smoothing is disabled by default. When enabled, bins are smoothed only when
    the Dirichlet posterior uncertainty ``SD[F]`` exceeds
    ``fes_target_sd_kT`` (using a pseudocount ``fes_alpha``). The Gaussian
    bandwidth adapts inversely with the effective sample size via
    :func:`adaptive_bandwidth`.

    Parameters
    ----------
    cv1, cv2
        Collective variable samples.
    bins
        Number of histogram bins in (x, y).
    temperature
        Simulation temperature in Kelvin.
    periodic
        Flags indicating whether each dimension is periodic.
    ranges
        Optional histogram ranges as ((xmin, xmax), (ymin, ymax)).
    smooth
        If True, smooth the density with a periodic KDE (deprecated).
    inpaint
        If True, fill empty bins using KDE estimate (deprecated).
    min_count
        Histogram bins with fewer samples are marked as empty.
    kde_bw_deg
        Bandwidth in degrees for periodic KDE.
    epsilon
        Numerical tolerance for bandwidth calculations.
    config
        Optional configuration object supplying fes_* smoothing parameters.
    grid_strategy
        Strategy for grid extent selection: "fixed" uses full data range,
        "adaptive" crops to [q1, q99] percentiles and adjusts bin counts
        to target finite_bins_fraction >= 0.6.

    Returns
    -------
    FESResult
        Dataclass containing the free-energy surface and bin edges.
    """

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

    grid_strategy = str(grid_strategy).lower()
    if grid_strategy not in {"fixed", "adaptive"}:
        raise ValueError("grid_strategy must be 'fixed' or 'adaptive'")

    if ranges is None:
        # Determine ranges based on grid strategy
        if grid_strategy == "adaptive" and not any(periodic):
            # Adaptive: crop to [q1, q99] percentiles to reduce empty bins
            try:
                x_quantiles = mquantiles(x, prob=[0.01, 0.99]).filled(np.nan)
                y_quantiles = mquantiles(y, prob=[0.01, 0.99]).filled(np.nan)
                x_q = np.asarray(x_quantiles, dtype=np.float64)
                y_q = np.asarray(y_quantiles, dtype=np.float64)
                if np.isfinite(x_q).all() and x_q[1] > x_q[0]:
                    xr = (float(x_q[0]), float(x_q[1]))
                else:
                    xr = (float(np.min(x)), float(np.max(x)))
                if np.isfinite(y_q).all() and y_q[1] > y_q[0]:
                    yr = (float(y_q[0]), float(y_q[1]))
                else:
                    yr = (float(np.min(y)), float(np.max(y)))
                # Clip samples into the selected range to keep edge bins populated
                x = np.clip(x, xr[0], xr[1]).astype(np.float64, copy=False)
                y = np.clip(y, yr[0], yr[1]).astype(np.float64, copy=False)
                logger.info(
                    "Adaptive grid: cropped to x=[%.3f, %.3f], y=[%.3f, %.3f]",
                    xr[0], xr[1], yr[0], yr[1]
                )
            except Exception:
                xr = (float(np.min(x)), float(np.max(x)))
                yr = (float(np.min(y)), float(np.max(y)))
        else:
            # Fixed or periodic: use full data range
            try:
                if not any(periodic):
                    x_quantiles = mquantiles(x, prob=[0.01, 0.99]).filled(np.nan)
                    y_quantiles = mquantiles(y, prob=[0.01, 0.99]).filled(np.nan)
                    x_q = np.asarray(x_quantiles, dtype=np.float64)
                    y_q = np.asarray(y_quantiles, dtype=np.float64)
                    if np.isfinite(x_q).all() and x_q[1] > x_q[0]:
                        xr = (float(x_q[0]), float(x_q[1]))
                    else:
                        xr = (float(np.min(x)), float(np.max(x)))
                    if np.isfinite(y_q).all() and y_q[1] > y_q[0]:
                        yr = (float(y_q[0]), float(y_q[1]))
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
    eps = float(epsilon)
    # sqrt rule baseline
    sqrt_bins = max(min_bins, int(np.sqrt(max(1, n_pts))))

    # Freedman–Diaconis per axis computed via SciPy helper to avoid custom logic
    def _fd_bins(arr: NDArray[np.float64], value_range: Tuple[float, float]) -> int:
        arr = np.asarray(arr, dtype=np.float64)
        span = float(value_range[1] - value_range[0])
        if arr.size <= 1 or not np.isfinite(span) or span <= 0:
            return 0
        bandwidth = 2.0 * iqr(arr, rng=(25, 75), nan_policy="omit")
        bandwidth /= np.cbrt(max(1, arr.size))
        if not np.isfinite(bandwidth) or bandwidth <= eps:
            return 0
        nb = int(np.ceil(span / bandwidth))
        if nb <= 0:
            return 0
        return int(np.clip(nb, min_bins, max_bins))

    bx_fd = _fd_bins(x, xr)
    by_fd = _fd_bins(y, yr)

    # For adaptive strategy, adjust bin counts to target finite_bins_fraction >= 0.6
    if grid_strategy == "adaptive":
        # Start with requested bins or FD estimate
        bx_initial = max(int(bx_req), int(bx_fd or 0), int(sqrt_bins))
        by_initial = max(int(by_req), int(by_fd or 0), int(sqrt_bins))

        # Iteratively reduce bins to achieve target finite bins fraction
        target_finite_fraction = 0.6
        max_iterations = 5

        bx, by = bx_initial, by_initial
        for iteration in range(max_iterations):
            # Test current bin counts
            test_x_edges = np.linspace(xr[0], xr[1], bx + 1).astype(np.float64, copy=False)
            test_y_edges = np.linspace(yr[0], yr[1], by + 1).astype(np.float64, copy=False)

            test_H, _, _ = np.histogram2d(x, y, bins=(test_x_edges, test_y_edges))
            test_empty_fraction = float(np.sum(test_H < min_count)) / test_H.size
            test_finite_fraction = 1.0 - test_empty_fraction

            if test_finite_fraction >= target_finite_fraction:
                logger.info(
                    "Adaptive grid converged: bins=(%d, %d), finite_fraction=%.3f",
                    bx, by, test_finite_fraction
                )
                break

            # Reduce bins by ~15% per axis to increase density
            reduction_factor = 0.85
            bx = max(min_bins, int(bx * reduction_factor))
            by = max(min_bins, int(by * reduction_factor))

            if iteration == max_iterations - 1:
                logger.info(
                    "Adaptive grid: reached max iterations with bins=(%d, %d), finite_fraction=%.3f",
                    bx, by, test_finite_fraction
                )
    else:
        # Fixed strategy: use standard bin selection
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
    H_density: NDArray[np.float64] = H_counts.astype(np.float64, copy=False)
    base_mask: NDArray[np.bool_] = H_counts < min_count

    total_count = float(H_counts.sum())
    if total_count <= 0.0:
        raise ValueError("Histogram counts sum to zero; cannot compute FES")
    density: NDArray[np.float64] = H_density / (total_count * bin_area)

    F_masked: NDArray[np.float64] = free_energy_from_density(
        density,
        temperature,
        mask=base_mask,
        inpaint=False,
    )

    finite_mask = np.isfinite(F_masked)
    if not finite_mask.any():
        raise ValueError("No finite free-energy values available for smoothing")
    fill_value = float(np.nanmax(F_masked[finite_mask]))
    F_numeric = np.where(finite_mask, F_masked, fill_value)

    config_obj: Any | None
    if isinstance(config, Mapping):
        config_obj = dict(config)
    elif config is not None:
        config_obj = {"__base_config__": config}
    else:
        config_obj = None

    def _config_get(name: str, default: Any) -> Any:
        if config_obj is None:
            return default
        if isinstance(config_obj, Mapping):
            if name in config_obj:
                return config_obj[name]
            base = config_obj.get("__base_config__")
            if base is not None:
                if isinstance(base, Mapping):
                    return base.get(name, default)
                return getattr(base, name, default)
            return default
        return getattr(config_obj, name, default)

    def _config_has(name: str) -> bool:
        if config_obj is None:
            return False
        if isinstance(config_obj, Mapping):
            if name in config_obj:
                return True
            base = config_obj.get("__base_config__")
            if base is None:
                return False
            if isinstance(base, Mapping):
                return name in base
            return hasattr(base, name)
        return hasattr(config_obj, name)

    deprecated_flags = (
        ("fes_empty_threshold", "--fes-empty-threshold"),
        ("fes_force_smooth_threshold", "--fes-force-smooth-threshold"),
    )
    for attr_name, cli_flag in deprecated_flags:
        if _config_has(attr_name):
            warnings.warn(
                f"{cli_flag} is deprecated and ignored. Use --fes-smoothing-mode/--fes-target-sd-kT.",
                DeprecationWarning,
                stacklevel=2,
            )

    mode_cfg = _config_get("fes_smoothing_mode", None)
    if mode_cfg is None:
        if smooth:
            warnings.warn(
                "The 'smooth' argument is deprecated; use fes_smoothing_mode instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            mode = "always"
        elif inpaint:
            warnings.warn(
                "The 'inpaint' argument is deprecated; use fes_smoothing_mode='auto' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            mode = "auto"
        else:
            mode = "never"
    else:
        mode = str(mode_cfg).lower()
        if smooth or inpaint:
            warnings.warn(
                "The smooth/inpaint arguments are ignored when fes_smoothing_mode is provided.",
                DeprecationWarning,
                stacklevel=2,
            )
    if mode not in {"never", "auto", "always"}:
        raise ValueError(f"Unknown fes_smoothing_mode={mode!r}")

    target_sd_cfg = _config_get("fes_target_sd_kT", None)
    target_sd_val = float(target_sd_cfg) if target_sd_cfg is not None else 0.5
    alpha = float(_config_get("fes_alpha", 1e-6))
    h0 = float(_config_get("fes_h0", 1.2))
    ess_ref = float(_config_get("fes_ess_ref", 50.0))
    h_min = float(_config_get("fes_h_min", 0.4))
    h_max = float(_config_get("fes_h_max", 3.0))
    if alpha <= 0:
        raise ValueError("fes_alpha must be positive")
    if h0 <= 0:
        raise ValueError("fes_h0 must be positive")
    if ess_ref <= 0:
        raise ValueError("fes_ess_ref must be positive")
    if h_min <= 0 or h_max <= 0 or h_min > h_max:
        raise ValueError("fes_h_min and fes_h_max must be positive with h_min <= h_max")

    kT_val = kT_kJ_per_mol(float(temperature))
    beta = 1.0 / kT_val
    kT_energy = beta_to_kT(beta)

    sd_map: NDArray[np.float64] | None = None
    bandwidth_map: NDArray[np.float64] | None = None
    smoothing_mask: NDArray[np.bool_] = np.zeros_like(base_mask, dtype=bool)
    if mode == "never":
        F_smoothed = F_numeric
        applied_mask = np.zeros_like(base_mask, dtype=bool)
    else:
        smoothing_mask_raw, sd_map = mark_bins_for_smoothing(
            H_counts,
            target_sd_kT=target_sd_val,
            alpha=alpha,
            kT=kT_energy,
        )
        smoothing_mask = np.asarray(smoothing_mask_raw, dtype=bool)
        ess_map = H_counts.astype(float)
        bandwidth_map = adaptive_bandwidth(
            ess_map,
            h0=h0,
            ess_ref=ess_ref,
            h_min=h_min,
            h_max=h_max,
        )
        apply_mask = smoothing_mask if mode == "auto" else None
        F_smoothed = smooth_F_with_adaptive_gaussian(
            F_numeric,
            h_map=bandwidth_map,
            apply_mask=apply_mask,
        )
        if mode == "always":
            applied_mask = np.ones_like(base_mask, dtype=bool)
        else:
            applied_mask = smoothing_mask

    final_mask = np.asarray(base_mask & ~applied_mask, dtype=bool)
    F_result = np.where(final_mask, np.nan, F_smoothed)
    finite_final = np.isfinite(F_result)
    if finite_final.any():
        F_result = F_result - float(np.nanmin(F_result[finite_final]))

    empty_bins_fraction = float(np.count_nonzero(base_mask)) / np.prod(H_counts.shape)

    smoothing_meta: dict[str, Any] = {
        "mode": mode,
        "target_sd_kT": float(target_sd_val),
        "alpha": float(alpha),
        "h0": float(h0),
        "ess_ref": float(ess_ref),
        "h_min": float(h_min),
        "h_max": float(h_max),
        "applied_fraction": float(np.mean(applied_mask.astype(float))),
    }
    if sd_map is not None:
        smoothing_meta["sd_map_kT"] = sd_map
    if bandwidth_map is not None:
        smoothing_meta["bandwidth_map"] = bandwidth_map
    if smoothing_mask.any():
        smoothing_meta["mask"] = smoothing_mask

    metadata = {
        "counts": density,
        "periodic": periodic,
        "temperature": temperature,
        "mask": final_mask,
        "empty_bins_fraction": empty_bins_fraction,
        "smoothing": smoothing_meta,
        "grid_strategy": grid_strategy,
        "grid_shape": (bx, by),
        "grid_ranges": {"x": xr, "y": yr},
    }

    # Enhanced sparse FES guardrail
    if empty_bins_fraction > 0.50:
        metadata["sparse_warning"] = (
            f"Sparse FES: {empty_bins_fraction*100.0:.1f}% empty bins detected. "
            f"Grid: {bx}×{by}. Consider using grid_strategy='adaptive' to reduce waste."
        )
        logger.warning(
            "Sparse FES detected: %.1f%% empty bins (grid=%dx%d, strategy=%s)",
            empty_bins_fraction * 100.0, bx, by, grid_strategy
        )

    result = FESResult(F=F_result, xedges=xedges, yedges=yedges, metadata=metadata)

    masked_fraction = float(final_mask.sum()) / np.prod(result.output_shape)
    logger.info("FES masked fraction=%0.3f", masked_fraction)

    return result


class FESCalculator:
    """Calculates Free Energy Surfaces (FES) from MSM results and projected data."""

    def __init__(self, config: dict[str, Any]):
        """
        Initializes the FESCalculator.

        :param config: Configuration dictionary, potentially containing settings
                       like temperature or default bin numbers.
        """
        self.config = config
        self.temperature = config.get(
            "temperature", 300.0
        )  # Example: Get temp from config
        # Convert temperature to energy units (kT in kJ/mol)
        kb = 0.00831446261815324  # Boltzmann constant in kJ/(mol*K)
        self.kbt = kb * self.temperature
        logger.info(
            f"FESCalculator initialized with T={self.temperature}K (kBT={self.kbt:.3f} kJ/mol)"
        )

    def calculate_fes(
        self,
        projection: List[np.ndarray],
        msm: Optional[object],  # More generic type hint for MSM object
        dtrajs: Optional[List[np.ndarray]] = None,  # Allow passing dtrajs explicitly
        bins: int = 150,
        max_energy_cap_kt: Optional[float] = 10.0,  # Cap in kT units
        dim_x: int = 0,
        dim_y: int = 1,
    ) -> Tuple[Optional[List[np.ndarray]], Optional[np.ndarray]]:
        """
        Calculates the 2D Free Energy Surface using MSM stationary distribution.

        :param projection: List of projected trajectory arrays (e.g., TICA output).
        :param msm: The estimated Markov State Model object (must have `stationary_distribution`).
        :param dtrajs: Discrete trajectories corresponding to the projection data.
                       If None, attempts to get them from the msm object.
        :param bins: Number of bins for the 2D histogram.
        :param max_energy_cap_kt: Maximum energy value (in kT) to cap the FES at for visualization.
                                  Set to None to disable capping.
        :param dim_x: Index of the first dimension (e.g., TICA component) to use.
        :param dim_y: Index of the second dimension (e.g., TICA component) to use.
        :return: Tuple containing (grid [xx, yy], fes_array_in_kt), or (None, None) on error.
        """
        if not projection:
            logger.error("Projection data is empty. Cannot calculate FES.")
            return None, None

        if msm is None:
            logger.error("MSM object is required for FES calculation.")
            return None, None

        if not hasattr(msm, "stationary_distribution"):
            logger.error(
                "MSM object does not have 'stationary_distribution' attribute."
            )
            return None, None

        # Try to get dtrajs from MSM object if not provided
        if dtrajs is None:
            if hasattr(msm, "discrete_trajectories"):
                dtrajs = getattr(msm, "discrete_trajectories")  # type: ignore[attr-defined]
            elif hasattr(msm, "_dtrajs"):  # Common private attribute name
                dtrajs = getattr(msm, "_dtrajs")  # type: ignore[attr-defined]
            else:
                logger.error(
                    "Discrete trajectories (dtrajs) not provided and not found on MSM object."
                )
                return None, None

        dtraj_arrays: List[np.ndarray]
        if isinstance(dtrajs, np.ndarray):
            dtraj_arrays = [np.asarray(dtrajs, dtype=int)]
        elif isinstance(dtrajs, Sequence):
            dtraj_arrays = [np.asarray(traj, dtype=int) for traj in dtrajs]  # type: ignore[arg-type]
        else:
            logger.error("Discrete trajectories must be a sequence of arrays.")
            return None, None

        # Validate dimensions
        if projection[0].shape[1] <= max(dim_x, dim_y):
            logger.error(
                f"Projection data has only {projection[0].shape[1]} dimensions, "
                f"but requested dimensions {dim_x} and {dim_y}."
            )
            return None, None

        logger.info(f"Calculating FES with bins={bins}, dims=({dim_x}, {dim_y})...")

        try:
            # Concatenate data for selected dimensions
            x_data = np.concatenate([p[:, dim_x] for p in projection])
            y_data = np.concatenate([p[:, dim_y] for p in projection])
            concatenated_dtrajs = np.concatenate(dtraj_arrays)

            # --- Weighting using stationary distribution ---
            pi_raw = getattr(msm, "stationary_distribution", None)
            if pi_raw is None:
                logger.error("MSM stationary distribution is empty or invalid.")
                return None, None

            pi = np.asarray(pi_raw, dtype=float)
            if pi.size == 0:
                logger.error("MSM stationary distribution is empty or invalid.")
                return None, None

            # Check consistency
            n_frames_proj = sum(len(p) for p in projection)
            n_frames_dtrajs = sum(len(d) for d in dtraj_arrays)
            if n_frames_proj != n_frames_dtrajs:
                logger.error(
                    f"Frame count mismatch: Projection ({n_frames_proj}) vs Dtrajs ({n_frames_dtrajs})."
                )
                return None, None
            if len(x_data) != len(concatenated_dtrajs):
                logger.error(
                    f"Length mismatch after concatenation: Projection ({len(x_data)}) vs Dtrajs ({len(concatenated_dtrajs)})."
                )
                return None, None

            # Map discrete states to weights (stationary probabilities)
            # Ensure indices are valid
            max_state_index = np.max(concatenated_dtrajs)
            if max_state_index >= len(pi):
                logger.warning(
                    f"Max discrete state index ({max_state_index}) >= length of pi ({len(pi)}). "
                    "Some states might be unvisited or outside the core set. Proceeding cautiously."
                )
                # Filter out invalid indices if necessary, though ideally clustering handles this
                valid_mask = concatenated_dtrajs < len(pi)
                if not np.all(valid_mask):
                    logger.warning(
                        f"Filtering {np.sum(~valid_mask)} frames with invalid state indices."
                    )
                    x_data = x_data[valid_mask]
                    y_data = y_data[valid_mask]
                    concatenated_dtrajs = concatenated_dtrajs[valid_mask]
                    if len(x_data) == 0:
                        logger.error(
                            "No valid frames remaining after filtering invalid state indices."
                        )
                        return None, None

            weights = pi[concatenated_dtrajs]

            # --- Calculate weighted 2D histogram ---
            hist, x_edges, y_edges = np.histogram2d(
                x_data, y_data, bins=bins, weights=weights, density=True
            )

            # Calculate bin centers
            x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
            y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])

            # Create grid for plotting
            xx, yy = np.meshgrid(x_centers, y_centers)
            grid = [xx, yy]

            # --- Calculate Free Energy ---
            # Avoid log(0) - replace zero probabilities with a very small number
            min_prob = np.finfo(
                hist.dtype
            ).tiny  # Smallest representable positive number
            hist = np.maximum(hist, min_prob)

            # FES = -kT * ln(Probability)
            # Probability is directly given by the density=True histogram
            fes_kj_mol = -self.kbt * np.log(hist)

            # Shift minimum to zero for relative free energies
            fes_kj_mol -= np.min(fes_kj_mol)

            # Convert to kT units for plotting consistency
            # (Use self.kbt which is already in kJ/mol)
            # fes_kt = fes_kj_mol # If kBT was 1, no change
            # Actually need to divide by kBT to get kT units
            fes_kt = fes_kj_mol / self.kbt

            # Optional: Cap the energy
            if max_energy_cap_kt is not None:
                fes_kt = np.clip(fes_kt, a_min=0, a_max=max_energy_cap_kt)
                logger.info(f"FES capped at {max_energy_cap_kt:.2f} kT.")

            logger.info("...FES calculation complete.")
            return grid, fes_kt

        except Exception as e:
            logger.exception(f"Error during FES calculation: {e}")
            return None, None
