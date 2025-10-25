from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from pmarlo import constants as const
from pmarlo.utils.path_utils import ensure_directory
from pmarlo.utils.thermodynamics import kT_kJ_per_mol


def save_transition_matrix_heatmap(
    T: np.ndarray, output_dir: str, name: str = "T_heatmap.png"
) -> Optional[str]:
    """Save a heatmap of a transition matrix to ``output_dir``.

    Returns the path to the written file if successful, otherwise ``None``.
    """

    out_dir = Path(output_dir)
    ensure_directory(out_dir)
    plt.figure(figsize=const.PLOT_FIGURE_SIZE_HEATMAP)
    plt.imshow(T, cmap="viridis", origin="lower")
    plt.colorbar(label="Transition Probability")
    plt.xlabel("j")
    plt.ylabel("i")
    plt.title("Transition Matrix")
    filepath = out_dir / name
    plt.tight_layout()
    plt.savefig(filepath, dpi=const.PLOT_DPI)
    plt.close()
    return str(filepath) if filepath.exists() else None


def save_fes_contour(
    F: np.ndarray,
    xedges: np.ndarray,
    yedges: np.ndarray,
    xlabel: str,
    ylabel: str,
    output_dir: str,
    filename: str,
    mask: Optional[np.ndarray] = None,
) -> Optional[str]:
    out_dir = Path(output_dir)
    ensure_directory(out_dir)
    x_centers = const.PLOT_BIN_EDGE_CENTER_FACTOR * (xedges[:-1] + xedges[1:])
    y_centers = const.PLOT_BIN_EDGE_CENTER_FACTOR * (yedges[:-1] + yedges[1:])
    plt.figure(figsize=const.PLOT_FIGURE_SIZE_FES_CONTOUR)
    finite_mask = np.isfinite(F)
    if F.size == 0:
        raise ValueError("F must contain at least one element")
    if not finite_mask.any():
        raise ValueError("FES contains no finite values; cannot render contour plot")

    empty_frac = 1.0 - float(np.count_nonzero(finite_mask)) / float(F.size)
    if empty_frac > const.FES_SPARSITY_ERROR_THRESHOLD:
        raise ValueError(
            f"FES is too sparse to plot reliably ({empty_frac*const.FES_PERCENTAGE_SCALE:.1f}% empty bins)"
        )

    F_for_plot = np.where(finite_mask, F, np.nan)
    c = plt.contourf(x_centers, y_centers, F_for_plot.T, levels=const.PLOT_CONTOUR_LEVELS, cmap="viridis")
    plt.colorbar(c, label="Free Energy (kJ/mol)")
    title_warn = ""
    if mask is not None:
        m = np.ma.masked_where(~mask.T, mask.T)
        plt.contourf(
            x_centers,
            y_centers,
            m,
            levels=list(const.PLOT_MASK_LEVELS),
            colors="none",
            hatches=["////"],
        )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"FES ({xlabel} vs {ylabel}){title_warn}")
    filepath = out_dir / filename
    plt.tight_layout()
    plt.savefig(filepath, dpi=const.PLOT_DPI)
    plt.close()
    return str(filepath) if filepath.exists() else None


def save_pmf_line(
    F: np.ndarray,
    edges: np.ndarray,
    xlabel: str,
    output_dir: str,
    filename: str,
) -> Optional[str]:
    """Save a 1D PMF line plot to ``output_dir``.

    Parameters
    ----------
    F:
        1D free energy values per bin (kJ/mol).
    edges:
        Bin edges of shape (n_bins + 1,).
    xlabel:
        Label for the x-axis.
    output_dir:
        Directory to write the plot into.
    filename:
        Output filename (e.g., "pmf_universal_metric.png").
    """
    out_dir = Path(output_dir)
    ensure_directory(out_dir)
    x_centers = const.PLOT_BIN_EDGE_CENTER_FACTOR * (edges[:-1] + edges[1:])
    plt.figure(figsize=const.PLOT_FIGURE_SIZE_PMF_LINE)
    plt.plot(x_centers, F, color="steelblue", lw=const.PLOT_LINE_WIDTH)
    plt.xlabel(xlabel)
    plt.ylabel("Free Energy (kJ/mol)")
    plt.title("1D PMF")
    filepath = out_dir / filename
    plt.tight_layout()
    plt.savefig(filepath, dpi=const.PLOT_DPI)
    plt.close()
    return str(filepath) if filepath.exists() else None


def fes2d(
    x,
    y,
    bins: int = 100,
    adaptive: bool = False,
    temperature: float = 300.0,
    min_count: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str | None]:
    """Compute a simple 2D FES with optional adaptive binning.

    Parameters
    ----------
    x, y: array-like
        CV samples of equal length
    bins: int
        Target number of bins per axis for non-adaptive mode
    adaptive: bool
        If True, use 1%–99% quantiles to define ranges and choose bins based on sample size
    temperature: float
        Temperature in Kelvin for kT scaling
    min_count: int
        Minimum count to treat a bin as occupied

    Returns
    -------
    F, xedges, yedges, warn
        F is in kJ/mol; warn is a human-readable warning or None
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    if x.size == 0 or y.size == 0 or x.shape != y.shape:
        raise ValueError("x and y must be non-empty and have the same shape")

    if adaptive:
        x_lo, x_hi = np.quantile(x, [const.FES_QUANTILE_LOW, const.FES_QUANTILE_HIGH])
        y_lo, y_hi = np.quantile(y, [const.FES_QUANTILE_LOW, const.FES_QUANTILE_HIGH])
        # Avoid zero-width ranges
        if (
            not np.isfinite([x_lo, x_hi, y_lo, y_hi]).all()
            or x_lo >= x_hi
            or y_lo >= y_hi
        ):
            x_lo, x_hi = float(np.min(x)), float(np.max(x))
            y_lo, y_hi = float(np.min(y)), float(np.max(y))
            if (
                not np.isfinite([x_lo, x_hi, y_lo, y_hi]).all()
                or x_lo >= x_hi
                or y_lo >= y_hi
            ):
                # Degenerate, return a trivial surface
                xe = np.linspace(
                    float(np.min(x)),
                    float(np.max(x)) + const.NUMERIC_RELATIVE_TOLERANCE,
                    const.FES_DEGENERATE_BINS,
                )
                ye = np.linspace(
                    float(np.min(y)),
                    float(np.max(y)) + const.NUMERIC_RELATIVE_TOLERANCE,
                    const.FES_DEGENERATE_BINS,
                )
                H = np.zeros((len(xe) - 1, len(ye) - 1), dtype=float)
                return np.full_like(H, np.nan), xe, ye, "Invalid FES ranges"
        nb = max(const.FES_ADAPTIVE_MIN_BINS, int(np.sqrt(len(x)) / const.FES_ADAPTIVE_BIN_DIVISOR))
        H, xe, ye = np.histogram2d(x, y, bins=nb, range=[[x_lo, x_hi], [y_lo, y_hi]])
    else:
        nb = int(bins)
        if nb <= 0:
            nb = const.FES_DEFAULT_BINS
        H, xe, ye = np.histogram2d(x, y, bins=nb)

    empty = (H < max(1, int(min_count))).mean() * const.FES_PERCENTAGE_SCALE
    if empty > const.FES_SPARSITY_WARNING_THRESHOLD * const.FES_PERCENTAGE_SCALE:
        warn = (
            f"Sparse FES: {empty:.1f}% empty bins (try adaptive bins or more sampling)"
        )
    else:
        warn = None

    kT = kT_kJ_per_mol(float(temperature))
    F = -kT * np.log(H + const.NUMERIC_MIN_POSITIVE)
    # Assign +inf to truly empty (below min_count) bins to avoid misleading minima
    F = np.where(H >= max(1, int(min_count)), F, np.inf)
    if np.any(np.isfinite(F)):
        F -= np.nanmin(F)
    return F.astype(float, copy=False), xe, ye, warn
