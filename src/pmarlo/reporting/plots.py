from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from pmarlo import constants as const
from pmarlo.utils.path_utils import ensure_directory
from pmarlo.utils.thermodynamics import kT_kJ_per_mol

logger = logging.getLogger(__name__)


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


def plot_sampling_validation(
    projected_data_1d: list[np.ndarray],
    max_traj_length_plot: int = 5000,
    bins: int = 50,
    ax: Optional[plt.Axes] = None,
    max_legend_entries: int = 20,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot 1D histograms and trajectory traces to validate sampling connectivity.

    This plot shows the 1D histogram (approximating the free energy landscape)
    and overlays the path of each individual shard (trajectory) to visually
    inspect if all basins are connected and reversibly sampled.

    Parameters
    ----------
    projected_data_1d
        List of 1D numpy arrays (e.g., [traj[:, 0] for traj in projection]).
    max_traj_length_plot
        Maximum frames to plot for trajectory traces (for clarity).
    bins
        Number of bins for the histogram (default 50 to reduce complexity).
    ax
        Matplotlib axis to plot on. If None, creates a new figure.
    max_legend_entries
        Maximum number of trajectories to show in legend. If there are more
        trajectories than this, no legend is shown to avoid image size issues.

    Returns
    -------
    fig, ax
        Matplotlib Figure and Axes objects.
    """
    logger.info(f"Starting sampling validation plot with {len(projected_data_1d)} trajectories")

    if ax is None:
        # Use constrained_layout instead of tight_layout to better control figure size
        fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True, dpi=100)
    else:
        fig = ax.get_figure()

    if not projected_data_1d:
        ax.text(0.5, 0.5, "No projection data available", ha="center", va="center")
        ax.axis("off")
        return fig, ax

    logger.info("Generating color map for trajectories")
    colors = plt.cm.jet(np.linspace(0, 1, len(projected_data_1d)))

    logger.info(f"Plotting histograms for {len(projected_data_1d)} trajectories...")
    for i, traj in enumerate(projected_data_1d):
        if traj.size == 0:
            continue
        if i % 5 == 0:  # Log every 5th trajectory to avoid spam
            logger.info(f"  Histogram {i+1}/{len(projected_data_1d)} (length: {len(traj)} frames)")
        ax.hist(traj, bins=bins, alpha=0.3, density=True, color=colors[i])

    logger.info("All histograms plotted, getting axis limits")
    ylims = ax.get_ylim()
    xlims = ax.get_xlim()

    # Determine if we should show legend entries
    show_legend = len(projected_data_1d) <= max_legend_entries

    logger.info(f"Plotting trajectory traces (max {max_traj_length_plot} frames each)...")
    for i, traj in enumerate(projected_data_1d):
        if traj.size == 0:
            continue
        plot_len = min(len(traj), max_traj_length_plot)
        if i % 5 == 0:  # Log every 5th trajectory
            logger.info(f"  Trajectory {i+1}/{len(projected_data_1d)} (plotting {plot_len} frames)")
        traj_segment = traj[:plot_len]
        y_time = np.linspace(ylims[0], ylims[1] * 0.9, plot_len)

        # Only add label if we're showing the legend
        label = f"Shard {i}" if show_legend else None
        ax.plot(
            traj_segment,
            y_time,
            alpha=0.5,
            color=colors[i],
            lw=0.5,
            label=label,
        )

    logger.info("Adding annotations and finalizing plot...")
    ax.annotate(
        "",
        xy=(0.95 * xlims[1], 0.7 * ylims[1]),
        xytext=(0.95 * xlims[1], 0.3 * ylims[1]),
        arrowprops=dict(fc="gray", ec="none", alpha=0.6, width=2),
    )
    ax.text(
        0.96 * xlims[1],
        0.5 * ylims[1],
        "time",
        ha="left",
        va="center",
        rotation=90,
        color="gray",
    )

    ax.set_xlabel("TICA Component 1")
    ax.set_ylabel("Histogram Density / Trajectory Time")
    ax.set_ylim(ylims)
    ax.set_xlim(xlims)

    # Only show legend if we have a reasonable number of trajectories
    if show_legend:
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), title="Trajectories")
    else:
        logger.info(f"Skipping legend ({len(projected_data_1d)} trajectories > {max_legend_entries} max)")


    logger.info("Sampling validation plot complete")
    return fig, ax


def plot_free_energy_2d(
    grid: Tuple[np.ndarray, np.ndarray],
    fes: np.ndarray,
    ax: Optional[plt.Axes] = None,
    cmap: str = "viridis",
    levels: int = 20,
    max_energy_kj_per_mol: float = 50.0,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot a 2D Free Energy Surface (FES) contour plot.

    Parameters
    ----------
    grid
        The coordinate grid, typically [xx, yy] from meshgrid or FES calculator.
    fes
        The 2D free energy array (in kJ/mol).
    ax
        Matplotlib axis to plot on. If None, creates a new figure.
    cmap
        Colormap to use.
    levels
        Number of contour levels.
    max_energy_kj_per_mol
        Maximum free energy to plot (in kJ/mol). Energies above this will be capped.

    Returns
    -------
    fig, ax
        Matplotlib Figure and Axes objects.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.get_figure()

    xx, yy = grid
    capped_fes = np.clip(fes, a_min=0, a_max=max_energy_kj_per_mol)

    contour = ax.contourf(xx, yy, capped_fes.T, levels=levels, cmap=cmap, extend="max")
    cbar = fig.colorbar(contour, ax=ax, label="Free Energy (kJ/mol)")

    ax.set_xlabel("TICA Component 1")
    ax.set_ylabel("TICA Component 2")
    ax.set_title("Free Energy Surface")
    fig.tight_layout()

    return fig, ax
