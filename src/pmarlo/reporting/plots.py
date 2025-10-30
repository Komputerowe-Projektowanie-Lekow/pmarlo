from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple, List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

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
    projected_data_1d: List[np.ndarray],
    max_traj_length_plot: int = 1000,
    bins: int = 150,  # Increased default bins
    stride: int = 10,
    alpha_hist: float = 0.15,
    alpha_traj: float = 0.4,
    lw_traj: float = 0.3,
    cmap_name: str = "viridis",
    ax: Optional[plt.Axes] = None,
) -> Figure:
    """
    Plots 1D histograms and trajectory traces to validate sampling connectivity.

    This plot shows the 1D histogram (approximating the free energy landscape)
    and overlays the path of each individual shard (trajectory) to visually
    inspect if all basins are connected and reversibly sampled.

    :param projected_data_1d: List of 1D numpy arrays (e.g., [traj[:, 0] for traj in projection]).
    :param max_traj_length_plot: Max frames to plot for trajectory traces (for clarity).
    :param bins: Number of bins for the histogram.
    :param stride: Plot every N-th point for trajectory traces to reduce density.
    :param alpha_hist: Transparency for histograms.
    :param alpha_traj: Transparency for trajectory lines.
    :param lw_traj: Line width for trajectory lines.
    :param cmap_name: Colormap name for trajectory lines.
    :param ax: Matplotlib axis to plot on.
    :return: Matplotlib Figure.
    """
    if not projected_data_1d:
        raise ValueError("projected_data_1d must contain at least one trajectory")

    empty_trajectories = [idx for idx, traj in enumerate(projected_data_1d) if len(traj) == 0]
    if empty_trajectories:
        raise ValueError(
            "projected_data_1d contains empty trajectories at indices: "
            + ", ".join(str(i) for i in empty_trajectories)
        )

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.get_figure()

    # Generate colors
    try:
        cmap = plt.get_cmap(cmap_name)
    except ValueError as exc:
        raise ValueError(f"Unknown colormap '{cmap_name}'") from exc
    colors = cmap(np.linspace(0, 1, len(projected_data_1d)))

    # --- 1. Plot Histograms ---
    # Combine all data for a single representative histogram
    all_data_1d = np.concatenate(projected_data_1d)
    ax.hist(all_data_1d, bins=bins, alpha=alpha_hist, density=True, color='grey', label='Overall Distribution')

    ylims = ax.get_ylim()
    xlims = ax.get_xlim()

    # --- 2. Plot Trajectory Traces ---
    num_shards_to_label = min(15, len(projected_data_1d)) # Limit legend entries
    for i, traj in enumerate(projected_data_1d):
        logger.debug(f"Shard {i}: traj length = {len(traj)}")

        plot_len = min(len(traj), max_traj_length_plot)
        logger.debug(f"Shard {i}: plot_len = {plot_len}, max_traj_length_plot = {max_traj_length_plot}")

        traj_segment = traj[:plot_len:stride]
        actual_plot_len = len(traj_segment)

        logger.debug(f"Shard {i}: stride = {stride}, actual_plot_len = {actual_plot_len}")
        logger.debug(f"Shard {i}: traj_segment shape = {traj_segment.shape}, dtype = {traj_segment.dtype}")

        # Log first few values for the first 3 shards
        if i < 3:
            logger.debug(f"Shard {i}: traj_segment[:10] = {traj_segment[:10]}")

        if actual_plot_len > 1:
            y_time = np.linspace(ylims[0], ylims[1] * 0.9, actual_plot_len)
            label = f"Shard {i}" if i < num_shards_to_label else None
            ax.plot(
                traj_segment,
                y_time,
                alpha=alpha_traj,
                color=colors[i],
                lw=lw_traj,
                label=label,
            )
            logger.debug(f"Shard {i}: PLOTTED with y_time range [{y_time[0]:.4f}, {y_time[-1]:.4f}]")

    # --- 3. Add Time Arrow ---
    # Ensure arrow/text are placed reasonably within potentially changed xlims
    arrow_x = xlims[0] + 0.95 * (xlims[1] - xlims[0])
    text_x = xlims[0] + 0.96 * (xlims[1] - xlims[0])
    ax.annotate(
        "",
        xy=(arrow_x, 0.7 * ylims[1]),
        xytext=(arrow_x, 0.3 * ylims[1]),
        arrowprops=dict(fc="gray", ec="none", alpha=0.6, width=2),
    )
    ax.text(
        text_x,
        0.5 * ylims[1],
        "$x(time)$",
        ha="left",
        va="center",
        rotation=90,
        color="gray",
    )

    ax.set_xlabel("TICA Component 1")
    ax.set_ylabel("Histogram Density / Trajectory Time")
    ax.set_ylim(ylims)
    ax.set_xlim(xlims)

    handles, labels = ax.get_legend_handles_labels()
    # Only show legend if there are labels
    if labels:
         # Place legend outside plot area
        ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.02, 1), title=f"Shards (Top {num_shards_to_label})")

    try:
        fig.tight_layout(rect=(0, 0, 0.85, 1)) # Adjust layout to make space for legend
    except ValueError:
        logger.warning("Could not adjust layout for sampling validation plot legend.")
        fig.tight_layout()

    return fig


def plot_free_energy_2d(
    grid: List[np.ndarray],
    fes: np.ndarray,
    ax: Optional[plt.Axes] = None,
    cmap: str = "viridis",
    levels: int = 25, # Increased levels for smoother look
    max_energy_kt: float = 7.0, # Reduced default cap
    add_contour_lines: bool = True,
    line_color: str = 'black',
    line_width: float = 0.5,
    line_alpha: float = 0.6,
) -> Figure:
    """
    Plots a 2D Free Energy Surface (FES) contour plot.

    :param grid: The coordinate grid [xx, yy] from FESCalculator.
    :param fes: The 2D free energy array from FESCalculator.
    :param ax: Matplotlib axis to plot on.
    :param cmap: Colormap to use.
    :param levels: Number of contour levels.
    :param max_energy_kt: Max free energy to plot (in kT). Energies above
                          this will be capped.
    :param add_contour_lines: Whether to overlay contour lines.
    :param line_color: Color for contour lines.
    :param line_width: Line width for contour lines.
    :param line_alpha: Transparency for contour lines.
    :return: Matplotlib Figure.
    """
    if not isinstance(grid, (list, tuple)) or len(grid) < 2:
        raise ValueError("grid must contain two coordinate arrays (xx, yy)")

    xx = np.asarray(grid[0], dtype=float)
    yy = np.asarray(grid[1], dtype=float)
    fes_array = np.asarray(fes, dtype=float)

    if xx.ndim != 2 or yy.ndim != 2 or fes_array.ndim != 2:
        raise ValueError("grid coordinates and fes must be 2-dimensional arrays")

    if xx.shape != yy.shape:
        raise ValueError("grid coordinate arrays must have the same shape")

    if fes_array.shape != xx.shape:
        raise ValueError("fes must have the same shape as the coordinate grid")

    if not np.isfinite(xx).all() or not np.isfinite(yy).all():
        raise ValueError("grid coordinates must be finite")

    finite_mask = np.isfinite(fes_array)
    if not finite_mask.any():
        raise ValueError("fes contains no finite values; cannot plot surface")

    finite_min = float(fes_array[finite_mask].min())
    max_energy_kt = float(max_energy_kt)
    if not np.isfinite(max_energy_kt) or max_energy_kt <= finite_min:
        raise ValueError("max_energy_kt must be finite and greater than the minimum fes value")

    capped_fes = np.clip(fes_array, a_min=finite_min, a_max=max_energy_kt)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.get_figure()

    contour = ax.contourf(
        xx,
        yy,
        capped_fes.T,
        levels=levels,
        cmap=cmap,
        extend="max",
    )
    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label("Free Energy ($k_B T$)")

    if add_contour_lines:
        ax.contour(
            xx, yy, capped_fes.T, levels=levels, colors=line_color,
            linewidths=line_width, alpha=line_alpha
        )

    ax.set_xlabel("TICA Component 1")
    ax.set_ylabel("TICA Component 2")
    ax.set_title("Free Energy Surface")
    fig.tight_layout()

    return fig
