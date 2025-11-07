from __future__ import annotations

"""Reusable diagnostics visualisations for sampling and free-energy validation.

These helpers are UI-agnostic wrappers around :mod:`pmarlo.reporting.plots`
that expose a concise, data-oriented API for downstream consumers (CLI
workflows, notebooks, or the Streamlit web app).
"""

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from pmarlo.reporting import plots as reporting_plots

__all__ = [
    "create_sampling_validation_plot",
    "create_fes_validation_plot",
]


@dataclass(frozen=True)
class _SamplingInputs:
    projected_data_1d: List[np.ndarray]
    trajectory_labels: List[str] | None
    discrete_overlay: List[np.ndarray] | None


def _validate_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer (got {value!r})")


def _ensure_figsize(name: str, value: tuple[float, float]) -> tuple[float, float]:
    if len(value) != 2:
        raise ValueError(f"{name} must contain two positive floats (width, height)")
    width, height = value
    if width <= 0 or height <= 0:
        raise ValueError(f"{name} values must be positive (got {value!r})")
    return float(width), float(height)


def _normalise_projection_data(
    projection_data: Sequence[np.ndarray | Sequence[float]],
    *,
    component: int = 0,
) -> tuple[list[np.ndarray], list[int]]:
    if len(projection_data) == 0:
        raise ValueError("projection_data must contain at least one trajectory")

    kept_indices: list[int] = []
    projected: list[np.ndarray] = []
    for idx, trajectory in enumerate(projection_data):
        arr = np.asarray(trajectory, dtype=float)
        if arr.size == 0:
            continue
        if arr.ndim == 1:
            arr1d = arr.reshape(-1)
        elif arr.ndim == 2:
            if arr.shape[1] == 0:
                continue
            if component >= arr.shape[1]:
                raise ValueError(
                    "Requested component index exceeds trajectory feature count: "
                    f"component={component}, columns={arr.shape[1]}"
                )
            arr1d = arr[:, component]
        else:
            raise ValueError(
                "Each trajectory must be 1D or 2D; "
                f"trajectory at index {idx} has shape {arr.shape}"
            )

        if arr1d.size == 0:
            continue
        projected.append(arr1d.astype(float, copy=False))
        kept_indices.append(idx)

    if not projected:
        raise ValueError("No non-empty trajectories available for sampling plot")
    return projected, kept_indices


def _normalise_run_labels(
    run_labels: Iterable[str] | None,
    kept_indices: Sequence[int],
) -> list[str] | None:
    if run_labels is None:
        return None
    labels = [str(label) for label in run_labels]
    filtered: list[str] = []
    for idx in kept_indices:
        if idx >= len(labels):
            raise ValueError(
                "run_labels is shorter than projection_data; "
                f"missing label for trajectory index {idx}"
            )
        filtered.append(labels[idx])
    return filtered


def _prepare_discrete_overlay(
    discrete_trajectories: Sequence[np.ndarray | Sequence[int]] | None,
    cluster_centers: np.ndarray | Sequence[Sequence[float]] | None,
    kept_indices: Sequence[int],
) -> list[np.ndarray] | None:
    if discrete_trajectories is None and cluster_centers is None:
        return None
    if discrete_trajectories is None or cluster_centers is None:
        raise ValueError(
            "Both dtraj_data and cluster_centers must be provided for the discrete overlay"
        )

    centers = np.asarray(cluster_centers, dtype=float)
    if centers.ndim != 2 or centers.shape[1] == 0:
        raise ValueError("cluster_centers must be a 2D array with at least one column")

    trajectories = list(discrete_trajectories)

    overlay: list[np.ndarray] = []
    for idx in kept_indices:
        if idx >= len(trajectories):
            raise ValueError(
                "dtraj_data must contain entries for each trajectory in projection_data"
            )
        labels = np.asarray(trajectories[idx])
        if labels.ndim != 1:
            raise ValueError(
                "Each discrete trajectory must be 1D; "
                f"trajectory at index {idx} has shape {labels.shape}"
            )
        if labels.size == 0:
            overlay.append(np.empty(0, dtype=float))
            continue
        if labels.min() < 0:
            raise ValueError(f"Discrete trajectory {idx} contains negative state indices")
        if labels.max() >= centers.shape[0]:
            raise ValueError(
                "Discrete trajectory references cluster outside cluster_centers bounds"
            )
        overlay.append(centers[labels.astype(int), 0])
    return overlay


def _collect_sampling_inputs(
    projection_data: Sequence[np.ndarray | Sequence[float]],
    *,
    run_labels: Iterable[str] | None,
    dtraj_data: Sequence[np.ndarray | Sequence[int]] | None,
    cluster_centers: np.ndarray | Sequence[Sequence[float]] | None,
    component: int,
) -> _SamplingInputs:
    if component < 0:
        raise ValueError("component_index must be non-negative")

    projected, kept_indices = _normalise_projection_data(
        projection_data, component=component
    )
    labels = _normalise_run_labels(run_labels, kept_indices)

    overlay = _prepare_discrete_overlay(dtraj_data, cluster_centers, kept_indices)

    return _SamplingInputs(projected, labels, overlay)


def create_sampling_validation_plot(
    projection_data: Sequence[np.ndarray | Sequence[float]],
    *,
    run_labels: Iterable[str] | None = None,
    dtraj_data: Sequence[np.ndarray | Sequence[int]] | None = None,
    cluster_centers: np.ndarray | Sequence[Sequence[float]] | None = None,
    component_index: int = 0,
    max_length: int = 1000,
    hist_bins: int = 150,
    stride: int = 10,
    figsize: tuple[float, float] = (10.0, 6.0),
) -> Figure:
    """Create a sampling connectivity validation plot.

    Parameters
    ----------
    projection_data
        Sequence of 1D or 2D trajectories (NumPy arrays or array-like). For 2D
        trajectories, the column specified by ``component_index`` is projected.
    run_labels
        Optional labels for each trajectory, used in the legend.
    dtraj_data
        Optional discrete trajectories aligned with ``projection_data``. Provide
        together with ``cluster_centers`` to overlay discrete states.
    cluster_centers
        Coordinates of the cluster centroids used to map discrete states back to
        the projected space. Required when providing ``dtraj_data``.
    component_index
        Zero-based index selecting which component to visualise from 2D
        trajectories. Ignored for 1D inputs.
    max_length
        Maximum number of frames from each trajectory to visualise. Must be
        positive.
    hist_bins
        Number of bins in the histogram base layer. Must be positive.
    stride
        Plot every ``stride``-th point to reduce overplotting. Must be positive.
    figsize
        Matplotlib ``(width, height)`` in inches for the generated figure.

    Returns
    -------
    Figure
        Matplotlib figure containing the sampling connectivity validation plot.

    Examples
    --------
    >>> import numpy as np
    >>> from pmarlo.visualization.diagnostics import create_sampling_validation_plot
    >>> traj_a = np.random.randn(10_000, 2)
    >>> traj_b = np.random.randn(8_000, 2)
    >>> fig = create_sampling_validation_plot([traj_a, traj_b], run_labels=["run-1", "run-2"])
    >>> fig.axes[0].get_xlabel()
    'TICA Component 1'
    """

    component_index = int(component_index)
    if component_index < 0:
        raise ValueError("component_index must be non-negative")

    _validate_positive("max_length", int(max_length))
    _validate_positive("hist_bins", int(hist_bins))
    _validate_positive("stride", int(stride))
    figsize = _ensure_figsize("figsize", figsize)

    inputs = _collect_sampling_inputs(
        projection_data,
        run_labels=run_labels,
        dtraj_data=dtraj_data,
        cluster_centers=cluster_centers,
        component=component_index,
    )

    fig, ax = plt.subplots(figsize=figsize)
    fig = reporting_plots.plot_sampling_validation(
        projected_data_1d=inputs.projected_data_1d,
        max_traj_length_plot=int(max_length),
        bins=int(hist_bins),
        stride=int(stride),
        trajectory_labels=inputs.trajectory_labels,
        discrete_data_1d=inputs.discrete_overlay,
        ax=ax,
    )
    return fig


def create_fes_validation_plot(
    fes_grid: tuple[np.ndarray | Sequence[Sequence[float]], np.ndarray | Sequence[Sequence[float]]],
    fes_data: np.ndarray | Sequence[Sequence[float]],
    *,
    max_kt: float = 7.0,
    levels: int = 25,
    cmap: str = "viridis",
    show_lines: bool = True,
    figsize: tuple[float, float] = (8.0, 6.0),
) -> Figure:
    """Create a 2D free energy surface (FES) plot on the provided grid.

    Parameters
    ----------
    fes_grid
        Tuple of coordinate grids ``(xx, yy)`` defining the mesh for the FES.
        Each entry must be a 2D array with matching shape.
    fes_data
        2D array of free energy values (in ``k_B T``) defined on ``fes_grid``.
    max_kt
        Upper colour scale cap expressed in ``k_B T`` units. Must exceed the
        minimum finite value in ``fes_data``.
    levels
        Number of contour levels to draw. Must be positive.
    cmap
        Matplotlib colormap name used for the filled contours.
    show_lines
        Whether to overlay contour lines on top of the filled surface.
    figsize
        Matplotlib ``(width, height)`` in inches for the generated figure.

    Returns
    -------
    Figure
        Matplotlib figure containing the 2D free energy surface.

    Examples
    --------
    >>> import numpy as np
    >>> from pmarlo.visualization.diagnostics import create_fes_validation_plot
    >>> x = y = np.linspace(-1.0, 1.0, 64)
    >>> xx, yy = np.meshgrid(x, y)
    >>> fes = xx**2 + yy**2
    >>> fig = create_fes_validation_plot((xx, yy), fes)
    >>> fig.axes[0].get_title()
    'Free Energy Surface'
    """

    _validate_positive("levels", int(levels))
    figsize = _ensure_figsize("figsize", figsize)

    if not isinstance(fes_grid, tuple) or len(fes_grid) != 2:
        raise ValueError("fes_grid must be a tuple of (xx, yy) coordinate arrays")

    xx = np.asarray(fes_grid[0], dtype=float)
    yy = np.asarray(fes_grid[1], dtype=float)
    fes_array = np.asarray(fes_data, dtype=float)

    if xx.ndim != 2 or yy.ndim != 2 or fes_array.ndim != 2:
        raise ValueError("fes_grid and fes_data must be 2D arrays")
    if xx.shape != yy.shape:
        raise ValueError("Coordinate arrays xx and yy must have identical shapes")
    if fes_array.shape != xx.shape:
        raise ValueError("fes_data shape must match the coordinate grid")
    if not np.isfinite(xx).all() or not np.isfinite(yy).all():
        raise ValueError("Coordinate grids must contain only finite values")
    if not np.isfinite(fes_array).any():
        raise ValueError("fes_data must contain at least one finite value")

    finite_values = fes_array[np.isfinite(fes_array)]
    finite_min = float(finite_values.min())
    max_kt = float(max_kt)
    if not np.isfinite(max_kt) or max_kt <= finite_min:
        raise ValueError(
            "max_kt must be finite and larger than the minimum finite fes_data value"
        )

    fig, ax = plt.subplots(figsize=figsize)
    fig = reporting_plots.plot_free_energy_2d(
        grid=[xx, yy],
        fes=fes_array,
        cmap=cmap,
        levels=int(levels),
        max_energy_kt=max_kt,
        add_contour_lines=bool(show_lines),
        ax=ax,
    )
    return fig

