from __future__ import annotations

"""Plot helpers for diagnostic metrics displayed in the Streamlit app."""

from typing import Any, Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from pmarlo.visualization import diagnostics as viz_diagnostics

__all__ = [
    "plot_canonical_correlations",
    "plot_autocorrelation_curves",
    "format_warnings",
    "create_sampling_validation_plot",
    "create_fes_validation_plot",
]


def _session_value(key: str, default: Any) -> Any:
    try:
        return st.session_state.get(key, default)
    except Exception:  # Streamlit runtime guard (e.g. during tests)
        return default


def plot_canonical_correlations(diagnostics: Dict[str, Any]) -> plt.Figure:
    """Visualise canonical correlations between inputs and CVs."""

    data = diagnostics.get("canonical_correlation", {}) if diagnostics else {}
    if not data:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.axis("off")
        ax.text(0.5, 0.5, "No canonical correlation data", ha="center", va="center")
        return fig

    fig, ax = plt.subplots(figsize=(6, 4))
    for split, values in sorted(data.items()):
        if not values:
            continue
        x = np.arange(1, len(values) + 1, dtype=float)
        ax.plot(x, values, marker="o", label=str(split))
    ax.set_xlabel("Component")
    ax.set_ylabel("Canonical correlation")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Canonical correlation (inputs vs CVs)")
    if len(data) > 1:
        ax.legend()
    ax.grid(True, alpha=0.3)
    return fig


def plot_autocorrelation_curves(diagnostics: Dict[str, Any]) -> plt.Figure:
    """Plot autocorrelation decay curves for each split."""

    entries = diagnostics.get("autocorrelation", {}) if diagnostics else {}
    if not entries:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.axis("off")
        ax.text(0.5, 0.5, "No autocorrelation data", ha="center", va="center")
        return fig

    fig, ax = plt.subplots(figsize=(6, 4))
    for split, payload in sorted(entries.items()):
        taus = np.asarray(payload.get("taus", []), dtype=float)
        values = np.asarray(payload.get("values", []), dtype=float)
        if taus.size == 0 or values.size == 0:
            continue
        tau_int = payload.get("tau_int")
        label = str(split)
        if tau_int is not None and np.isfinite(tau_int):
            label = f"{label} (τ_int≈{tau_int:.0f})"
        (line,) = ax.plot(taus, values, marker="o", label=label)
        color = line.get_color()
        if tau_int is not None and np.isfinite(tau_int):
            ax.axvline(
                tau_int,
                color=color,
                linestyle="--",
                alpha=0.4,
                linewidth=1.0,
            )
        window = payload.get("lag_window")
        if isinstance(window, (list, tuple)) and len(window) == 2:
            start, stop = window
            if np.isfinite(start) and np.isfinite(stop) and stop > start:
                ax.axvspan(
                    start,
                    stop,
                    color=color,
                    alpha=0.05,
                )
    ax.set_xlabel("Lag τ")
    ax.set_ylabel("ρ(τ)")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("CV autocorrelation")
    if len(entries) > 1:
        ax.legend()
    ax.grid(True, alpha=0.3)
    return fig


def format_warnings(diagnostics: Dict[str, Any]) -> list[str]:
    """Extract warning messages for display."""

    if not diagnostics:
        return []
    warnings = diagnostics.get("warnings")
    if isinstance(warnings, list):
        return [str(msg) for msg in warnings]
    return []


def create_sampling_validation_plot(
    app_state: Any | None = None,
    *,
    projection_data: Sequence[np.ndarray] | None = None,
    run_labels: Iterable[str] | None = None,
    metabiased_runs: Iterable[bool] | None = None,
    dtraj_data: Sequence[np.ndarray] | None = None,
    cluster_centers: np.ndarray | None = None,
    component_index: int = 0,
    max_length: int | None = None,
    hist_bins: int | None = None,
    stride: int | None = None,
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """Generate the sampling validation plot for the current selection."""

    if projection_data is None and app_state is not None:
        projection_data = getattr(app_state, "projection_data", None)
    if projection_data is None:
        raise ValueError("Projection data not found - cannot create sampling plot")

    if run_labels is None and app_state is not None:
        run_labels = getattr(app_state, "run_labels", None)
    if metabiased_runs is None and app_state is not None:
        metabiased_runs = getattr(app_state, "metabiased_runs", None)
    if dtraj_data is None and app_state is not None:
        dtraj_data = getattr(app_state, "dtraj_data", None)
    if cluster_centers is None and app_state is not None:
        cluster_centers = getattr(app_state, "cluster_centers", None)

    max_length = int(max_length) if max_length is not None else int(
        _session_value("val_plot_max_len", 1000)
    )
    hist_bins = int(hist_bins) if hist_bins is not None else int(
        _session_value("val_plot_hist_bins", 150)
    )
    stride = int(stride) if stride is not None else int(
        _session_value("val_plot_stride", 10)
    )
    figsize = figsize or (10.0, 6.0)

    return viz_diagnostics.create_sampling_validation_plot(
        projection_data=projection_data,
        run_labels=run_labels,
        metabiased_runs=metabiased_runs,
        dtraj_data=dtraj_data,
        cluster_centers=cluster_centers,
        component_index=component_index,
        max_length=max_length,
        hist_bins=hist_bins,
        stride=stride,
        figsize=figsize,
    )


def create_fes_validation_plot(
    app_state: Any | None = None,
    *,
    fes_grid: tuple[np.ndarray, np.ndarray] | None = None,
    fes_data: np.ndarray | None = None,
    max_kt: float | None = None,
    levels: int | None = None,
    cmap: str | None = None,
    show_lines: bool | None = None,
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """Generate the 2D FES plot using parameters from session state if available."""

    if app_state is not None:
        if fes_grid is None:
            fes_grid = getattr(app_state, "fes_grid", None)
        if fes_data is None:
            fes_data = getattr(app_state, "fes_data", None)

    if fes_grid is None or fes_data is None:
        st.warning("FES data not found. Cannot create FES plot.")
        fig, ax = plt.subplots(figsize=figsize or (8.0, 6.0))
        ax.text(0.5, 0.5, "No FES data available", ha="center", va="center")
        ax.axis("off")
        return fig

    max_kt = float(max_kt) if max_kt is not None else float(
        _session_value("fes_plot_max_kt", 7.0)
    )
    levels = int(levels) if levels is not None else int(
        _session_value("fes_plot_levels", 25)
    )
    cmap = cmap or str(_session_value("fes_plot_cmap", "viridis"))
    show_lines = bool(show_lines) if show_lines is not None else bool(
        _session_value("fes_plot_lines", True)
    )
    figsize = figsize or (8.0, 6.0)

    try:
        return viz_diagnostics.create_fes_validation_plot(
            fes_grid=fes_grid,
            fes_data=fes_data,
            max_kt=max_kt,
            levels=levels,
            cmap=cmap,
            show_lines=show_lines,
            figsize=figsize,
        )
    except Exception as exc:  # Keep UI-friendly fallback behaviour
        st.error(f"Error creating FES plot: {exc}")
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "Error generating plot", ha="center", va="center")
        ax.axis("off")
        return fig
