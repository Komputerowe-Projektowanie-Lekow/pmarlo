from __future__ import annotations

"""Plot helpers for diagnostic metrics displayed in the Streamlit app."""

from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from pmarlo.reporting import plots as pmarlo_plots

__all__ = [
    "plot_canonical_correlations",
    "plot_autocorrelation_curves",
    "format_warnings",
    "create_sampling_validation_plot",
    "create_fes_validation_plot",
]


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
        ax.plot(taus, values, marker="o", label=str(split))
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


def create_sampling_validation_plot(app_state) -> plt.Figure:
    """Generate the sampling validation plot using parameters from session state if available.

    Parameters
    ----------
    app_state
        Application state object containing projection_data.

    Returns
    -------
    plt.Figure
        Matplotlib figure ready for display.
    """
    projection = app_state.projection_data

    # Debug: Check projection data
    print(f"[DEBUG] projection is None: {projection is None}")
    if projection is not None:
        print(f"[DEBUG] projection type: {type(projection)}")
        print(f"[DEBUG] projection length: {len(projection)}")

    if projection is None:
        raise ValueError("Projection data not found - cannot create sampling plot")

    # Debug: Check each trajectory shape before filtering
    print(f"[DEBUG] Number of trajectories in projection: {len(projection)}")
    for i, traj in enumerate(projection):
        print(f"[DEBUG] Trajectory {i}: shape={traj.shape}, dtype={traj.dtype}")

    projected_data_1d = [traj[:, 0] for traj in projection if traj.shape[1] > 0]

    # Debug: Check filtered results
    print(f"[DEBUG] Length of projected_data_1d after filtering: {len(projected_data_1d)}")
    for i, arr in enumerate(projected_data_1d):
        print(f"[DEBUG] projected_data_1d[{i}]: shape={arr.shape}, length={len(arr)}, dtype={arr.dtype}")

    if not projected_data_1d:
        raise ValueError("No valid 1D projection data found - projection data has insufficient dimensions")

    # Get parameters from session state (set by UI widgets in app.py)
    # Provide sensible defaults if not found in state
    max_len = st.session_state.get("val_plot_max_len", 1000)
    bins = st.session_state.get("val_plot_hist_bins", 150)
    stride = st.session_state.get("val_plot_stride", 10)

    print(f"[DEBUG] Parameters: max_len={max_len}, bins={bins}, stride={stride}")

    # Call the updated library function with parameters
    fig = pmarlo_plots.plot_sampling_validation(
        projected_data_1d=projected_data_1d,
        max_traj_length_plot=max_len,
        bins=bins,
        stride=stride,
        alpha_hist=0.15,  # Keep alpha/lw/cmap fixed or add widgets too
        alpha_traj=0.4,
        lw_traj=0.3,
        cmap_name='viridis'
    )
    return fig


def create_fes_validation_plot(app_state) -> plt.Figure:
    """Generate the 2D FES plot using parameters from session state if available.

    Parameters
    ----------
    app_state
        Application state object containing fes_grid and fes_data.

    Returns
    -------
    plt.Figure
        Matplotlib figure ready for display.
    """
    try:
        grid = app_state.fes_grid
        fes = app_state.fes_data

        if grid is None or fes is None:
            st.warning("FES data not found. Cannot create FES plot.")
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, "No FES data available", ha="center", va="center")
            ax.axis("off")
            return fig

        # Get parameters from session state
        max_energy_kt = st.session_state.get("fes_plot_max_kt", 7.0)
        levels = st.session_state.get("fes_plot_levels", 25)
        cmap = st.session_state.get("fes_plot_cmap", "viridis")
        add_lines = st.session_state.get("fes_plot_lines", True)

        # Call the updated library function with parameters
        fig = pmarlo_plots.plot_free_energy_2d(
            grid=grid,
            fes=fes,
            max_energy_kt=max_energy_kt,
            levels=levels,
            cmap=cmap,
            add_contour_lines=add_lines
        )
        return fig

    except Exception as e:
        st.error(f"Error creating FES plot: {e}")
        # Ensure a figure object is always returned
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "Error generating plot", ha='center', va='center')
        ax.axis("off")
        return fig
