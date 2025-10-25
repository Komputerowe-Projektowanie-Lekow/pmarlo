from __future__ import annotations

"""Plot helpers for diagnostic metrics displayed in the Streamlit app."""

from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np

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


def create_sampling_validation_plot(projection_data: List[np.ndarray]) -> plt.Figure:
    """Generate the sampling validation plot from projection data.

    Parameters
    ----------
    projection_data
        List of 2D projection arrays (n_frames, n_components) for each shard.

    Returns
    -------
    plt.Figure
        Matplotlib figure ready for display.
    """
    try:
        if not projection_data:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, "No projection data available", ha="center", va="center")
            ax.axis("off")
            return fig

        projected_data_1d = [traj[:, 0] for traj in projection_data if traj.shape[1] >= 1]

        if not projected_data_1d:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, "Projection data has insufficient dimensions", ha="center", va="center")
            ax.axis("off")
            return fig

        fig, ax = pmarlo_plots.plot_sampling_validation(projected_data_1d)
        return fig

    except Exception as e:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f"Error creating sampling plot:\n{e}", ha="center", va="center")
        ax.axis("off")
        return fig


def create_fes_validation_plot(fes_result: Any) -> plt.Figure:
    """Generate the 2D FES plot from FES result data.

    Parameters
    ----------
    fes_result
        FESResult object or dict-like containing F, xedges, yedges.

    Returns
    -------
    plt.Figure
        Matplotlib figure ready for display.
    """
    try:
        if fes_result is None:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, "No FES data available", ha="center", va="center")
            ax.axis("off")
            return fig

        try:
            F = np.asarray(getattr(fes_result, "F"))
            xedges = np.asarray(getattr(fes_result, "xedges"))
            yedges = np.asarray(getattr(fes_result, "yedges"))
        except Exception:
            F = np.asarray(fes_result.get("F"))
            xedges = np.asarray(fes_result.get("xedges"))
            yedges = np.asarray(fes_result.get("yedges"))

        Xc = 0.5 * (xedges[:-1] + xedges[1:])
        Yc = 0.5 * (yedges[:-1] + yedges[1:])
        XX, YY = np.meshgrid(Xc, Yc, indexing="ij")
        grid = (XX, YY)

        fig, ax = pmarlo_plots.plot_free_energy_2d(grid, F)
        return fig

    except Exception as e:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, f"Error creating FES plot:\n{e}", ha="center", va="center")
        ax.axis("off")
        return fig
