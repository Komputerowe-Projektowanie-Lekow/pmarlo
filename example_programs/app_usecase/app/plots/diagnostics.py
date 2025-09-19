from __future__ import annotations

"""Plot helpers for diagnostic metrics displayed in the Streamlit app."""

from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np

__all__ = [
    "plot_canonical_correlations",
    "plot_autocorrelation_curves",
    "format_warnings",
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
