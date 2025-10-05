"""Matplotlib helpers to visualize MSM and FES results."""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

__all__ = ["plot_msm", "plot_fes"]


def plot_msm(T: np.ndarray | None, pi: np.ndarray | None) -> plt.Figure:
    """Render a transition matrix heatmap with a stationary distribution bar."""

    if T is None or pi is None or T.size == 0 or pi.size == 0:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.axis("off")
        ax.text(0.5, 0.5, "No MSM available", ha="center", va="center")
        return fig

    T = np.asarray(T, dtype=float)
    pi = np.asarray(pi, dtype=float).reshape(-1)
    n = T.shape[0]
    fig = plt.figure(figsize=(8, 4))
    gs = fig.add_gridspec(1, 2, width_ratios=[3, 1], wspace=0.25)
    axT = fig.add_subplot(gs[0, 0])
    axP = fig.add_subplot(gs[0, 1])

    im = axT.imshow(T, cmap="viridis", origin="lower", aspect="auto")
    axT.set_title("Transition Matrix T")
    axT.set_xlabel("j")
    axT.set_ylabel("i")
    fig.colorbar(im, ax=axT, fraction=0.046, pad=0.04)

    x = np.arange(n)
    axP.barh(x, pi, color="tab:blue")
    axP.set_ylim(-0.5, n - 0.5)
    axP.set_title("Stationary π")
    axP.set_xlabel("probability")
    axP.set_yticks([])
    return fig


def plot_fes(fes: Any | None) -> plt.Figure:
    """Render a 2D FES contour if present."""

    if fes is None:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.axis("off")
        ax.text(0.5, 0.5, "No FES available", ha="center", va="center")
        return fig

    try:
        F = np.asarray(getattr(fes, "F"))
        xedges = np.asarray(getattr(fes, "xedges"))
        yedges = np.asarray(getattr(fes, "yedges"))
        meta = getattr(fes, "metadata", {})
    except Exception:
        F = np.asarray(fes.get("F"))  # type: ignore[assignment]
        xedges = np.asarray(fes.get("xedges"))  # type: ignore[assignment]
        yedges = np.asarray(fes.get("yedges"))  # type: ignore[assignment]
        meta = fes.get("metadata", {})  # type: ignore[assignment]

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    Xc = 0.5 * (xedges[:-1] + xedges[1:])
    Yc = 0.5 * (yedges[:-1] + yedges[1:])
    cs = ax.contourf(Xc, Yc, F.T, levels=20, cmap="magma")
    fig.colorbar(cs, ax=ax, label="F (kJ/mol)")

    names = None
    try:
        names = meta.get("names") or ()
    except Exception:
        names = ()
    if isinstance(names, (list, tuple)) and len(names) >= 2:
        ax.set_xlabel(str(names[0]))
        ax.set_ylabel(str(names[1]))
    else:
        ax.set_xlabel("cv1")
        ax.set_ylabel("cv2")

    try:
        frac = meta.get("empty_bins_fraction") if isinstance(meta, dict) else None
        if frac is not None and float(frac) > 0.30:
            ax.text(
                0.5,
                1.02,
                "Sparse sampling: >30% empty bins",
                transform=ax.transAxes,
                ha="center",
                va="bottom",
                color="crimson",
                fontsize=10,
                fontweight="bold",
            )
        if isinstance(meta, dict) and meta.get("sparse_banner"):
            ax.text(
                0.5,
                1.06,
                str(meta.get("sparse_banner")),
                transform=ax.transAxes,
                ha="center",
                va="bottom",
                color="darkorange",
                fontsize=10,
                fontweight="bold",
            )
    except Exception:
        pass

    ax.set_title("Free Energy Surface")
    return fig
