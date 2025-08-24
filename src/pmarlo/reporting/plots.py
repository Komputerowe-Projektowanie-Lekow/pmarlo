from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


def save_transition_matrix_heatmap(
    T: np.ndarray, output_dir: str, name: str = "T_heatmap.png"
) -> Optional[str]:
    """Save a heatmap of a transition matrix to ``output_dir``.

    Returns the path to the written file if successful, otherwise ``None``.
    """

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("matplotlib is required for plotting") from exc

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 5))
    plt.imshow(T, cmap="viridis", origin="lower")
    plt.colorbar(label="Transition Probability")
    plt.xlabel("j")
    plt.ylabel("i")
    plt.title("Transition Matrix")
    filepath = out_dir / name
    plt.tight_layout()
    plt.savefig(filepath, dpi=200)
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
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("matplotlib is required for plotting") from exc

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    x_centers = 0.5 * (xedges[:-1] + xedges[1:])
    y_centers = 0.5 * (yedges[:-1] + yedges[1:])
    plt.figure(figsize=(7, 6))
    c = plt.contourf(x_centers, y_centers, F.T, levels=20, cmap="viridis")
    plt.colorbar(c, label="Free Energy (kJ/mol)")
    if mask is not None:
        m = np.ma.masked_where(~mask.T, mask.T)
        plt.contourf(
            x_centers,
            y_centers,
            m,
            levels=[0.5, 1.5],
            colors="none",
            hatches=["////"],
        )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"FES ({xlabel} vs {ylabel})")
    filepath = out_dir / filename
    plt.tight_layout()
    plt.savefig(filepath, dpi=200)
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
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("matplotlib is required for plotting") from exc

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    x_centers = 0.5 * (edges[:-1] + edges[1:])
    plt.figure(figsize=(7, 4))
    plt.plot(x_centers, F, color="steelblue", lw=2)
    plt.xlabel(xlabel)
    plt.ylabel("Free Energy (kJ/mol)")
    plt.title("1D PMF")
    filepath = out_dir / filename
    plt.tight_layout()
    plt.savefig(filepath, dpi=200)
    plt.close()
    return str(filepath) if filepath.exists() else None
