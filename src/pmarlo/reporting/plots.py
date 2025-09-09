from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

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
    # If F is extremely sparse (lots of NaN/inf), fallback to hexbin density plot
    finite_mask = np.isfinite(F)
    empty_frac = 1.0 - float(np.count_nonzero(finite_mask)) / float(
        F.size if F.size else 1
    )
    if empty_frac > 0.60:
        # Build centers mesh and map F to densities by exp(-F/kT) relative scale if possible
        # For fallback, just plot a hexbin over grid centers using finite F values
        try:
            import matplotlib.pyplot as plt

            Xc, Yc = np.meshgrid(x_centers, y_centers, indexing="ij")
            xv = Xc[finite_mask]
            yv = Yc[finite_mask]
            fv = (
                F.T[finite_mask.T]
                if F.shape == (len(x_centers), len(y_centers))
                else F.T[finite_mask.T]
            )
            hb = plt.hexbin(xv, yv, C=fv, gridsize=40, cmap="viridis")
            plt.colorbar(hb, label="Free Energy (kJ/mol)")
            title_warn = f" (Sparse FES: {empty_frac*100.0:.1f}% empty)"
        except Exception:
            c = plt.contourf(x_centers, y_centers, F.T, levels=20, cmap="viridis")
            plt.colorbar(c, label="Free Energy (kJ/mol)")
            title_warn = f" (Sparse FES: {empty_frac*100.0:.1f}% empty)"
    else:
        c = plt.contourf(x_centers, y_centers, F.T, levels=20, cmap="viridis")
        plt.colorbar(c, label="Free Energy (kJ/mol)")
        title_warn = ""
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
    plt.title(f"FES ({xlabel} vs {ylabel}){title_warn}")
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


def _kT_kJ_per_mol(temperature_kelvin: float) -> float:
    try:
        from scipy import constants

        return float(constants.k * temperature_kelvin * constants.Avogadro / 1000.0)
    except Exception:
        # Fallback constant if SciPy is unavailable
        k_B = 1.380649e-23  # J/K
        N_A = 6.02214076e23  # 1/mol
        return float(k_B * temperature_kelvin * N_A / 1000.0)


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
        x_lo, x_hi = np.quantile(x, [0.01, 0.99])
        y_lo, y_hi = np.quantile(y, [0.01, 0.99])
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
                xe = np.linspace(float(np.min(x)), float(np.max(x)) + 1e-8, 41)
                ye = np.linspace(float(np.min(y)), float(np.max(y)) + 1e-8, 41)
                H = np.zeros((len(xe) - 1, len(ye) - 1), dtype=float)
                return np.full_like(H, np.nan), xe, ye, "Invalid FES ranges"
        nb = max(40, int(np.sqrt(len(x)) / 4))
        H, xe, ye = np.histogram2d(x, y, bins=nb, range=[[x_lo, x_hi], [y_lo, y_hi]])
    else:
        nb = int(bins)
        if nb <= 0:
            nb = 100
        H, xe, ye = np.histogram2d(x, y, bins=nb)

    empty = (H < max(1, int(min_count))).mean() * 100.0
    if empty > 30.0:
        warn = (
            f"Sparse FES: {empty:.1f}% empty bins (try adaptive bins or more sampling)"
        )
    else:
        warn = None

    kT = _kT_kJ_per_mol(float(temperature))
    F = -kT * np.log(H + 1e-12)
    # Assign +inf to truly empty (below min_count) bins to avoid misleading minima
    F = np.where(H >= max(1, int(min_count)), F, np.inf)
    if np.any(np.isfinite(F)):
        F -= np.nanmin(F)
    return F.astype(float, copy=False), xe, ye, warn
