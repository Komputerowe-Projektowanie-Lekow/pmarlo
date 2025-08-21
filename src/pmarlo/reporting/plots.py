from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def save_transition_matrix_heatmap(
    T: np.ndarray, output_dir: str, name: str = "T_heatmap.png"
) -> Optional[str]:
    try:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(6, 5))
        plt.imshow(T, cmap="viridis", origin="lower")
        plt.colorbar(label="Transition Prob.")
        plt.xlabel("j")
        plt.ylabel("i")
        plt.title("Transition Matrix")
        filepath = out_dir / name
        plt.tight_layout()
        plt.savefig(filepath, dpi=200)
        plt.close()
        return str(filepath)
    except Exception:
        return None


def save_fes_contour(
    F: np.ndarray,
    xedges: np.ndarray,
    yedges: np.ndarray,
    xlabel: str,
    ylabel: str,
    output_dir: str,
    filename: str,
) -> Optional[str]:
    try:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        x_centers = 0.5 * (xedges[:-1] + xedges[1:])
        y_centers = 0.5 * (yedges[:-1] + yedges[1:])
        plt.figure(figsize=(7, 6))
        c = plt.contourf(x_centers, y_centers, F.T, levels=20, cmap="viridis")
        plt.colorbar(c, label="Free Energy (kJ/mol)")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(f"FES ({xlabel} vs {ylabel})")
        filepath = out_dir / filename
        plt.tight_layout()
        plt.savefig(filepath, dpi=200)
        plt.close()
        return str(filepath)
    except Exception:
        return None
