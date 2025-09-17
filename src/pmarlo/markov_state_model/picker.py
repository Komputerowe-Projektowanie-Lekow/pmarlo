from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np


def find_local_minima_2d(F: np.ndarray) -> List[Tuple[int, int]]:
    """Find simple local minima in a 2D array by 8-neighborhood comparison."""
    minima: List[Tuple[int, int]] = []
    nx, ny = F.shape
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            val = F[i, j]
            if not np.isfinite(val):
                continue
            neighbors = F[i - 1 : i + 2, j - 1 : j + 2]
            if np.all(val <= neighbors) and np.any(val < neighbors):
                minima.append((i, j))
    return minima


def pick_frames_around_minima(
    cv1: np.ndarray,
    cv2: np.ndarray,
    F: np.ndarray,
    xedges: np.ndarray,
    yedges: np.ndarray,
    deltaF_kJmol: float = 3.0,
) -> Dict[str, Any]:
    """Pick frame indices near FES minima within a free energy threshold."""
    mins = find_local_minima_2d(F)
    xcenters = 0.5 * (xedges[:-1] + xedges[1:])
    ycenters = 0.5 * (yedges[:-1] + yedges[1:])

    # Map frames to nearest bin indices
    ix = np.clip(np.digitize(cv1, xedges) - 1, 0, len(xcenters) - 1)
    iy = np.clip(np.digitize(cv2, yedges) - 1, 0, len(ycenters) - 1)

    picked: List[Dict[str, Any]] = []
    for i, j in mins:
        F0 = float(F[i, j]) if np.isfinite(F[i, j]) else np.inf
        if not np.isfinite(F0):
            continue
        mask = (ix == i) & (iy == j)
        if not np.any(mask):
            # Allow neighborhood within deltaF
            mask = np.isfinite(F[ix, iy]) & (F[ix, iy] <= F0 + float(deltaF_kJmol))
        frames = np.where(mask)[0].tolist()
        picked.append(
            {
                "minimum_bin": (int(i), int(j)),
                "F0": F0,
                "num_frames": int(len(frames)),
                "frames": frames[:1000],  # cap for safety
            }
        )
    return {"minima": picked}
