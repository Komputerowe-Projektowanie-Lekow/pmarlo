from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy import ndimage

logger = logging.getLogger(__name__)


def find_local_minima_2d(F: np.ndarray) -> List[Tuple[int, int]]:
    """Find simple local minima in a 2D array by 8-neighborhood comparison."""
    array = np.asarray(F)
    if array.ndim != 2:
        raise ValueError("find_local_minima_2d expects a 2D array")

    nx, ny = array.shape
    if nx < 3 or ny < 3:
        return []

    interior_mask = np.zeros_like(array, dtype=bool)
    interior_mask[1:-1, 1:-1] = True

    finite_mask = np.isfinite(array)
    if not np.any(finite_mask & interior_mask):
        return []

    min_filtered = ndimage.minimum_filter(array, size=3, mode="constant", cval=np.inf)
    max_filtered = ndimage.maximum_filter(array, size=3, mode="constant", cval=-np.inf)

    minima_mask = (
        interior_mask & finite_mask & (array == min_filtered) & (max_filtered > array)
    )

    minima_indices = np.argwhere(minima_mask)
    return [(int(i), int(j)) for i, j in minima_indices]


def pick_frames_around_minima(
    cv1: np.ndarray,
    cv2: np.ndarray,
    F: np.ndarray,
    xedges: np.ndarray,
    yedges: np.ndarray,
    deltaF_kJmol: float = 3.0,
) -> Dict[str, Any]:
    """Pick frame indices near FES minima within a free energy threshold."""
    energy_grid = np.asarray(F)
    if energy_grid.ndim < 2:
        raise ValueError("F must be at least 2D for minima detection")
    if energy_grid.ndim > 2:
        original_shape = energy_grid.shape
        collapsed = energy_grid.reshape(original_shape[0], original_shape[1], -1)
        energy_grid = np.nanmean(collapsed, axis=2)
        logger.warning(
            "pick_frames_around_minima received F array with shape %s; "
            "averaging trailing axes to produce a 2D grid (resulting shape=%s)",
            original_shape,
            energy_grid.shape,
        )

    mins = find_local_minima_2d(energy_grid)
    xcenters = 0.5 * (xedges[:-1] + xedges[1:])
    ycenters = 0.5 * (yedges[:-1] + yedges[1:])

    # Map frames to nearest bin indices
    ix = np.clip(np.digitize(cv1, xedges) - 1, 0, len(xcenters) - 1)
    iy = np.clip(np.digitize(cv2, yedges) - 1, 0, len(ycenters) - 1)

    picked: List[Dict[str, Any]] = []
    for i, j in mins:
        F0 = float(energy_grid[i, j]) if np.isfinite(energy_grid[i, j]) else np.inf
        if not np.isfinite(F0):
            continue
        mask = (ix == i) & (iy == j)
        if not np.any(mask):
            # Allow neighborhood within deltaF
            mask = np.isfinite(energy_grid[ix, iy]) & (
                energy_grid[ix, iy] <= F0 + float(deltaF_kJmol)
            )
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
