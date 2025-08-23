"""High-level helper APIs for PMARLO."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .msm.fes import FESResult


def generate_fes_and_pick_minima(
    fes: FESResult,
) -> Tuple[Tuple[float, float], FESResult]:
    """Return the minima coordinates from a free energy surface.

    Parameters
    ----------
    fes:
        Precomputed free energy surface.

    Returns
    -------
    (float, float), FESResult
        Coordinates of the global minimum and the original FESResult.
    """

    F = fes.F
    min_index = np.unravel_index(np.nanargmin(F), F.shape)
    x_min = float(fes.xedges[min_index[0]])
    y_min = float(fes.yedges[min_index[1]])
    return (x_min, y_min), fes
