from __future__ import annotations

from typing import List

import numpy as np


def linear_temperature_ladder(
    min_temp: float, max_temp: float, n_replicas: int
) -> List[float]:
    """Generate a linearly spaced temperature ladder inclusive of bounds."""
    if n_replicas <= 0:
        return []
    if n_replicas == 1:
        return [float(min_temp)]
    temps = np.linspace(min_temp, max_temp, n_replicas)
    return [float(t) for t in temps]


def exponential_temperature_ladder(
    min_temp: float, max_temp: float, n_replicas: int
) -> List[float]:
    """Generate an exponentially spaced temperature ladder inclusive of bounds.

    This uses :func:`numpy.geomspace` which ensures a strictly monotonic ladder
    and avoids the awkward ``max_temp / max_temp`` guard that previously caused
    all-zero schedules when ``min_temp`` was zero.  REMD requires positive
    temperatures so we raise a :class:`ValueError` if either bound is non-
    positive.
    """
    if n_replicas <= 0:
        return []
    if n_replicas == 1:
        return [float(min_temp)]
    if min_temp <= 0 or max_temp <= 0:
        raise ValueError("Temperatures must be positive for exponential ladder")
    temps = np.geomspace(min_temp, max_temp, n_replicas)
    return [float(t) for t in temps]


def power_of_two_temperature_ladder(
    min_temp: float, max_temp: float, n_replicas: int | None = None
) -> List[float]:
    """Generate a temperature ladder with a power-of-two number of replicas.

    Behavior:
    - If ``n_replicas`` is None, choose a power-of-two count that yields ~5 K spacing
      between ``min_temp`` and ``max_temp`` (inclusive of bounds).
    - If ``n_replicas`` is provided, it will be rounded UP to the next power of two
      (minimum of 2) to satisfy REMD parallelism and exchange quality expectations.
    - Temperatures are linearly spaced and include both endpoints.

    Args:
        min_temp: Lower temperature bound (K).
        max_temp: Upper temperature bound (K).
        n_replicas: Desired number of replicas; coerced to a power of two if given.

    Returns:
        A sorted list of temperatures from low to high, inclusive.
    """
    # Normalize bounds
    tmin = float(min(min_temp, max_temp))
    tmax = float(max(min_temp, max_temp))

    # Degenerate case
    if np.isclose(tmin, tmax):
        return [tmin]

    def _next_power_of_two(x: int) -> int:
        if x <= 2:
            return 2
        return 1 << (int(np.ceil(np.log2(max(2, x)))))

    if n_replicas is None:
        target_step = 5.0  # aim for ~5 K spacing as per example
        delta = tmax - tmin
        approx_points = int(max(2, round(delta / target_step) + 1))
        npts = _next_power_of_two(approx_points)
    else:
        npts = _next_power_of_two(int(n_replicas))

    # Guard upper bound to a reasonable maximum (avoid extremely large ladders by mistake)
    npts = int(max(2, min(npts, 1 << 12)))  # cap at 4096 for safety

    temps = np.linspace(tmin, tmax, npts)
    return [float(t) for t in temps]
