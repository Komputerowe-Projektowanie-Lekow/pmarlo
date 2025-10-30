from __future__ import annotations

import logging
from typing import List

import numpy as np

logger = logging.getLogger(__name__)


def linear_temperature_ladder(
    min_temp: float, max_temp: float, n_replicas: int
) -> list[float]:
    """Generate a linearly spaced temperature ladder inclusive of bounds."""
    if n_replicas < 1:
        raise ValueError("n_replicas must be positive")
    if n_replicas == 1:
        return [float(min_temp)]
    tmin, tmax = sorted((float(min_temp), float(max_temp)))
    logger.debug(
        "Generating linear ladder from %s to %s with %d replicas",
        tmin,
        tmax,
        n_replicas,
    )
    temps = np.linspace(tmin, tmax, n_replicas)
    return [float(t) for t in temps]


def exponential_temperature_ladder(
    min_temp: float, max_temp: float, n_replicas: int
) -> List[float]:
    """Generate an exponentially spaced temperature ladder inclusive of bounds.

    Uses :func:`numpy.geomspace` to ensure strictly monotonic spacing.
    REMD requires positive temperatures, so both bounds must be > 0.
    """
    if n_replicas <= 0:
        raise ValueError("n_replicas must be positive")
    if n_replicas == 1:
        return [float(min_temp)]
    if min_temp <= 0 or max_temp <= 0:
        raise ValueError("Temperatures must be positive for exponential ladder")

    tmin, tmax = sorted((float(min_temp), float(max_temp)))

    logger.debug(
        "Generating exponential ladder from %s to %s with %d replicas",
        tmin,
        tmax,
        n_replicas,
    )

    temps = np.geomspace(tmin, tmax, n_replicas)
    return [float(t) for t in temps]


def geometric_temperature_ladder(
    min_temp: float, max_temp: float, n_replicas: int
) -> list[float]:
    """Generate a geometrically spaced temperature ladder.

    This is an alias for :func:`exponential_temperature_ladder` for clarity in
    UI contexts. Returns a sorted, strictly positive list including bounds.
    """
    return list(exponential_temperature_ladder(min_temp, max_temp, n_replicas))


def geometric_ladder(
    tmin: float, tmax: float, n: int, endpoint: bool = True
) -> np.ndarray:
    """Return a stable geometric ladder from ``tmin`` to ``tmax``.

    Parameters
    ----------
    tmin : float
        Minimum temperature (K), strictly positive.
    tmax : float
        Maximum temperature (K), strictly greater than ``tmin``.
    n : int
        Number of points (>= 2).
    endpoint : bool, default True
        If True, include ``tmax``; if False, last point is ``tmax/r``.

    Returns
    -------
    np.ndarray
        Geometric sequence of temperatures in ascending order.
    """
    import numpy as _np

    tmin = float(tmin)
    tmax = float(tmax)
    n = int(n)
    if n < 2:
        raise ValueError("n must be >= 2")
    if not (tmin > 0.0 and tmax > 0.0 and tmax > tmin):
        raise ValueError("Require 0 < tmin < tmax")
    vals = _np.geomspace(tmin, tmax, num=n, endpoint=endpoint)
    return _np.asarray(vals, dtype=float)


def power_of_two_temperature_ladder(
    min_temp: float, max_temp: float, n_replicas: int | None = None
) -> list[float]:
    """Generate a temperature ladder with a power-of-two number of replicas.

    Behavior:
    - ``n_replicas`` must be provided explicitly; automatic inference is not
      supported because it can hide configuration mistakes.
    - The provided ``n_replicas`` value is rounded UP to the next power of two
      (minimum of 2) to satisfy REMD parallelism and exchange quality
      expectations.
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

    # Automatic inference is disallowed to avoid silent misconfiguration.
    if n_replicas is None:
        raise ValueError(
            "n_replicas must be provided; automatic power-of-two selection was removed"
        )

    # Degenerate case
    if np.isclose(tmin, tmax):
        n_rep = int(n_replicas)
        if n_rep != 1:
            raise ValueError(
                "Identical temperature bounds require n_replicas=1 for a valid ladder"
            )
        return [tmin]

    def _next_power_of_two(x: int) -> int:
        if x <= 2:
            return 2
        return 1 << (int(np.ceil(np.log2(max(2, x)))))

    n_rep = int(n_replicas)
    if n_rep < 1:
        raise ValueError("n_replicas must be positive")
    npts = _next_power_of_two(n_rep)
    logger.debug("Rounded n_replicas=%s to power-of-two=%s", n_rep, npts)

    # Guard upper bound to a reasonable maximum (avoid extremely large ladders by mistake)
    npts = int(max(2, min(npts, 1 << 12)))  # cap at 4096 for safety

    temps = np.linspace(tmin, tmax, npts)
    return [float(t) for t in temps]
