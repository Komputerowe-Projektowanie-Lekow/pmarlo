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

    Matches the behavior already used in `ReplicaExchange._generate_temperature_ladder`.
    """
    if n_replicas <= 0:
        return []
    if n_replicas == 1:
        return [float(min_temp)]
    ratios = np.arange(n_replicas) / (n_replicas - 1)
    temps = (
        min_temp
        * (max_temp / max_temp if min_temp == 0 else (max_temp / min_temp)) ** ratios
    )
    # In the typical case min_temp>0; safety above avoids zero division;
    # if min_temp==0, ladder degenerates
    return [float(t) for t in temps]
