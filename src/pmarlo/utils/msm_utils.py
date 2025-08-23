from __future__ import annotations

from typing import List, Optional


def candidate_lag_ladder(
    min_lag: int = 1,
    max_lag: int = 200,
    n_candidates: Optional[int] = None,
) -> List[int]:
    """Generate a robust set of candidate lag times for MSM ITS analysis.

    Behavior:
    - Uses a curated set of "nice" lags (1, 2, 3, 5, 8 and 10× multiples)
      commonly used for implied-timescale scans.
    - Filters to the inclusive range [min_lag, max_lag].
    - Optionally downsamples to ``n_candidates`` approximately evenly across
      the filtered list while keeping endpoints.

    Args:
        min_lag: Minimum lag value (inclusive), coerced to >= 1.
        max_lag: Maximum lag value (inclusive), coerced to >= min_lag.
        n_candidates: If provided and > 0, downsample to this many points.

    Returns:
        An increasing list of integer lag times.
    """
    lo = int(max(1, min_lag))
    hi = int(max(lo, max_lag))

    # Curated ladder spanning typical analysis ranges
    base: List[int] = [
        1,
        2,
        3,
        5,
        8,
        10,
        15,
        20,
        30,
        50,
        75,
        100,
        150,
        200,
        300,
        500,
        750,
        1000,
        1500,
        2000,
    ]

    filtered: List[int] = [x for x in base if lo <= x <= hi]
    if not filtered:
        # Fallback: minimal sensible ladder within bounds
        if lo == hi:
            return [lo]
        return [lo, hi]

    if n_candidates is None or n_candidates <= 0 or n_candidates >= len(filtered):
        return filtered

    # Downsample approximately evenly over the filtered ladder, keep endpoints
    if n_candidates == 1:
        return [filtered[0]]
    if n_candidates == 2:
        return [filtered[0], filtered[-1]]

    step = (len(filtered) - 1) / (n_candidates - 1)
    picks = sorted({int(round(i * step)) for i in range(n_candidates)})
    # Ensure endpoints are present
    picks[0] = 0
    picks[-1] = len(filtered) - 1
    return [filtered[i] for i in picks]
