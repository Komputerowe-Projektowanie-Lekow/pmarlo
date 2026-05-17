"""Utilities for reasoning about (t, t+tau) transition pair counts."""

from __future__ import annotations

from typing import Iterable, Sequence

__all__ = ["expected_pairs"]


def expected_pairs(
    lengths: Iterable[int] | Sequence[int],
    tau: int,
    stride: int | Iterable[int] | Sequence[int] = 1,
) -> int:
    """Return the expected number of (t, t+tau) pairs for each segment.

    Parameters
    ----------
    lengths
        Iterable with the number of frames in each segment.
    tau
        Lag time (number of frames separating source and destination).
    stride
        Step size between successive pair starts. Either a single integer applied
        to every segment or an iterable providing a stride per segment.

    Returns
    -------
    int
        Total number of (t, t+tau) pairs given the supplied configuration.
    """

    if tau < 0:
        raise ValueError("tau must be non-negative")

    length_list = [int(length) for length in lengths]
    if any(length < 0 for length in length_list):
        raise ValueError("lengths must be non-negative")
    if not length_list or not any(length_list):
        return 0

    if isinstance(stride, (str, bytes)):
        raise TypeError("stride must be an integer or iterable of integers")

    if isinstance(stride, Iterable):
        stride_values = [int(value) for value in stride]
    else:
        stride_values = [int(stride)]
    if not stride_values:
        raise ValueError("stride iterable must not be empty")
    if any(value <= 0 for value in stride_values):
        raise ValueError("stride values must be positive")

    total_pairs = 0
    last_stride = stride_values[-1]
    for idx, length in enumerate(length_list):
        if length <= 0:
            continue
        effective = length - tau
        if effective <= 0:
            continue
        if idx < len(stride_values):
            step_value = stride_values[idx]
        else:
            step_value = last_stride
        pairs = 1 + (effective - 1) // step_value
        total_pairs += pairs
    return total_pairs
