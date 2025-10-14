"""Utilities for reasoning about (t, t+tau) transition pair counts."""

from __future__ import annotations

from itertools import zip_longest
from typing import Iterable, Sequence

__all__ = ["expected_pairs"]


def expected_pairs(
    lengths: Iterable[int] | Sequence[int],
    tau: int,
    stride: int | Iterable[int] | Sequence[int] = 1,
) -> int:
    """Return the expected number of (t, t+tau) pairs for each shard.

    Parameters
    ----------
    lengths
        Iterable with the number of frames in each shard/segment.
    tau
        Lag time (number of frames separating source and destination).
    stride
        Step size between successive pair starts. Either a single integer applied
        to every shard or an iterable providing a stride per shard.

    Returns
    -------
    int
        Total number of (t, t+tau) pairs given the supplied configuration.
    """

    if tau < 0:
        raise ValueError("tau must be non-negative")

    normalised_lengths = [max(0, int(length)) for length in lengths if int(length) > 0]
    if not normalised_lengths:
        return 0

    if isinstance(stride, (str, bytes)):
        raise TypeError("stride must be an integer or iterable of integers")

    if isinstance(stride, Iterable):
        stride_iter = [
            max(1, int(value)) if value is not None else 1 for value in stride
        ]
    else:
        stride_iter = [max(1, int(stride))]

    total_pairs = 0
    for length, step in zip_longest(
        normalised_lengths, stride_iter, fillvalue=stride_iter[-1]
    ):
        if length is None:
            continue
        effective = length - tau
        if effective <= 0:
            continue
        step_value = 1 if step is None else max(1, int(step))
        pairs = 1 + (effective - 1) // step_value
        total_pairs += pairs
    return total_pairs
