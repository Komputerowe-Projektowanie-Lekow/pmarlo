"""Helpers for reproducible naming of remaps and permutations.

These functions provide small cached layers that convert array shapes and
permutation mappings into deterministic strings.  By caching the results
we ensure that repeated calls across a workflow yield identical objects,
which simplifies logging and makes debugging across passes repeatable.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Tuple


@lru_cache(maxsize=None)
def base_shape_str(shape: Tuple[int, ...]) -> str:
    """Return a canonical string representation for ``shape``.

    Parameters
    ----------
    shape:
        Tuple describing the base shape of an array or collection.

    Returns
    -------
    str
        A string formatted as ``"d0xd1x..."`` that can be used as a
        deterministic identifier in logs.
    """

    return "x".join(str(int(dim)) for dim in shape)


@lru_cache(maxsize=None)
def permutation_name(mapping: Tuple[int, ...]) -> str:
    """Return a stable name for a permutation mapping.

    Parameters
    ----------
    mapping:
        The permutation as a tuple of indices.

    Returns
    -------
    str
        Deterministic string representing the permutation.
    """

    return "-".join(str(int(idx)) for idx in mapping)
