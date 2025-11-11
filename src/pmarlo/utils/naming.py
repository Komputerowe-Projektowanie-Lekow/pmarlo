"""Helpers for reproducible naming of remaps and permutations.

These functions provide small cached layers that convert array shapes and
permutation mappings into deterministic strings. By caching the results we
ensure that repeated calls across a workflow yield identical objects, which
simplifies logging and makes debugging across passes repeatable.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import datetime
from functools import lru_cache
from typing import Optional, Tuple

_UNSAFE_SLUG_CHARS = re.compile(r"[^a-z0-9_-]")


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


def timestamp() -> str:
    """Generate standardized timestamp string for file naming and logging.

    Returns
    -------
    str
        Timestamp string in format "YYYYMMDD-HHMMSS" (e.g., "20251107-143022").

    Examples
    --------
    >>> ts = timestamp()
    >>> len(ts)
    15
    >>> ts[8]
    '-'

    Notes
    -----
    This function is commonly used for generating unique file names and
    timestamping log entries in a consistent format across PMARLO workflows.
    """
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def slugify(label: Optional[str]) -> Optional[str]:
    """Return a deterministic slug for ``label`` suitable for filenames.

    This function sanitizes strings for safe use in file/directory names by
    replacing special characters with underscores and converting to lowercase.

    Parameters
    ----------
    label : Optional[str]
        The input string to slugify. If None or empty, returns None.

    Returns
    -------
    Optional[str]
        A sanitized slug suitable for filenames, or None if input is empty.
        The slug contains only lowercase alphanumeric characters, underscores,
        and hyphens.

    Examples
    --------
    >>> slugify("My Project Name!")
    'my_project_name'
    >>> slugify("Test-123_ABC")
    'test-123_abc'
    >>> slugify("")
    >>> slugify(None)

    Notes
    -----
    The function performs the following transformations:
    - Unicode normalization (NFKD)
    - ASCII encoding (non-ASCII characters are removed)
    - Lowercase conversion
    - Special characters replaced with underscores
    - Leading/trailing underscores stripped
    """
    if not label:
        return None
    normalised = unicodedata.normalize("NFKD", str(label)).strip()
    ascii_label = normalised.encode("ascii", "ignore").decode("ascii")
    slug = _UNSAFE_SLUG_CHARS.sub("_", ascii_label.lower())
    slug = slug.strip("_")
    return slug or None
