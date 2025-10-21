from __future__ import annotations

"""
Helpers for consistently extracting temperature information from nested metadata.
"""

import math
from collections import deque
from typing import Any, Iterable, Mapping

_TEMPERATURE_KEYS = ("temperature_K", "temperature")
_SEQUENCE_TYPES = (list, tuple, set)


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        result = float(value)
        if math.isnan(result):
            return None
        return result
    if isinstance(value, str):
        trimmed = value.strip()
        if not trimmed:
            return None
        suffix = trimmed[-1]
        if suffix in {"K", "k"}:
            trimmed = trimmed[:-1].strip()
        try:
            result = float(trimmed)
        except ValueError:
            return None
        if math.isnan(result):
            return None
        return result
    return None


def _iter_candidate_nodes(root: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    queue: deque[Mapping[str, Any]] = deque([root])
    seen: set[int] = set()
    while queue:
        node = queue.popleft()
        marker = id(node)
        if marker in seen:
            continue
        seen.add(marker)
        yield node
        for value in node.values():
            if isinstance(value, Mapping):
                queue.append(value)
            elif isinstance(value, _SEQUENCE_TYPES):
                for item in value:
                    if isinstance(item, Mapping):
                        queue.append(item)


def collect_temperature_values(
    meta: Mapping[str, Any] | None,
    *,
    dedupe_tol: float = 0.0,
) -> list[float]:
    """Return unique temperature values discovered within a metadata mapping.

    Parameters
    ----------
    meta:
        Arbitrary mapping-like object containing top-level or nested temperature
        information (e.g. shard metadata).
    dedupe_tol:
        Non-negative absolute tolerance used to treat candidates as duplicates.
    """
    if not isinstance(meta, Mapping):
        return []

    tol = dedupe_tol if dedupe_tol >= 0.0 else 0.0
    values: list[float] = []

    for node in _iter_candidate_nodes(meta):
        for key in _TEMPERATURE_KEYS:
            if key not in node:
                continue
            value = _coerce_float(node.get(key))
            if value is None:
                continue
            if all(abs(value - existing) > tol for existing in values):
                values.append(value)
    return values


def primary_temperature(
    meta: Mapping[str, Any] | None,
    *,
    dedupe_tol: float = 0.0,
) -> float | None:
    """Return the first temperature discovered within metadata, if any."""
    temps = collect_temperature_values(meta, dedupe_tol=dedupe_tol)
    if temps:
        return temps[0]
    return None
