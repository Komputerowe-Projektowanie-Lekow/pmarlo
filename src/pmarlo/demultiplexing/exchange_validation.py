"""Utilities for validating replica-exchange assignment tables."""

from __future__ import annotations

import logging
from typing import Any, Sequence, TypeVar

ErrorType = TypeVar("ErrorType", bound=Exception)


def _format_context(context: str | None) -> str:
    return f" ({context})" if context else ""


def normalize_exchange_mapping(
    mapping: Sequence[Any],
    *,
    expected_size: int | None = None,
    context: str | None = None,
    error_cls: type[ErrorType] = ValueError,
    repair_on_duplicates: bool = False,
) -> list[int]:
    """Validate and normalise a single exchange mapping.

    Parameters
    ----------
    mapping:
        Sequence describing which replica occupies each temperature slice.
    expected_size:
        Optional number of replicas expected in the mapping. If omitted the
        length of ``mapping`` is used.
    context:
        Optional label added to raised error messages to ease debugging.
    error_cls:
        Exception class used when validation fails. Defaults to ``ValueError``.

    Returns
    -------
    list[int]
        Normalised integer mapping if validation succeeds.

    Raises
    ------
    ``error_cls``
        If the mapping contains non-integer entries, has the wrong cardinality
        or is not a permutation of ``0..expected_size-1``.
    """

    try:
        values = [int(val) for val in mapping]
    except Exception as exc:  # pragma: no cover - defensive guard
        msg = f"Exchange mapping contains non-integer entries{_format_context(context)}: {list(mapping)}"
        raise error_cls(msg) from exc

    size = expected_size if expected_size is not None else len(values)
    if len(values) != size:
        msg = (
            f"Exchange mapping length {len(values)} does not match expected {size}"
            f"{_format_context(context)}"
        )
        raise error_cls(msg)

    if size <= 0:
        msg = f"Exchange mapping must contain at least one replica{_format_context(context)}"
        raise error_cls(msg)

    lower_bound, upper_bound = 0, size - 1
    out_of_range = [val for val in values if val < lower_bound or val > upper_bound]
    if out_of_range:
        msg = (
            "Replica index out of range"
            f"{_format_context(context)}: {out_of_range}; expected values in [0, {upper_bound}]"
        )
        raise error_cls(msg)

    if len(set(values)) != size:
        if not repair_on_duplicates:
            msg = (
                "Exchange mapping is not a permutation"
                f"{_format_context(context)}: {values}"
            )
            raise error_cls(msg)
        logging.getLogger("pmarlo").warning(
            "Exchange mapping requires repair%s: %s",
            _format_context(context),
            values,
        )

    return values


__all__ = ["normalize_exchange_mapping"]
