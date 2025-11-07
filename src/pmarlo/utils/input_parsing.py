from __future__ import annotations

"""Helpers for parsing delimited user input into structured values."""

from collections.abc import Sequence
from typing import Any, Callable

Number = int | float
CastFunc = Callable[[Any], Number]


def _normalize_tokens(raw: str) -> list[str]:
    """Split a delimited string on commas/semicolons and drop blanks."""
    tokens = [token.strip() for token in raw.replace(";", ",").split(",")]
    return [token for token in tokens if token]


def _coerce_numeric_sequence(
    values: Sequence[Any],
    *,
    cast: CastFunc,
    empty_error: str,
    invalid_error: str,
) -> list[Number]:
    coerced: list[Number] = []
    for value in values:
        try:
            coerced.append(cast(value))
        except (TypeError, ValueError) as exc:
            raise ValueError(invalid_error.format(value=value)) from exc
    if not coerced:
        raise ValueError(empty_error)
    return coerced


def parse_temperature_ladder(raw: str | Sequence[Number]) -> list[float]:
    """Parse a temperature ladder specification into floats."""
    if isinstance(raw, str):
        tokens = _normalize_tokens(raw)
        temps = _coerce_numeric_sequence(
            tokens,
            cast=float,
            empty_error="Provide at least one temperature in Kelvin.",
            invalid_error="Invalid temperature value '{value}'",
        )
    else:
        temps = _coerce_numeric_sequence(
            raw,
            cast=float,
            empty_error="Provide at least one temperature in Kelvin.",
            invalid_error="Invalid temperature value '{value}'",
        )
    return [float(temp) for temp in temps]


def parse_tau_schedule(raw: str | Sequence[Number]) -> list[int]:
    """Parse a tau schedule specification into sorted unique integers."""
    if isinstance(raw, str):
        tokens = _normalize_tokens(raw)
        tau_values = _coerce_numeric_sequence(
            tokens,
            cast=int,
            empty_error="Provide at least one tau value.",
            invalid_error="Invalid tau value '{value}'",
        )
    else:
        tau_values = _coerce_numeric_sequence(
            raw,
            cast=int,
            empty_error="Provide at least one tau value.",
            invalid_error="Invalid tau value '{value}'",
        )

    normalized: list[int] = []
    for value in tau_values:
        if value <= 0:
            raise ValueError("Tau values must be positive integers.")
        normalized.append(int(value))

    # Remove duplicates while keeping deterministic order for callers.
    return sorted(set(normalized))


__all__ = ["parse_temperature_ladder", "parse_tau_schedule"]
