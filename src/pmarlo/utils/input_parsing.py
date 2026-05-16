from __future__ import annotations

"""Helpers for parsing delimited user input into structured values."""

from collections.abc import Sequence
from typing import Any, Callable, TypeVar

Number = int | float
NumericT = TypeVar("NumericT", int, float)
CastFunc = Callable[[Any], NumericT]


def _normalize_tokens(raw: str) -> list[str]:
    """Split a delimited string on commas/semicolons and drop blanks."""
    tokens = [token.strip() for token in raw.replace(";", ",").split(",")]
    return [token for token in tokens if token]


def _coerce_numeric_sequence(
    values: Sequence[Any],
    *,
    cast: CastFunc[NumericT],
    empty_error: str,
    invalid_error: str,
    skip_invalid: bool = False,
) -> list[NumericT]:
    coerced: list[NumericT] = []
    for value in values:
        try:
            coerced.append(cast(value))
        except (TypeError, ValueError) as exc:
            if skip_invalid:
                continue
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
    return temps


def parse_tau_schedule(raw: str | Sequence[Number], *, strict: bool = True) -> list[int]:
    """Parse a tau schedule specification into sorted unique integers.

    Args:
        raw: Delimited string or sequence of numeric tau values.
        strict: When true, invalid values raise ValueError. When false, invalid
            and non-positive values are skipped, but at least one valid tau is
            still required.

    Raises:
        ValueError: If no valid tau values are provided or if values are non-positive.
    """
    if isinstance(raw, str):
        tokens = _normalize_tokens(raw)
        tau_values = _coerce_numeric_sequence(
            tokens,
            cast=int,
            empty_error="Provide at least one tau value.",
            invalid_error="Invalid tau value '{value}'",
            skip_invalid=not strict,
        )
    else:
        tau_values = _coerce_numeric_sequence(
            raw,
            cast=int,
            empty_error="Provide at least one tau value.",
            invalid_error="Invalid tau value '{value}'",
            skip_invalid=not strict,
        )

    normalized: list[int] = []
    for value in tau_values:
        if value <= 0:
            if not strict:
                continue
            raise ValueError("Tau values must be positive integers.")
        normalized.append(int(value))

    if not normalized:
        raise ValueError("Provide at least one positive tau value.")

    # Remove duplicates while keeping deterministic order for callers.
    return sorted(set(normalized))


def parse_bins(
    entries: Sequence[str],
    *,
    default: dict[str, int] | None = None,
) -> dict[str, int]:
    """Parse bin specifications from command-line style strings.

    Args:
        entries: Sequence of "cv=count" strings (e.g., ["Rg=72", "RMSD_ref=72"])
        default: Optional values returned when no entries are provided.

    Returns:
        Dictionary mapping collective variable names to bin counts.
        If no entries are provided and no default is set, returns an empty dict.

    Raises:
        ValueError: If any entry doesn't contain '=' or if the count isn't a valid integer.

    Examples:
        >>> parse_bins(["Rg=72", "RMSD_ref=50"])
        {'Rg': 72, 'RMSD_ref': 50}
        >>> parse_bins([])
        {}
    """
    bins: dict[str, int] = {}
    for item in entries:
        if "=" not in item:
            raise ValueError(f"Invalid bin specification '{item}', expected cv=count.")
        key, value = item.split("=", 1)
        bins[key.strip()] = int(value.strip())
    if not bins:
        return dict(default or {})
    return bins


def parse_hidden_layers(raw: str | Sequence[Number], *, strict: bool = True) -> tuple[int, ...]:
    """Parse hidden layer specification from various formats.

    Args:
        raw: Can be:
            - A string like "128,128" or "64, 128, 64"
            - A list or tuple of integers
        strict: When true, invalid values raise ValueError. When false, invalid
            and non-positive values are skipped, but at least one valid layer is
            still required.

    Returns:
        Tuple of positive integers representing hidden layer sizes.

    Examples:
        >>> parse_hidden_layers("128,128")
        (128, 128)
        >>> parse_hidden_layers([64, 128, 64])
        (64, 128, 64)
    """
    if isinstance(raw, str):
        values: Sequence[Any] = _normalize_tokens(raw)
    else:
        if not isinstance(raw, Sequence):
            raise ValueError("Hidden layers must be provided as a string or sequence.")
        values = raw

    layer_values = _coerce_numeric_sequence(
        values,
        cast=int,
        empty_error="Provide at least one hidden layer size.",
        invalid_error="Invalid hidden layer size '{value}'",
        skip_invalid=not strict,
    )

    layers: list[int] = []
    for value in layer_values:
        if value <= 0:
            if not strict:
                continue
            raise ValueError("Hidden layer sizes must be positive integers.")
        layers.append(int(value))

    if not layers:
        raise ValueError("Provide at least one positive hidden layer size.")
    return tuple(layers)


__all__ = [
    "parse_temperature_ladder",
    "parse_tau_schedule",
    "parse_bins",
    "parse_hidden_layers",
]
