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
    """Parse a tau schedule specification into sorted unique integers.

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


def coerce_tau_schedule(raw: Any) -> tuple[int, ...]:
    """Parse tau schedule from various formats with lenient error handling.

    This is a lenient version of parse_tau_schedule that silently skips invalid
    values and returns an empty tuple if no valid values are found.

    Args:
        raw: Can be:
            - A string like "2,5,10,20" or "2; 5; 10; 20"
            - A list or tuple of integers
            - Any other type (returns empty tuple)

    Returns:
        Tuple of sorted unique positive integers representing tau values.
        Returns empty tuple if no valid values are found.

    Examples:
        >>> coerce_tau_schedule("2,5,10,20")
        (2, 5, 10, 20)
        >>> coerce_tau_schedule([2, 5, 10, 20])
        (2, 5, 10, 20)
        >>> coerce_tau_schedule("2,invalid,10")
        (2, 10)
        >>> coerce_tau_schedule([2, -5, 10])
        (2, 10)
        >>> coerce_tau_schedule("")
        ()
        >>> coerce_tau_schedule(None)
        ()
    """
    values: list[int] = []

    if isinstance(raw, (list, tuple)):
        for item in raw:
            try:
                v = int(item)
                if v > 0:
                    values.append(v)
            except (TypeError, ValueError):
                continue
    elif isinstance(raw, str):
        tokens = raw.replace(";", ",").split(",")
        for token in tokens:
            token = token.strip()
            if not token:
                continue
            try:
                v = int(token)
                if v > 0:
                    values.append(v)
            except ValueError:
                continue

    if not values:
        return ()
    return tuple(sorted(set(values)))


def parse_bins(entries: Sequence[str]) -> dict[str, int]:
    """Parse bin specifications from command-line style strings.

    Args:
        entries: Sequence of "cv=count" strings (e.g., ["Rg=72", "RMSD_ref=72"])

    Returns:
        Dictionary mapping collective variable names to bin counts.
        If no entries are provided, returns default bins: {"Rg": 72, "RMSD_ref": 72}

    Raises:
        ValueError: If any entry doesn't contain '=' or if the count isn't a valid integer.

    Examples:
        >>> parse_bins(["Rg=72", "RMSD_ref=50"])
        {'Rg': 72, 'RMSD_ref': 50}
        >>> parse_bins([])
        {'Rg': 72, 'RMSD_ref': 72}
    """
    bins: dict[str, int] = {}
    for item in entries:
        if "=" not in item:
            raise ValueError(f"Invalid bin specification '{item}', expected cv=count.")
        key, value = item.split("=", 1)
        bins[key.strip()] = int(value.strip())
    if not bins:
        bins = {"Rg": 72, "RMSD_ref": 72}
    return bins


def parse_hidden_layers(raw: Any) -> tuple[int, ...]:
    """Parse hidden layer specification from various formats.

    Args:
        raw: Can be:
            - A string like "128,128" or "64, 128, 64"
            - A list or tuple of integers
            - Any other type (returns default)

    Returns:
        Tuple of positive integers representing hidden layer sizes.
        Returns (128, 128) as default if no valid layers are found.

    Examples:
        >>> parse_hidden_layers("128,128")
        (128, 128)
        >>> parse_hidden_layers([64, 128, 64])
        (64, 128, 64)
        >>> parse_hidden_layers("")
        (128, 128)
        >>> parse_hidden_layers(None)
        (128, 128)
    """
    layers: list[int] = []

    if isinstance(raw, (list, tuple)):
        for item in raw:
            try:
                layer_size = int(item)
                if layer_size > 0:
                    layers.append(layer_size)
            except (TypeError, ValueError):
                continue
    elif isinstance(raw, str):
        for token in raw.split(","):
            token = token.strip()
            if not token:
                continue
            try:
                layer_size = int(token)
                if layer_size > 0:
                    layers.append(layer_size)
            except ValueError:
                continue

    if layers:
        return tuple(layers)
    return (128, 128)


__all__ = [
    "parse_temperature_ladder",
    "parse_tau_schedule",
    "coerce_tau_schedule",
    "parse_bins",
    "parse_hidden_layers",
]
