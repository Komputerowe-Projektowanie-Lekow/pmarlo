from typing import List, Sequence

def _format_tau_schedule(values: Sequence[int]) -> str:
    if not values:
        return ""
    return ", ".join(str(int(v)) for v in values)

def _parse_lag_sequence(raw: str) -> List[int]:
    """Parse lag times from a comma- or semicolon-delimited string."""

    tokens = [token.strip() for token in raw.replace(";", ",").split(",")]
    lags: List[int] = []
    for token in tokens:
        if not token:
            continue
        try:
            value = int(float(token))
        except ValueError as exc:  # pragma: no cover - user input driven
            raise ValueError(f"Invalid lag time '{token}'") from exc
        if value < 1:
            raise ValueError(
                f"Lag times must be positive integers; received {value}"
            )
        lags.append(int(value))
    if not lags:
        raise ValueError("Provide at least one lag time for implied timescale analysis.")
    return lags

def _format_lag_sequence(values: Sequence[int]) -> str:
    """Render lag time sequence as a comma-separated string."""

    if not values:
        return ""
    return ", ".join(str(int(v)) for v in values)
