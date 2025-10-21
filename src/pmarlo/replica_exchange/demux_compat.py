from __future__ import annotations

"""
Compatibility helpers for demultiplexing.

These functions were historically defined under `pmarlo.remd.demux` and are
now hosted here as lightweight utilities. Prefer the streaming demux in
`pmarlo.demultiplexing` for production use.
"""

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Mapping, Sequence

from pmarlo.demultiplexing.exchange_validation import normalize_exchange_mapping


@dataclass(frozen=True)
class ExchangeRecord:
    step_index: int
    temp_to_replica: List[int]


def parse_temperature_ladder(source: List[float] | str) -> List[float]:
    """Parse a temperature ladder from a list, CSV string or JSON file path."""
    if isinstance(source, str):
        p = Path(source)
        if p.exists():
            data = json.loads(p.read_text())
            if not isinstance(data, list) or not all(
                isinstance(x, (int, float)) for x in data
            ):
                raise ValueError("ladder file must contain a JSON list of numbers")
            return [float(x) for x in data]
        # Support comma-separated string e.g. "300,310,320"
        try:
            return [float(x.strip()) for x in source.split(",") if x.strip()]
        except Exception as exc:
            raise ValueError(f"invalid ladder spec: {source}") from exc
    return [float(x) for x in source]


def parse_exchange_log(path: str | Path) -> List[ExchangeRecord]:
    """Parse a canonical CSV exchange log into `ExchangeRecord` rows.

    Expected header:
        slice,replica_for_T0,replica_for_T1,...,replica_for_TR-1
    Values must be integers in [0, R-1] and each row must be a permutation.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"exchange log not found: {p}")
    with p.open("r", newline="") as f:
        reader = csv.DictReader(f)
        temp_cols = _extract_temperature_columns(reader.fieldnames)
        rows = [
            _parse_exchange_row(line, temp_cols)
            for line in reader
            if any(line.values())
        ]
    _validate_exchange_rows(rows)
    return rows


def _extract_temperature_columns(fieldnames: Sequence[str] | None) -> List[str]:
    if not fieldnames:
        raise ValueError("exchange log missing header row")
    temp_cols = [c for c in fieldnames if c.lower().startswith("replica_for_t")]
    if not temp_cols:
        raise ValueError("exchange log must contain columns replica_for_T*")
    try:
        temp_cols.sort(key=_temperature_column_key)
    except Exception:
        pass
    return temp_cols


def _temperature_column_key(label: str) -> int:
    digits = "".join([c for c in label if c.isdigit()])
    return int(digits) if digits else -1


def _parse_exchange_row(
    line: Mapping[str, str | None], temp_cols: Iterable[str]
) -> ExchangeRecord:
    step = _parse_step_index(line)
    mapping = [_parse_mapping_value(line, col) for col in temp_cols]
    return ExchangeRecord(step_index=step, temp_to_replica=mapping)


def _parse_step_index(line: Mapping[str, str | None]) -> int:
    try:
        raw = line.get("slice") or line.get("step") or line.get("index") or 0
        return int(raw)
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError(f"invalid slice index in exchange log: {line}") from exc


def _parse_mapping_value(line: Mapping[str, str | None], column: str) -> int:
    value = line.get(column)
    if value is None:
        raise ValueError(f"missing column '{column}' in exchange log row: {line}")
    try:
        return int(value)
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError(f"non-integer mapping at {column} in row {line}") from exc


def _validate_exchange_rows(rows: Iterable[ExchangeRecord]) -> None:
    expected_size: int | None = None
    for rec in rows:
        if expected_size is None:
            expected_size = len(rec.temp_to_replica)
        normalize_exchange_mapping(
            rec.temp_to_replica,
            expected_size=expected_size,
            context=f"slice={rec.step_index}",
        )


__all__ = ["ExchangeRecord", "parse_temperature_ladder", "parse_exchange_log"]
