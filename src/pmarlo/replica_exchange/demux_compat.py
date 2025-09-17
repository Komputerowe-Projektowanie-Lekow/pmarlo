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
from typing import List


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
    rows: List[ExchangeRecord] = []
    with p.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("exchange log missing header row")
        temp_cols = [
            c for c in reader.fieldnames if c.lower().startswith("replica_for_t")
        ]
        try:
            temp_cols.sort(
                key=lambda s: int("".join([c for c in s if c.isdigit()]) or -1)
            )
        except Exception:
            pass
        if not temp_cols:
            raise ValueError("exchange log must contain columns replica_for_T*")
        for line in reader:
            try:
                step = int(
                    line.get("slice") or line.get("step") or line.get("index") or 0
                )
            except Exception as exc:
                raise ValueError(
                    f"invalid slice index in exchange log: {line}"
                ) from exc
            mapping: List[int] = []
            for col in temp_cols:
                v = line.get(col)
                if v is None:
                    raise ValueError(
                        f"missing column '{col}' in exchange log row: {line}"
                    )
                try:
                    mapping.append(int(v))
                except Exception as exc:
                    raise ValueError(
                        f"non-integer mapping at {col} in row {line}"
                    ) from exc
            rows.append(ExchangeRecord(step_index=step, temp_to_replica=mapping))
    # Validate bijectivity at each slice
    for rec in rows:
        R = len(rec.temp_to_replica)
        vals = rec.temp_to_replica
        if not all(0 <= x < R for x in vals):
            raise ValueError(
                f"replica index out of range in slice={rec.step_index}: {vals}"
            )
        if len(set(vals)) != R:
            raise ValueError(
                f"non-bijective assignment at slice={rec.step_index}; expected a permutation of 0..{R-1}, got {vals}"
            )
    return rows


__all__ = ["ExchangeRecord", "parse_temperature_ladder", "parse_exchange_log"]
