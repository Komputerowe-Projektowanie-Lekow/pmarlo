from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List


def _normalize_for_json_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure JSON rows carry the same fields and JSON-serializable types as CSV rows.

    - Convert numpy scalars/arrays to Python types/lists.
    - Pass through nested dicts (e.g., mfpt_to, pair, periodic).
    """
    out: Dict[str, Any] = {}
    for k, v in row.items():
        try:
            import numpy as np  # local import to avoid hard dep here

            if isinstance(v, (np.generic,)):
                out[k] = v.item()
            elif isinstance(v, np.ndarray):
                out[k] = v.tolist()
            else:
                out[k] = v
        except Exception:
            out[k] = v
    return out


def write_conformations_csv_json(
    output_dir: str,
    items: List[Dict[str, Any]],
    csv_name: str = "conformations_summary.csv",
    json_name: str = "states.json",
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # CSV
    normalized_items = [_normalize_for_json_row(it) for it in items]
    if normalized_items:
        keys = sorted({k for it in normalized_items for k in it.keys()})
        with open(out / csv_name, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for it in normalized_items:
                writer.writerow(it)

    # JSON (identical content to CSV rows)
    with open(out / json_name, "w", encoding="utf-8") as f:
        json.dump(normalized_items, f, indent=2)
