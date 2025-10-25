from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

from pmarlo.utils.json_io import normalize_for_json_row
from pmarlo.utils.path_utils import ensure_directory


def write_conformations_csv_json(
    output_dir: str,
    items: List[Dict[str, Any]],
    csv_name: str = "conformations_summary.csv",
    json_name: str = "states.json",
) -> None:
    out = Path(output_dir)
    ensure_directory(out)

    # CSV
    normalized_items = [normalize_for_json_row(it) for it in items]
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
