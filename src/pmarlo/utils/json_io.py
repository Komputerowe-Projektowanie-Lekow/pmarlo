from __future__ import annotations

"""Shared helpers for reading JSON payloads from disk."""

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np

__all__ = ["load_json_file", "normalize_for_json_row"]


def load_json_file(path: Path | str, *, encoding: str = "utf-8") -> Any:
    """Read and decode JSON from ``path`` with a helpful error message."""

    json_path = Path(path)
    text = json_path.read_text(encoding=encoding)
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise json.JSONDecodeError(
            f"{exc.msg} (file: {json_path})", exc.doc, exc.pos
        ) from exc


def normalize_for_json_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure JSON rows carry the same fields and JSON-serializable types as CSV rows.

    - Convert numpy scalars/arrays to Python types/lists.
    - Pass through nested dicts (e.g., mfpt_to, pair, periodic).
    """
    out: Dict[str, Any] = {}
    for k, v in row.items():
        if isinstance(v, np.generic):
            out[k] = v.item()
        elif isinstance(v, np.ndarray):
            out[k] = v.tolist()
        else:
            out[k] = v
    return out
