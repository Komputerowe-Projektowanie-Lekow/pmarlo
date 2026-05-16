from __future__ import annotations

"""Shared helpers for reading JSON payloads from disk."""

import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping, cast

import numpy as np

from pmarlo.utils.path_utils import ensure_directory

__all__ = [
    "load_json_file",
    "normalize_for_json_row",
    "sanitize",
    "write_json",
]


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

    - Convert numpy scalars/arrays to Python types/lists recursively.
    - Convert non-finite float values to null-compatible ``None``.
    """
    return cast(Dict[str, Any], sanitize(row))


def sanitize(obj: Any) -> Any:
    """Recursively convert arbitrary objects into JSON-serializable structures.

    This function handles various Python and NumPy types and converts them to
    JSON-compatible types:
    - None, bool, int, float, str are passed through (with NaN/Inf -> None)
    - Path objects are converted to strings
    - NumPy arrays are converted to lists
    - NumPy scalars are converted to Python scalars
    - Mappings are converted to dicts with string keys
    - Lists, tuples, sets are converted to lists
    - All other types are converted to strings
    """
    if obj is None or isinstance(obj, bool):
        return obj
    if isinstance(obj, (int, float, str)):
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return sanitize(obj.tolist())
    if isinstance(obj, np.generic):
        return sanitize(obj.item())
    if isinstance(obj, Mapping):
        return {str(k): sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [sanitize(v) for v in obj]
    return str(obj)


def write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    """Write a JSON file with automatic sanitization and directory creation.

    Args:
        path: The file path where the JSON will be written
        payload: The data to serialize to JSON

    Returns:
        The path to the written file
    """
    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(sanitize(payload), handle, indent=2, sort_keys=True)
    return path
