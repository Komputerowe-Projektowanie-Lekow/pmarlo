from __future__ import annotations

"""Shared helpers for reading JSON payloads from disk."""

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict

import numpy as np

__all__ = [
    "load_json_file",
    "normalize_for_json_row",
    "sanitize",
    "write_json",
    "sanitize_deeptica_payload",
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


def _preprocess_value(value: Any) -> Any:
    """Convert domain-specific objects to JSON-friendly primitives before validation."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        scalar = value.item()
        if isinstance(scalar, float):
            return scalar if math.isfinite(scalar) else None
        return scalar
    if isinstance(value, Mapping):
        return value
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return str(value)


def _normalize_sequence(value: Any) -> Any:
    if isinstance(value, list):
        return value
    if isinstance(value, (tuple, set)):
        return list(value)
    return value


def _normalize_mapping(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): v for k, v in value.items()}
    return value


def sanitize(obj: Any) -> Any:
    """Normalize arbitrary objects into JSON-serializable structures recursively."""
    value = _preprocess_value(obj)
    if isinstance(value, list):
        normalized = _normalize_sequence(value)
        return [sanitize(item) for item in normalized]
    if isinstance(value, Mapping):
        normalized = _normalize_mapping(value)
        return {str(k): sanitize(v) for k, v in normalized.items()}
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    """Write a JSON file with automatic sanitization and directory creation.

    Args:
        path: The file path where the JSON will be written
        payload: The data to serialize to JSON

    Returns:
        The path to the written file
    """
    from pmarlo.utils.path_utils import ensure_directory

    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(sanitize(payload), handle, indent=2, sort_keys=True)
    return path


def sanitize_deeptica_payload(raw: Mapping[str, Any]) -> Dict[str, Any]:
    """Extract and sanitize specific fields from DeepTICA training results.

    This function creates a summary of DeepTICA training results by extracting
    relevant fields and trimming verbose data structures to keep only the most
    important information.

    Args:
        raw: The raw DeepTICA training result mapping

    Returns:
        A sanitized dictionary containing only the most relevant fields
    """
    summary: Dict[str, Any] = {}
    fields = [
        "applied",
        "skipped",
        "reason",
        "method",
        "lag",
        "lag_used",
        "n_out",
        "pairs_total",
        "warnings",
        "lag_candidates",
    ]
    for field in fields:
        if field in raw:
            summary[field] = raw[field]

    attempts = raw.get("attempts")
    if isinstance(attempts, list):
        trimmed: list[Dict[str, Any]] = []
        for attempt in attempts[:5]:
            if not isinstance(attempt, Mapping):
                continue
            trimmed.append(
                {
                    "lag": attempt.get("lag"),
                    "pairs_total": attempt.get("pairs_total"),
                    "status": attempt.get("status"),
                    "warnings": attempt.get("warnings"),
                }
            )
        if trimmed:
            summary["attempts"] = trimmed

    per_shard = raw.get("per_shard")
    if isinstance(per_shard, list):
        trimmed_shards: list[Dict[str, Any]] = []
        for shard_info in per_shard:
            if not isinstance(shard_info, Mapping):
                continue
            trimmed_shards.append(
                {
                    "shard_id": shard_info.get("shard_id") or shard_info.get("id"),
                    "pairs": shard_info.get("pairs"),
                    "frames": shard_info.get("frames"),
                }
            )
        if trimmed_shards:
            summary["per_shard"] = trimmed_shards

    training_metrics = raw.get("training_metrics")
    if isinstance(training_metrics, Mapping):
        summary["training_metrics"] = {
            "wall_time_s": training_metrics.get("wall_time_s"),
            "final_objective": training_metrics.get("final_objective"),
            "output_variance": training_metrics.get("output_variance"),
        }
    return summary
