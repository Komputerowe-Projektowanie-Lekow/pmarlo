from __future__ import annotations

"""Shared helpers for reading JSON payloads from disk."""

import json
from pathlib import Path
from typing import Any

__all__ = ["load_json_file"]


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
