from __future__ import annotations

"""Small helpers to persist app state/manifest under a workspace directory.

Schema (JSON):
- params: latest parameters used for sim/build
- shards: {count: int, last_ids: list[str]}
- last_build: {
    bundle: str,
    dataset_hash: str,
    digest: str,
    flags: dict,
    time: str,
  }

Files are stored under ``workspace/state.json`` with UTF-8 encoding.
"""

from pathlib import Path
from typing import Any, Dict
import json


def _state_path(workspace: Path) -> Path:
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    return workspace / "state.json"


def read_manifest(workspace: Path) -> Dict[str, Any]:
    p = _state_path(workspace)
    if not p.exists():
        return {
            "params": {},
            "shards": {"count": 0, "last_ids": []},
            "last_build": None,
        }
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {
            "params": {},
            "shards": {"count": 0, "last_ids": []},
            "last_build": None,
        }


def write_manifest(workspace: Path, data: Dict[str, Any]) -> None:
    p = _state_path(workspace)
    text = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    p.write_text(text, encoding="utf-8")
