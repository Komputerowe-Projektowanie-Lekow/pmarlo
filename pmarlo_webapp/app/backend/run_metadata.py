from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

RUN_PLAN_FILENAME = "run_plan.json"


def load_run_plan(run_dir: Path) -> Optional[Dict[str, Any]]:
    """Load the serialized run plan if it exists."""
    path = run_dir / RUN_PLAN_FILENAME
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def save_run_plan(run_dir: Path, payload: Dict[str, Any]) -> Path:
    """Persist the run plan next to the simulation outputs."""
    path = run_dir / RUN_PLAN_FILENAME
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


__all__ = [
    "RUN_PLAN_FILENAME",
    "load_run_plan",
    "save_run_plan",
]
