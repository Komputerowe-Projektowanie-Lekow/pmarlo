from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Union


def timestamp_dir(base_dir: Union[str, Path]) -> Path:
    """Create and return a unique timestamped directory under base_dir.

    The directory name uses YYYYMMDD-HHMMSS to preserve lexicographic sort order.
    The directory is created if it does not already exist.
    """
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(base_dir) / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir
