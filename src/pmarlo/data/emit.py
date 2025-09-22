from __future__ import annotations

"""
Emit deterministic shard files from many short trajectory inputs.

You provide a pluggable CV extractor callable returning:
- cvs: dict name -> 1-D arrays (equal lengths)
- dtraj: optional 1-D integer labels, or None
- source_info: extra provenance merged into the shard metadata

The function writes shard_{i:04d}.npz/.json under an output directory with
canonical JSON and integrity hashes suitable for reproducible map→reduce.
"""

from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np

from .shard import write_shard

ProgressCB = Callable[[str, Mapping[str, Any]], None]


class ProgressReporter:
    """Minimal progress reporter used when the full transform stack is unavailable."""

    def __init__(self, cb: Optional[ProgressCB]) -> None:
        self._cb = cb

    def emit(self, event: str, data: Mapping[str, Any]) -> None:
        if self._cb is None:
            return
        try:
            self._cb(event, dict(data))
        except Exception:
            pass


# Type alias for CV extractor callable
ExtractCVs = Callable[[Path], Tuple[Dict[str, np.ndarray], np.ndarray | None, Dict]]


def _validate_cvs(cvs: Dict[str, np.ndarray]) -> Tuple[Tuple[str, ...], int]:
    if not cvs:
        raise ValueError("extract_cvs returned no CVs")
    names = tuple(sorted(cvs.keys()))
    n = -1
    for k in names:
        arr = np.asarray(cvs[k])
        if arr.ndim != 1:
            raise ValueError(f"CV '{k}' must be 1-D array, got shape {arr.shape}")
        if n < 0:
            n = int(arr.shape[0])
        elif int(arr.shape[0]) != n:
            raise ValueError("All CV arrays must have the same length")
    return names, n


def emit_shards_from_trajectories(
    traj_files: Iterable[Path],
    out_dir: Path,
    *,
    extract_cvs: ExtractCVs,
    seed_start: int = 0,
    temperature: float = 300.0,
    periodic_by_cv: Dict[str, bool] | None = None,
    progress_callback: Optional[ProgressCB] = None,
) -> List[Path]:
    """Emit deterministic shards from a list of trajectory files.

    Parameters
    ----------
    traj_files:
        Iterable of trajectory paths. Order is made stable by sorting.
    out_dir:
        Output directory where shard ``.json``/``.npz`` files are written.
    extract_cvs:
        Callable extracting (cvs, dtraj, source_info) from a trajectory path.
    seed_start:
        Base seed; seed per shard is ``seed_start + i``.
    temperature:
        Temperature to store in metadata.
    periodic_by_cv:
        Optional map of CV name to periodicity; defaults to False.

    Returns
    -------
    list[Path]
        Absolute paths to the emitted shard JSON files.
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [Path(p) for p in traj_files]
    paths.sort()

    # Determine next shard index to avoid overwriting existing outputs
    existing_indices: List[int] = []
    for existing_json in sorted(out_dir.glob("shard_*.json")):
        stem = existing_json.stem
        try:
            existing_indices.append(int(stem.split("_")[1]))
        except (IndexError, ValueError):
            continue
    start_index = max(existing_indices) + 1 if existing_indices else 0
    seed_offset = int(seed_start) + start_index

    json_paths: List[Path] = []

    # Wrap callback with ProgressReporter to enrich with elapsed and ETA
    reporter = ProgressReporter(progress_callback)

    def _emit(event: str, data: Mapping[str, Any]) -> None:
        try:
            reporter.emit(event, data)
        except Exception:
            pass

    _emit(
        "emit_begin",
        {
            "n_inputs": len(paths),
            "out_dir": str(out_dir),
            "temperature": float(temperature),
            # Provide normalized progress fields for ETA computation
            "current": 0,
            "total": int(len(paths)),
        },
    )
    for i, traj in enumerate(paths):
        shard_index = start_index + i
        _emit(
            "emit_one_begin",
            {
                "index": int(i),
                "traj": str(traj),
                # Normalized progress for console ETA
                "current": int(i + 1),
                "total": int(len(paths)),
            },
        )
        cvs, dtraj, source_info = extract_cvs(traj)
        names, _ = _validate_cvs(cvs)
        periodic = {
            name: bool((periodic_by_cv or {}).get(name, False)) for name in names
        }
        shard_id = f"shard_{shard_index:04d}"
        json_path = write_shard(
            out_dir=out_dir,
            shard_id=shard_id,
            cvs=cvs,
            dtraj=dtraj,
            periodic=periodic,
            seed=int(seed_offset + i),
            temperature=float(temperature),
            source=dict(source_info),
        )
        json_paths.append(json_path.resolve())
        _emit(
            "emit_one_end",
            {
                "index": int(i),
                "traj": str(traj),
                "shard": shard_id,
                # Frames processed in this input if extractor provided it
                "frames": int(
                    source_info.get("n_frames", 0)
                    if isinstance(source_info, dict)
                    else 0
                ),
                # Keep progress fields consistent for ETA
                "current": int(i + 1),
                "total": int(len(paths)),
            },
        )

    _emit(
        "emit_end",
        {
            "n_shards": len(json_paths),
            "current": int(len(paths)),
            "total": int(len(paths)),
        },
    )
    return json_paths
