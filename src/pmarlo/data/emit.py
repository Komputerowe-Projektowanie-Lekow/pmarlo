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

from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np

from pmarlo.shards.indexing import initialise_shard_indices
from pmarlo.utils.path_utils import ensure_directory

from .shard import write_shard

ProgressCB = Callable[[str, Mapping[str, Any]], None]


_RUN_PATTERN = re.compile(r"run[-_]?[\w\d]+", re.IGNORECASE)
_SEGMENT_PATTERN = re.compile(r"(?:segment|seg|part)[-_]?(\d+)", re.IGNORECASE)
_REPLICA_PATTERN = re.compile(r"rep(?:lica)?[-_]?(\d+)", re.IGNORECASE)


def _infer_run_id(path: Path) -> str:
    for parent in path.parents:
        match = _RUN_PATTERN.search(parent.name)
        if match:
            return parent.name
    return path.parent.name or "run"


def _infer_segment_id(path: Path) -> int:
    stem = path.stem
    match = _SEGMENT_PATTERN.search(stem)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            pass
    return 0


def _infer_replica_id(path: Path, kind: str) -> int:
    stem = path.stem
    match = _REPLICA_PATTERN.search(stem)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            pass
    if kind == "replica":
        raise ValueError(f"Unable to infer replica id from path '{path}'")
    return 0


def _normalise_source_metadata(path: Path, source: Mapping[str, Any]) -> Dict[str, Any]:
    data = dict(source) if source is not None else {}
    data.setdefault("traj", str(path))
    if "created_at" not in data:
        data["created_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
            "+00:00", "Z"
        )

    kind = str(data.get("kind", "")).lower()
    if not kind:
        stem = path.stem.lower()
        if "replica" in stem:
            kind = "replica"
        else:
            kind = "demux"
    if kind not in {"demux", "replica"}:
        raise ValueError(f"Unsupported shard kind '{kind}' for path '{path}'")
    data["kind"] = kind

    if "run_id" not in data or not str(data["run_id"]).strip():
        data["run_id"] = _infer_run_id(path)

    if "segment_id" not in data:
        data["segment_id"] = _infer_segment_id(path)

    if "replica_id" not in data:
        data["replica_id"] = _infer_replica_id(path, kind)

    if "exchange_window_id" not in data:
        data["exchange_window_id"] = 0

    return data


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
    ensure_directory(out_dir)
    paths = [Path(p) for p in traj_files]
    paths.sort()

    # Determine next shard index to avoid overwriting existing outputs
    shard_state = initialise_shard_indices(out_dir, seed_start)
    start_index = shard_state.start_index

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
        names, n_frames = _validate_cvs(cvs)
        column_order = tuple(cvs.keys())
        periodic_flags = {
            name: bool((periodic_by_cv or {}).get(name, False)) for name in names
        }
        seed = shard_state.seed_for(shard_index)
        source = _normalise_source_metadata(traj, source_info)
        required_keys = {"kind", "run_id", "segment_id", "replica_id"}
        missing_keys = required_keys - set(source)
        if missing_keys:
            raise ValueError(
                f"extract_cvs must provide source metadata keys: {sorted(missing_keys)}"
            )
        replica_id = int(source["replica_id"])
        segment_id = int(source["segment_id"])
        exchange_window_id = int(source.get("exchange_window_id", 0))
        source["replica_id"] = replica_id
        source["segment_id"] = segment_id
        source["exchange_window_id"] = exchange_window_id
        source["seed"] = int(seed)
        source["n_frames"] = n_frames
        ordered_periodic = [bool(periodic_flags.get(name, False)) for name in column_order]
        source["periodic"] = ordered_periodic
        t_kelvin = int(round(float(temperature)))
        # Include kind in shard_id to prevent collisions
        kind = str(source.get("kind", "demux")).lower()
        if kind == "replica":
            shard_id = f"replica_T{t_kelvin}K_seg{segment_id:04d}_rep{replica_id:03d}"
        else:
            shard_id = f"T{t_kelvin}K_seg{segment_id:04d}_rep{replica_id:03d}"
        json_path = write_shard(
            out_dir=out_dir,
            shard_id=shard_id,
            cvs=cvs,
            dtraj=dtraj,
            periodic=periodic_flags,
            seed=seed,
            temperature=float(temperature),
            source=source,
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
                    source.get("n_frames", 0) if isinstance(source, dict) else 0
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
