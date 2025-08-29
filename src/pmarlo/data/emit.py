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
from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np

from .shard import write_shard

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

    json_paths: List[Path] = []
    for i, traj in enumerate(paths):
        cvs, dtraj, source_info = extract_cvs(traj)
        names, _ = _validate_cvs(cvs)
        periodic = {
            name: bool((periodic_by_cv or {}).get(name, False)) for name in names
        }
        shard_id = f"shard_{i:04d}"
        json_path = write_shard(
            out_dir=out_dir,
            shard_id=shard_id,
            cvs=cvs,
            dtraj=dtraj,
            periodic=periodic,
            seed=int(seed_start + i),
            temperature=float(temperature),
            source=dict(source_info),
        )
        json_paths.append(json_path.resolve())

    return json_paths
