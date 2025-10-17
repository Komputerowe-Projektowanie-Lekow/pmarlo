from __future__ import annotations

"""Shard emission helpers for deterministic datasets."""

from pathlib import Path
from typing import Callable, Iterable, List, Optional

from pmarlo.transform.progress import ProgressCB, ProgressReporter
from pmarlo.utils.path_utils import ensure_directory

from .format import write_shard
from .schema import Shard, validate_invariants

__all__ = ["emit_shards_from_trajectories", "ExtractShard"]

ExtractShard = Callable[[Path], Shard]


def emit_shards_from_trajectories(
    traj_files: Iterable[Path],
    out_dir: Path,
    *,
    build_shard: ExtractShard,
    progress_callback: Optional[ProgressCB] = None,
) -> List[Path]:
    """Emit shard files from trajectory inputs using provided builder."""

    paths = sorted(Path(p) for p in traj_files)
    out_dir = Path(out_dir)
    ensure_directory(out_dir)

    reporter = ProgressReporter(progress_callback)
    total = len(paths)
    reporter.emit(
        "emit_begin",
        {
            "n_inputs": total,
            "out_dir": str(out_dir),
            "current": 0,
            "total": total,
        },
    )

    written: List[Path] = []
    for idx, path in enumerate(paths):
        reporter.emit(
            "emit_one_begin",
            {
                "index": idx,
                "traj": str(path),
                "current": idx,
                "total": total,
            },
        )
        shard = build_shard(path)
        validate_invariants(shard)
        json_path = write_shard(shard, out_dir)
        written.append(json_path.resolve())
        reporter.emit(
            "emit_one_end",
            {
                "index": idx,
                "traj": str(path),
                "shard": shard.meta.shard_id,
                "current": idx + 1,
                "total": total,
            },
        )

    reporter.emit(
        "emit_end",
        {
            "n_shards": len(written),
            "current": total,
            "total": total,
        },
    )
    return written
