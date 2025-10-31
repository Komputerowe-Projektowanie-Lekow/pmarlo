from pathlib import Path
from typing import Sequence, Tuple, Dict, Any, List


def emit_shards(
        self,
        simulation: SimulationResult,
        request: ShardRequest,
        *,
        provenance: Optional[Dict[str, Any]] = None,
) -> ShardResult:
    shard_dir = self.layout.shards_dir / simulation.run_id
    ensure_directory(shard_dir)
    created = _timestamp()
    note = {
        "created_at": created,
        "kind": "demux",
        "run_id": simulation.run_id,
        "analysis_temperatures": simulation.analysis_temperatures,
        "topology": str(simulation.pdb_path),
        "traj_files": [str(p) for p in simulation.traj_files],
    }
    if provenance:
        note.update(provenance)
    shard_paths = emit_shards_rg_rmsd_windowed(
        pdb_file=simulation.pdb_path,
        traj_files=[str(p) for p in simulation.traj_files],
        out_dir=str(shard_dir),
        reference=str(request.reference) if request.reference else None,
        stride=int(max(1, request.stride)),
        temperature=float(request.temperature),
        seed_start=int(max(0, request.seed_start)),
        frames_per_shard=int(max(1, request.frames_per_shard)),
        hop_frames=(
            int(request.hop_frames)
            if request.hop_frames is not None and request.hop_frames > 0
            else None
        ),
        provenance=note,
    )
    shard_paths = _coerce_path_list(shard_paths)
    n_frames = 0
    for path in shard_paths:
        try:
            meta, _, _ = read_shard(path)
            n_frames += int(getattr(meta, "n_frames", 0))
        except Exception:
            continue
    result = ShardResult(
        run_id=simulation.run_id,
        shard_dir=shard_dir.resolve(),
        shard_paths=shard_paths,
        n_frames=int(n_frames),
        n_shards=len(shard_paths),
        temperature=float(request.temperature),
        stride=int(max(1, request.stride)),
        frames_per_shard=int(max(1, request.frames_per_shard)),
        hop_frames=(
            int(request.hop_frames)
            if request.hop_frames is not None and request.hop_frames > 0
            else None
        ),
        created_at=created,
    )
    self.state.append_shards(
        {
            "run_id": simulation.run_id,
            "directory": str(result.shard_dir),
            "paths": [str(p) for p in shard_paths],
            "temperature": float(request.temperature),
            "stride": int(max(1, request.stride)),
            "n_shards": len(shard_paths),
            "n_frames": int(n_frames),
            "frames_per_shard": int(max(1, request.frames_per_shard)),
            "hop_frames": (
                int(request.hop_frames)
                if request.hop_frames is not None and request.hop_frames > 0
                else None
            ),
            "created_at": created,
        }
    )
    return result

def delete_shard_batch(self, index: int) -> bool:
    """Delete a shard batch and its associated files."""
    entry = self.state.remove_shards(index)
    if entry is None:
        return False

    try:
        # Delete individual shard files
        paths = entry.get("paths", [])
        for path_str in paths:
            path = self._path_from_value(path_str)
            if path is not None and path.exists():
                path.unlink()  # Delete the .json file
                # Also delete associated .npz file
                npz_path = path.with_suffix(".npz")
                if npz_path.exists():
                    npz_path.unlink()

        # Delete shard directory if empty
        directory = self._path_from_value(entry.get("directory"))
        if directory is not None and directory.exists() and directory.is_dir():
            try:
                directory.rmdir()  # Only removes if empty
            except OSError:
                pass  # Directory not empty, that's OK

        return True
    except Exception:
        return False


def _reconcile_shard_state(self) -> None:
    """Remove shard batches from state if all referenced files are missing."""
    try:
        to_delete: List[int] = []
        for i, entry in enumerate(list(self.state.shards)):
            raw_paths = entry.get("paths", [])
            paths = [
                self._path_from_value(p)
                for p in raw_paths
            ]
            existing = [p for p in paths if p is not None and p.exists()]
            if len(existing) == 0:
                to_delete.append(i)
        for i in reversed(to_delete):
            # Best-effort removal (also attempts to clean empty dirs)
            if not self.delete_shard_batch(i):
                try:
                    self.state.remove_shards(i)
                except Exception:
                    pass
    except Exception:
        # Non-fatal; leave state as-is
        pass

def discover_shards(self) -> List[Path]:
    if not self.layout.shards_dir.exists():
        return []
    return sorted(self.layout.shards_dir.rglob("*.json"))


def shard_summaries(self) -> List[Dict[str, Any]]:
    # Return only shard batches that have existing files; trim missing paths.
    info: List[Dict[str, Any]] = []
    for entry in self.state.shards:
        resolved_paths: List[str] = []
        for raw_path in entry.get("paths", []):
            candidate = self._path_from_value(raw_path)
            if candidate is not None and candidate.exists():
                resolved_paths.append(str(candidate))
        if not resolved_paths:
            # Skip batches that no longer have files on disk
            continue
        e = dict(entry)
        e["paths"] = resolved_paths
        e["n_shards"] = len(resolved_paths)
        info.append(e)
    return info

class ShardsMixin:
    """Methods for processing trajectory shards."""

    def create_shards(
            self,
            result: "SimulationResult",
            reference_pdb: Path,
            ...
    ) -> Tuple[List[Path], str]:

    # ... your create_shards implementation

    def discover_shards(self, run_id: Optional[str] = None) -> List[Path]:

    # ... your discover_shards implementation

    def delete_shard_batch(self, index: int) -> bool:

    # ... your delete_shard_batch implementation

    def _reconcile_shard_state(self) -> None:
# ... your implementation
