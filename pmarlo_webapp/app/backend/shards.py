import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from pmarlo.data.shard import read_shard
from pmarlo.utils.path_utils import ensure_directory

from .types import ShardRequest, ShardResult, SimulationResult
from .utils import _coerce_path_list, _timestamp, emit_shards_rg_rmsd_windowed

logger = logging.getLogger(__name__)


class ShardsMixin:
    """Methods for processing trajectory shards.

    This class is mixed into the Backend class to provide shard demultiplexing,
    discovery, and management operations.
    """

    def emit_shards(
        self,
        simulation: SimulationResult,
        request: ShardRequest,
        *,
        provenance: Optional[Dict[str, Any]] = None,
    ) -> ShardResult:
        """Generate shards from a simulation result.

        Parameters
        ----------
        simulation : SimulationResult
            The simulation result containing trajectories to shard
        request : ShardRequest
            Configuration for shard generation
        provenance : Optional[Dict[str, Any]]
            Additional metadata to include in shard provenance

        Returns
        -------
        ShardResult
            Information about the generated shards
        """
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

        logger.info(f"Shard emission returned {len(shard_paths)} paths")
        for idx, p in enumerate(shard_paths):
            logger.debug(f"  Shard {idx}: {p}, exists={p.exists()}, size={p.stat().st_size if p.exists() else 'N/A'}")

        # Count total frames across all shards
        n_frames = 0
        failed_reads = []
        for path in shard_paths:
            try:
                # Read the shard to count frames
                meta, data, src = read_shard(path)
                frames_in_shard = int(getattr(meta, "n_frames", 0))
                n_frames += frames_in_shard
                logger.debug(f"Shard {path.name}: {frames_in_shard} frames, data shape: {data.shape if data is not None else 'None'}")
            except Exception as e:
                logger.error(f"Failed to read shard {path}: {type(e).__name__}: {e}", exc_info=True)
                failed_reads.append(path.name)
                # Check if file exists
                if not path.exists():
                    logger.error(f"  → Shard file does not exist: {path}")
                else:
                    try:
                        size = path.stat().st_size
                        logger.error(f"  → Shard file exists, size: {size} bytes")
                        # Check for companion npz file
                        npz_path = path.with_suffix(".npz")
                        if npz_path.exists():
                            npz_size = npz_path.stat().st_size
                            logger.error(f"  → NPZ file exists, size: {npz_size} bytes")
                        else:
                            logger.error(f"  → NPZ file missing: {npz_path}")
                    except Exception as stat_err:
                        logger.error(f"  → Could not stat file: {stat_err}")
                continue

        if n_frames == 0 and len(shard_paths) > 0:
            logger.warning(
                f"Shard emission produced {len(shard_paths)} shard files but counted 0 frames total. "
                f"Failed to read {len(failed_reads)} shards: {failed_reads}. "
                f"This may indicate an issue with shard generation or file I/O."
            )

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

        logger.info(f"Generated {len(shard_paths)} shards for run {simulation.run_id}, total frames: {n_frames}")
        return result

    def delete_shard_batch(self, index: int) -> bool:
        """Delete a shard batch and its associated files.

        Parameters
        ----------
        index : int
            Index of the shard batch in state

        Returns
        -------
        bool
            True if deletion was successful, False otherwise
        """
        entry = self.state.remove_shards(index)
        if entry is None:
            logger.warning(f"No shard batch found at index {index}")
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

            logger.info(f"Deleted shard batch at index {index}")
            return True
        except Exception as e:
            logger.error(f"Error deleting shard batch {index}: {e}")
            return False

    def discover_shards(self) -> List[Path]:
        """Discover all shard JSON files in the shards directory.

        Returns
        -------
        List[Path]
            List of paths to shard JSON files
        """
        if not self.layout.shards_dir.exists():
            return []
        return sorted(self.layout.shards_dir.rglob("*.json"))

    def shard_summaries(self) -> List[Dict[str, Any]]:
        """Get summaries of all shard batches with existing files.

        Returns only shard batches that have existing files on disk,
        and trims missing paths from each batch.

        Returns
        -------
        List[Dict[str, Any]]
            List of shard batch summaries
        """
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
