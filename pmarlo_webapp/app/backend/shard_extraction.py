"""Shard extraction with configurable feature profiles."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import mdtraj as md

from pmarlo.api import compute_features
from pmarlo.data.shard import write_shard
from pmarlo.utils.path_utils import ensure_directory

logger = logging.getLogger(__name__)

__all__ = ["extract_shards_with_features"]


def extract_shards_with_features(
    pdb_file: str | Path,
    traj_files: List[str | Path],
    out_dir: str | Path,
    feature_specs: List[str],
    *,
    stride: int = 1,
    temperature: float = 300.0,
    seed_start: int = 0,
    frames_per_shard: int = 5000,
    hop_frames: int | None = None,
    provenance: Dict[str, Any] | None = None,
) -> List[Path]:
    """Extract shards with configurable molecular features.

    Parameters
    ----------
    pdb_file : str | Path
        Topology PDB file
    traj_files : List[str | Path]
        Trajectory DCD files
    out_dir : str | Path
        Output directory for shards
    feature_specs : List[str]
        Feature specifications (e.g., ["distance([0, 1])", "angle([0, 1, 2])"])
    stride : int
        Frame stride for sampling
    temperature : float
        Temperature in Kelvin
    seed_start : int
        Starting seed for shard IDs
    frames_per_shard : int
        Maximum frames per shard
    hop_frames : int | None
        Hop size for overlapping windows (None = no overlap)
    provenance : Dict[str, Any] | None
        Additional metadata

    Returns
    -------
    List[Path]
        Paths to created shard JSON files
    """
    pdb_path = Path(pdb_file)
    out_path = Path(out_dir)
    ensure_directory(out_path)

    logger.info(f"Loading trajectories with topology {pdb_path}")
    logger.info(f"  Trajectory files: {len(traj_files)}")
    logger.info(f"  Feature specs: {len(feature_specs)}")
    logger.info(f"  Stride: {stride}, Temperature: {temperature}K")

    # Load all trajectories
    all_frames = []
    for traj_file in traj_files:
        logger.info(f"Loading {Path(traj_file).name}...")
        traj = md.load(str(traj_file), top=str(pdb_path), stride=stride)
        all_frames.append(traj)
        logger.info(f"  Loaded {traj.n_frames} frames (after stride)")

    # Concatenate all trajectories
    if not all_frames:
        raise ValueError("No trajectory frames loaded")

    logger.info("Concatenating all trajectory frames...")
    full_traj = md.join(all_frames)
    logger.info(f"Total frames: {full_traj.n_frames}")

    # Compute features
    logger.info("Computing molecular features...")
    X, columns, periodic = compute_features(full_traj, feature_specs)
    logger.info(f"Features computed: shape={X.shape}, columns={columns}")

    # Create shards
    shard_paths = []
    total_frames = X.shape[0]
    shard_idx = 0

    if hop_frames is None or hop_frames <= 0:
        # Non-overlapping shards
        frame_idx = 0
        while frame_idx < total_frames:
            end_idx = min(frame_idx + frames_per_shard, total_frames)
            shard_data = X[frame_idx:end_idx]

            if shard_data.shape[0] == 0:
                break

            # Calculate original trajectory frame range (accounting for stride)
            original_start = frame_idx * stride
            original_end = end_idx * stride

            shard_path = _write_shard(
                out_dir=out_path,
                shard_idx=shard_idx,
                seed_start=seed_start,
                data=shard_data,
                columns=columns,
                periodic_flags=periodic,
                temperature=temperature,
                frame_range=(original_start, original_end),
                traj_files=traj_files,
                provenance=provenance,
                stride=stride,
            )
            shard_paths.append(shard_path)

            logger.info(f"Created shard {shard_idx}: {shard_path.name} ({shard_data.shape[0]} frames)")
            frame_idx = end_idx
            shard_idx += 1
    else:
        # Overlapping shards with hop_frames
        frame_idx = 0
        while frame_idx < total_frames:
            end_idx = min(frame_idx + frames_per_shard, total_frames)
            shard_data = X[frame_idx:end_idx]

            if shard_data.shape[0] == 0:
                break

            # Calculate original trajectory frame range (accounting for stride)
            original_start = frame_idx * stride
            original_end = end_idx * stride

            shard_path = _write_shard(
                out_dir=out_path,
                shard_idx=shard_idx,
                seed_start=seed_start,
                data=shard_data,
                columns=columns,
                periodic_flags=periodic,
                temperature=temperature,
                frame_range=(original_start, original_end),
                traj_files=traj_files,
                provenance=provenance,
                stride=stride,
            )
            shard_paths.append(shard_path)

            logger.info(f"Created shard {shard_idx}: {shard_path.name} ({shard_data.shape[0]} frames)")
            frame_idx += hop_frames
            shard_idx += 1

    logger.info(f"Created {len(shard_paths)} shards with {total_frames} total frames")
    return shard_paths


def _write_shard(
    out_dir: Path,
    shard_idx: int,
    seed_start: int,
    data: np.ndarray,
    columns: List[str],
    periodic_flags: np.ndarray,
    temperature: float,
    frame_range: tuple[int, int],
    traj_files: List[str | Path],
    provenance: Dict[str, Any] | None,
    stride: int = 1,
) -> Path:
    """Write a single shard file.

    Parameters
    ----------
    out_dir : Path
        Output directory
    shard_idx : int
        Shard index
    seed_start : int
        Base seed
    data : np.ndarray
        Feature matrix (n_frames, n_features)
    columns : List[str]
        Feature column names
    periodic_flags : np.ndarray
        Boolean array indicating which features are periodic
    temperature : float
        Temperature in Kelvin
    frame_range : tuple[int, int]
        Frame range [start, stop) from original trajectory
    traj_files : List[str | Path]
        Original trajectory file paths
    provenance : Dict[str, Any] | None
        Metadata
    stride : int
        Frame stride used when reading the original trajectory

    Returns
    -------
    Path
        Path to created shard JSON file
    """
    # Create CV dictionary from data columns
    cvs = {col: data[:, idx] for idx, col in enumerate(columns)}

    # Create periodic dictionary from flags (column -> bool)
    periodic = {
        col: bool(periodic_flags[idx]) if idx < len(periodic_flags) else False
        for idx, col in enumerate(columns)
    }

    # Prepare source metadata with required fields
    from datetime import datetime

    # Generate a run_id if not provided
    run_id = "shard_extraction"
    if provenance and "run_id" in provenance:
        run_id = provenance["run_id"]

    # Calculate the original trajectory frame count (before stride)
    # This is critical for effective_frame_stride calculation in aggregation
    original_frame_count = frame_range[1] - frame_range[0]

    source_metadata = {
        "created_at": datetime.now().isoformat(),
        "kind": "demux",  # Use "demux" for standard shard extraction (not replica exchange)
        "run_id": run_id,
        "replica_id": 0,
        "segment_id": shard_idx,
        "range": list(frame_range),  # Frame range from original trajectory (matches existing convention)
        "stride": stride,  # Frame stride used when reading trajectory
        "frame_stride": stride,  # Explicitly set frame_stride for conformations analysis
        "n_frames": original_frame_count,  # Original frame count (before stride) for effective_frame_stride calculation
        "traj": str(traj_files[0]) if traj_files else "",  # Primary trajectory file (matches existing convention)
        "traj_files": [str(p) for p in traj_files],  # All trajectory files for multi-file support
    }
    # Merge with any user-provided provenance
    if provenance:
        source_metadata.update(provenance)

    enriched_source = dict(source_metadata)
    enriched_source.update(
        temperature_K=float(temperature),
        columns=list(columns),
        periodic=periodic,
    )

    # Generate canonical shard_id that matches canonical_shard_id() format
    # For kind="demux": T{t_kelvin}K_{run_suffix}_seg{segment:04d}_rep{replica:03d}
    t_kelvin = int(temperature)
    replica_id = 0
    run_suffix = str(run_id).replace("run_", "") if run_id else "default"
    shard_id = f"T{t_kelvin}K_{run_suffix}_seg{shard_idx:04d}_rep{replica_id:03d}"

    # Write shard
    shard_path = write_shard(
        out_dir=out_dir,
        shard_id=shard_id,
        cvs=cvs,
        dtraj=None,  # No discretization yet
        periodic=periodic,
        seed=seed_start + shard_idx,
        temperature=temperature,
        source=enriched_source,
    )

    try:
        with shard_path.open("r+", encoding="utf-8") as handle:
            payload = json.load(handle)
            payload["source"] = enriched_source
            handle.seek(0)
            json.dump(payload, handle, indent=2)
            handle.truncate()
    except Exception:
        logger.warning("Failed to enrich shard %s with top-level source metadata", shard_path, exc_info=True)
        raise

    # write_shard returns the JSON file path directly
    return shard_path
