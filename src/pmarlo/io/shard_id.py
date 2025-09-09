"""
Canonical shard identification system for collision-free dataset management.

This module provides a dataclass-based approach to shard identification that
eliminates collisions across multiple runs and file types by encoding run metadata,
source kind, and positional information into a canonical string key.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional


@dataclass(frozen=True)
class ShardId:
    """
    Canonical shard identifier with collision-resistant properties.

    This dataclass encapsulates all metadata needed to uniquely identify a shard
    across different runs, file types, and temperatures/replicas.

    Attributes
    ----------
    run_id : str
        Run identifier extracted from directory name (e.g., "run-20250906-170155")
    source_kind : Literal["demux", "replica"]
        Type of source trajectory file ("demux" for demux_T*.dcd, "replica" for replica_*.dcd)
    temperature_K : Optional[int]
        Temperature in Kelvin for demux files (None for replica files)
    replica_index : Optional[int]
        Replica index for replica files (None for demux files)
    local_index : int
        Monotone index within run for stable ordering (0-based)
    source_path : Path
        Absolute path to the source trajectory file
    dataset_hash : str
        SHA-256 hash of the dataset content for integrity verification
    """

    run_id: str
    source_kind: Literal["demux", "replica"]
    temperature_K: Optional[int]
    replica_index: Optional[int]
    local_index: int
    source_path: Path
    dataset_hash: str

    def canonical(self) -> str:
        """
        Generate canonical string representation for this shard.

        Returns
        -------
        str
            Canonical identifier in format: "{run_id}:{source_kind}:{temp_or_replica}:{local_index}"
            Examples:
            - "run-20250906-170155:demux:300:0"
            - "run-20250906-170155:replica:0:1"
        """
        temp_or_replica = (
            f"T{self.temperature_K}"
            if self.temperature_K is not None
            else f"R{self.replica_index}"
        )
        return f"{self.run_id}:{self.source_kind}:{temp_or_replica}:{self.local_index}"

    @classmethod
    def from_canonical(
        cls, canonical_str: str, source_path: Path, dataset_hash: str
    ) -> ShardId:
        """
        Reconstruct ShardId from canonical string representation.

        Parameters
        ----------
        canonical_str : str
            Canonical identifier string
        source_path : Path
            Path to the source trajectory file
        dataset_hash : str
            SHA-256 hash of the dataset content

        Returns
        -------
        ShardId
            Reconstructed shard identifier

        Raises
        ------
        ValueError
            If canonical string format is invalid
        """
        parts = canonical_str.split(":")
        if len(parts) != 4:
            raise ValueError(f"Invalid canonical format: {canonical_str}")

        run_id, source_kind, temp_or_replica, local_index_str = parts

        # Validate source_kind
        if source_kind not in ["demux", "replica"]:
            raise ValueError(
                f"Invalid source_kind '{source_kind}' in canonical: {canonical_str}"
            )

        # Parse temperature/replica
        temperature_K = None
        replica_index = None

        if temp_or_replica.startswith("T"):
            try:
                temperature_K = int(temp_or_replica[1:])
            except ValueError:
                raise ValueError(
                    f"Invalid temperature format '{temp_or_replica}' in canonical: {canonical_str}"
                )
        elif temp_or_replica.startswith("R"):
            try:
                replica_index = int(temp_or_replica[1:])
            except ValueError:
                raise ValueError(
                    f"Invalid replica format '{temp_or_replica}' in canonical: {canonical_str}"
                )
        else:
            raise ValueError(
                f"Invalid temp/replica format '{temp_or_replica}' in canonical: {canonical_str}"
            )

        # Parse local index
        try:
            local_index = int(local_index_str)
        except ValueError:
            raise ValueError(
                f"Invalid local_index '{local_index_str}' in canonical: {canonical_str}"
            )

        return cls(
            run_id=run_id,
            source_kind=source_kind,  # type: ignore[arg-type]
            temperature_K=temperature_K,
            replica_index=replica_index,
            local_index=local_index,
            source_path=source_path,
            dataset_hash=dataset_hash,
        )


def parse_shard_id(
    source_path: Path, dataset_hash: str = "", require_exists: bool = True
) -> ShardId:
    """
    Parse trajectory file path and extract canonical shard identifier.

    This function analyzes the file path to extract run metadata, determines the
    source kind based on filename patterns, and assigns a stable local index
    within the run.

    Parameters
    ----------
    source_path : Path
        Path to the trajectory file (demux_T*.dcd or replica_*.dcd)
    dataset_hash : str, optional
        SHA-256 hash of the dataset content (default: empty string)

    Returns
    -------
    ShardId
        Parsed canonical shard identifier

    Raises
    ------
    PMarloError
        If path cannot be parsed or required metadata is missing

    Examples
    --------
    >>> path = Path("/data/run-20250906-170155/demux_T300K.dcd")
    >>> shard_id = parse_shard_id(path)
    >>> shard_id.canonical()
    'run-20250906-170155:demux:T300:0'

    >>> path = Path("/data/run-20250906-170155/replica_00.dcd")
    >>> shard_id = parse_shard_id(path)
    >>> shard_id.canonical()
    'run-20250906-170155:replica:R0:1'
    """
    if require_exists and not source_path.exists():
        raise ValueError(f"Source path does not exist: {source_path}")

    source_path = source_path.resolve()

    # Extract run_id from nearest run-* directory
    run_id = _extract_run_id(source_path)
    if not run_id:
        raise ValueError(f"Could not extract run_id from path: {source_path}")

    # Determine source_kind and extract temperature/replica
    filename = source_path.name
    temperature_K = None
    replica_index = None
    source_kind = None

    # Check for demux pattern: demux_T{temp}K.dcd
    demux_match = re.match(r"demux_T(\d+)K\.dcd$", filename)
    if demux_match:
        source_kind = "demux"
        temperature_K = int(demux_match.group(1))
    else:
        # Check for replica pattern: replica_{index}.dcd
        replica_match = re.match(r"replica_(\d+)\.dcd$", filename)
        if replica_match:
            source_kind = "replica"
            replica_index = int(replica_match.group(1))
        else:
            # Strict mode: do not guess from arbitrary filenames
            raise ValueError(f"Unrecognized filename pattern: {filename}")

    # Compute local_index by stable sort within run
    local_index = (
        _compute_local_index(source_path, run_id, source_kind) if require_exists else 0
    )

    return ShardId(
        run_id=run_id,
        source_kind=source_kind,  # type: ignore[arg-type]
        temperature_K=temperature_K,
        replica_index=replica_index,
        local_index=local_index,
        source_path=source_path,
        dataset_hash=dataset_hash,
    )


def _extract_run_id(source_path: Path) -> Optional[str]:
    """
    Extract run identifier from path by finding nearest run-* directory.

    Parameters
    ----------
    source_path : Path
        File path to analyze

    Returns
    -------
    Optional[str]
        Run identifier (e.g., "run-20250906-170155") or None if not found
    """
    current = source_path.parent

    # Search up to 5 levels up for run-* directory
    for _ in range(5):
        if current.name.startswith("run-"):
            return current.name
        if current.parent == current:  # Reached filesystem root
            break
        current = current.parent

    return None


def _compute_local_index(source_path: Path, run_id: str, source_kind: str) -> int:
    """
    Compute stable local index for shard within its run and source kind.

    This function finds all files of the same source_kind within the run directory
    and assigns a stable index based on alphabetical ordering of filenames.

    Parameters
    ----------
    source_path : Path
        Path to the trajectory file
    run_id : str
        Run identifier
    source_kind : str
        Source kind ("demux" or "replica")

    Returns
    -------
    int
        Zero-based local index for stable ordering
    """
    # Find run directory
    current = source_path.parent
    run_dir = None

    for _ in range(5):
        if current.name == run_id:
            run_dir = current
            break
        if current.parent == current:
            break
        current = current.parent

    if run_dir is None:
        return 0  # Fallback if run directory not found

    # Collect all files of same source_kind in run directory
    pattern = "demux_T*K.dcd" if source_kind == "demux" else "replica_*.dcd"
    sibling_files = sorted(run_dir.glob(pattern))

    # Find position of our file in sorted list
    try:
        return sibling_files.index(source_path)
    except ValueError:
        # File not in list (shouldn't happen), return 0 as fallback
        return 0
