"""
Dataset catalog management with canonical shard identification.

This module provides functionality to catalog shards using collision-resistant
canonical identifiers, enabling robust validation and deduplication across
multiple runs and file types.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from pmarlo.io.shard_id import ShardId, parse_shard_id

logger = logging.getLogger(__name__)


class ShardCatalog:
    """
    Catalog of shards with canonical identification and validation.

    This class manages a collection of shards using canonical IDs as primary keys,
    providing robust validation against missing/extra shards and source kind mixing.

    Attributes
    ----------
    shards : Dict[str, ShardId]
        Mapping from canonical ID to ShardId object
    source_kinds : Set[str]
        Set of source kinds present in catalog
    run_ids : Set[str]
        Set of run IDs present in catalog
    """

    def __init__(self) -> None:
        """Initialize empty shard catalog."""
        self.shards: Dict[str, ShardId] = {}
        self.source_kinds: Set[str] = set()
        self.run_ids: Set[str] = set()

    def add_shard(self, shard_id: ShardId) -> None:
        """
        Add a shard to the catalog.

        Parameters
        ----------
        shard_id : ShardId
            Canonical shard identifier to add

        Raises
        ------
        PMarloError
            If shard with same canonical ID already exists
        """
        canonical = shard_id.canonical()

        if canonical in self.shards:
            existing = self.shards[canonical]
            if existing.source_path != shard_id.source_path:
                raise ValueError(
                    f"Canonical ID collision: {canonical} maps to both "
                    f"{existing.source_path} and {shard_id.source_path}"
                )
            # Same path, skip silently (idempotent)
            return

        self.shards[canonical] = shard_id
        self.source_kinds.add(shard_id.source_kind)
        self.run_ids.add(shard_id.run_id)

        logger.debug(f"Added shard to catalog: {canonical}")

    def add_from_path(self, source_path: Path, dataset_hash: str = "") -> None:
        """
        Parse and add shard from trajectory file path.

        Parameters
        ----------
        source_path : Path
            Path to trajectory file
        dataset_hash : str, optional
            Dataset hash for integrity verification
        """
        try:
            p = Path(source_path)
            # If a shard JSON is provided, resolve the original trajectory path from metadata
            if p.suffix.lower() == ".json":
                try:
                    from pmarlo.data.shard import (  # lazy import to avoid cycles
                        read_shard,
                    )

                    meta, _, _ = read_shard(p)
                    src = dict(getattr(meta, "source", {}))
                    src_path_str = (
                        src.get("traj")
                        or src.get("path")
                        or src.get("file")
                        or src.get("source_path")
                    )
                    if src_path_str:
                        source_path = Path(src_path_str)
                        shard_id = parse_shard_id(
                            Path(source_path), dataset_hash, require_exists=False
                        )
                        self.add_shard(shard_id)
                        return
                    else:
                        raise ValueError("no source path in shard metadata")
                except Exception as ex:
                    # Fallback: synthesize ShardId directly from JSON using tolerant parser
                    try:
                        from pmarlo.io.shards import (
                            build_shard_id_from_json_fallback as _fallback,
                        )

                        sid = _fallback(p, dataset_hash)
                        self.add_shard(sid)
                        logger.debug(
                            "Catalog: used tolerant fallback for %s -> %s",
                            str(p),
                            sid.canonical(),
                        )
                        return
                    except Exception as e2:
                        logger.debug(
                            "Catalog fallback parse failed for %s: %s", str(p), e2
                        )
                        raise ValueError(f"cannot resolve source from shard JSON: {ex}")

            # Non-JSON path: parse as a trajectory
            shard_id = parse_shard_id(
                Path(source_path), dataset_hash, require_exists=False
            )
            self.add_shard(shard_id)
        except Exception as e:
            logger.debug(f"Catalog: failed to parse shard from {source_path}: {e}")

    def add_from_paths(self, source_paths: List[Path], dataset_hash: str = "") -> None:
        """
        Add multiple shards from trajectory file paths.

        Parameters
        ----------
        source_paths : List[Path]
            List of paths to trajectory files
        dataset_hash : str, optional
            Dataset hash for integrity verification
        """
        for path in source_paths:
            self.add_from_path(path, dataset_hash)

    def validate_against_used(
        self, used_canonical_ids: Set[str] | List[str]
    ) -> Dict[str, List[str]]:
        """
        Validate catalog against set of used canonical IDs.

        This method compares the catalog's canonical IDs against those reported
        as used by the build process, identifying missing and extra shards.

        Parameters
        ----------
        used_canonical_ids : Set[str]
            Set of canonical IDs reported as used by build process

        Returns
        -------
        Dict[str, List[str]]
            Validation results with keys:
            - "missing": List of canonical IDs present in catalog but not used
            - "extra": List of canonical IDs used but not in catalog
            - "warnings": List of warning messages about data quality issues
        """
        catalog_ids = set(self.shards.keys())
        used_ids_set = set(used_canonical_ids)
        missing_ids = catalog_ids - used_ids_set
        extra_ids = used_ids_set - catalog_ids

        warnings = []

        # Check for source kind mixing
        if len(self.source_kinds) > 1:
            warnings.append(
                f"Mixed source kinds detected: {sorted(self.source_kinds)}. "
                "Consider processing demux and replica files separately for cleaner analysis."
            )

        # Check for temperature/replica distribution
        temp_distribution = self._analyze_temperature_distribution()
        if temp_distribution.get("warnings"):
            warnings.extend(temp_distribution["warnings"])

        # Check for run distribution
        if len(self.run_ids) > 1:
            warnings.append(
                f"Multiple runs detected: {sorted(self.run_ids)}. "
                "Ensure consistent parameters across runs."
            )

        return {
            "missing": sorted(missing_ids),
            "extra": sorted(extra_ids),
            "warnings": warnings,
        }

    def get_canonical_ids(self) -> List[str]:
        """
        Get all canonical IDs in the catalog.

        Returns
        -------
        List[str]
            Sorted list of all canonical IDs
        """
        return sorted(self.shards.keys())

    def get_shard_info_table(self) -> List[Dict[str, str]]:
        """
        Generate tabular information about shards for diagnostics.

        Returns
        -------
        List[Dict[str, str]]
            List of dictionaries with shard information suitable for display
        """
        rows = []
        for canonical, shard in self.shards.items():
            temp_or_replica = (
                f"T{shard.temperature_K}K"
                if shard.temperature_K is not None
                else f"R{shard.replica_index}"
            )

            rows.append(
                {
                    "canonical_id": canonical,
                    "run_id": shard.run_id,
                    "source_kind": shard.source_kind,
                    "temp_or_replica": temp_or_replica,
                    "local_index": str(shard.local_index),
                    "source_path": str(shard.source_path),
                }
            )

        # Sort by canonical ID for consistent ordering
        return sorted(rows, key=lambda x: x["canonical_id"])

    def _analyze_temperature_distribution(self) -> Dict[str, List[str]]:
        """
        Analyze temperature/replica distribution for quality warnings.

        Returns
        -------
        Dict[str, List[str]]
            Analysis results with "warnings" key containing warning messages
        """
        warnings = []

        # Group by source kind
        demux_temps = []
        replica_indices = []

        for shard in self.shards.values():
            if shard.source_kind == "demux" and shard.temperature_K is not None:
                demux_temps.append(shard.temperature_K)
            elif shard.source_kind == "replica" and shard.replica_index is not None:
                replica_indices.append(shard.replica_index)

        # Check demux temperature gaps
        if demux_temps:
            unique_temps = sorted(set(demux_temps))
            expected_range = range(
                min(unique_temps), max(unique_temps) + 50, 50
            )  # Assume 50K spacing
            missing_temps = set(expected_range) - set(unique_temps)
            if missing_temps:
                warnings.append(
                    f"Missing temperatures in demux data: {sorted(missing_temps)}K. "
                    "This may indicate incomplete demultiplexing."
                )

        # Check replica index continuity
        if replica_indices:
            unique_indices = sorted(set(replica_indices))
            expected_indices = list(range(len(unique_indices)))
            if unique_indices != expected_indices:
                warnings.append(
                    f"Replica indices are not contiguous: found {unique_indices}, "
                    f"expected {expected_indices}. This may indicate missing replicas."
                )

        return {"warnings": warnings}


def build_catalog_from_paths(
    source_paths: List[Path], dataset_hash: str = ""
) -> ShardCatalog:
    """
    Build a shard catalog from a list of trajectory file paths.

    Parameters
    ----------
    source_paths : List[Path]
        List of paths to trajectory files
    dataset_hash : str, optional
        Dataset hash for integrity verification

    Returns
    -------
    ShardCatalog
        Populated catalog of shards
    """
    catalog = ShardCatalog()
    catalog.add_from_paths(source_paths, dataset_hash)
    return catalog


def validate_shard_usage(
    available_paths: List[Path], used_canonical_ids: Set[str], dataset_hash: str = ""
) -> Dict[str, List[str]]:
    """
    High-level validation function for shard usage consistency.

    This is a convenience function that builds a catalog from available paths
    and validates it against used canonical IDs.

    Parameters
    ----------
    available_paths : List[Path]
        List of available trajectory file paths
    used_canonical_ids : Set[str]
        Set of canonical IDs reported as used
    dataset_hash : str, optional
        Dataset hash for integrity verification

    Returns
    -------
    Dict[str, List[str]]
        Validation results as returned by ShardCatalog.validate_against_used()
    """
    catalog = build_catalog_from_paths(available_paths, dataset_hash)
    return catalog.validate_against_used(used_canonical_ids)
