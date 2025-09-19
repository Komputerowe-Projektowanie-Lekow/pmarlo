from __future__ import annotations

"""Shard catalog utilities backed by strict shard metadata."""

import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set

from pmarlo.shards.discover import discover_shard_jsons
from pmarlo.shards.meta import load_shard_meta

from .shard_id import ShardId

logger = logging.getLogger(__name__)

__all__ = [
    "ShardCatalog",
    "build_catalog_from_paths",
    "validate_shard_usage",
]


class ShardCatalog:
    """Catalog of shards keyed by canonical identifiers."""

    def __init__(self) -> None:
        self.shards: Dict[str, ShardId] = {}
        self.source_kinds: Set[str] = set()
        self.run_ids: Set[str] = set()

    def add_shard(self, shard_id: ShardId) -> None:
        canonical = shard_id.canonical()
        existing = self.shards.get(canonical)
        if existing is not None and existing.json_path != shard_id.json_path:
            raise ValueError(
                "Canonical ID collision: "
                f"{canonical} already mapped to {existing.json_path}, got {shard_id.json_path}"
            )

        self.shards[canonical] = shard_id
        if shard_id.source_kind:
            self.source_kinds.add(shard_id.source_kind)
        if shard_id.run_id:
            self.run_ids.add(shard_id.run_id)

    def add_from_path(self, json_path: Path, dataset_hash: str = "") -> None:
        meta = load_shard_meta(json_path)
        shard_id = ShardId.from_meta(meta, json_path, dataset_hash)
        self.add_shard(shard_id)

    def add_from_paths(self, paths: Iterable[Path], dataset_hash: str = "") -> None:
        for entry in paths:
            path = Path(entry)
            if path.is_dir():
                candidates = discover_shard_jsons(path)
            elif path.suffix.lower() == ".json":
                candidates = [path]
            else:
                logger.debug("Ignoring non-metadata path in catalog scan: %s", path)
                continue

            for candidate in candidates:
                try:
                    self.add_from_path(candidate, dataset_hash)
                except Exception as exc:
                    logger.warning(
                        "Failed to load shard metadata %s: %s", candidate, exc
                    )

    def add_from_roots(self, roots: Sequence[Path]) -> None:
        self.add_from_paths(roots)

    def get_canonical_ids(self) -> List[str]:
        return sorted(self.shards.keys())

    def validate_against_used(
        self, used_canonical_ids: Set[str]
    ) -> Dict[str, List[str]]:
        catalog_ids = set(self.shards.keys())
        missing = sorted(set(used_canonical_ids) - catalog_ids)
        extras = sorted(catalog_ids - set(used_canonical_ids))
        warnings: List[str] = []

        if len(self.source_kinds) > 1:
            warnings.append(
                "Mixed shard kinds detected; expected a single DEMUX source."
            )

        warnings.extend(self._analyze_temperature_distribution())

        return {
            "missing": missing,
            "extras": extras,
            "warnings": warnings,
        }

    def get_shard_info_table(self) -> List[Dict[str, str]]:
        rows: List[Dict[str, str]] = []
        for canonical, shard in self.shards.items():
            rows.append(
                {
                    "canonical_id": canonical,
                    "shard_id": shard.shard_id,
                    "temperature_K": f"{shard.temperature_K:.3f}",
                    "replica_id": str(shard.replica_index),
                    "segment_id": str(shard.segment_id),
                    "run_id": shard.run_id,
                    "source_kind": shard.source_kind,
                    "path": str(shard.json_path),
                }
            )
        return sorted(rows, key=lambda x: x["canonical_id"])

    def _analyze_temperature_distribution(self) -> List[str]:
        warnings: List[str] = []
        temps = sorted({shard.temperature_K for shard in self.shards.values()})
        if not temps:
            return warnings

        # Simple check for missing temperatures assuming equal spacing
        if len(temps) > 1:
            spacing = temps[1] - temps[0]
            expected = {temps[0] + i * spacing for i in range(len(temps))}
            missing = expected - set(temps)
            if missing:
                warnings.append(
                    "Missing temperatures detected: "
                    + ", ".join(f"{t:.1f}" for t in sorted(missing))
                )
        return warnings


def build_catalog_from_paths(
    source_paths: Iterable[Path], dataset_hash: str = ""
) -> ShardCatalog:
    catalog = ShardCatalog()
    catalog.add_from_paths(source_paths, dataset_hash)
    return catalog


def validate_shard_usage(
    available_paths: Iterable[Path],
    used_canonical_ids: Set[str],
    dataset_hash: str = "",
) -> Dict[str, List[str]]:
    catalog = build_catalog_from_paths(available_paths, dataset_hash)
    return catalog.validate_against_used(used_canonical_ids)
