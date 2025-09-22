from __future__ import annotations

"""Shard catalog utilities backed by strict shard metadata."""

import logging
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set

from pmarlo.shards.discover import discover_shard_jsons

from .shard_id import ShardId, parse_shard_id

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
        existing_path = existing.json_path or existing.source_path if existing else None
        new_path = shard_id.json_path or shard_id.source_path
        if existing is not None and existing_path != new_path:
            raise ValueError(
                "Canonical ID collision: "
                f"{canonical} already mapped to {existing_path}, got {new_path}"
            )

        self.shards[canonical] = shard_id
        if shard_id.source_kind:
            self.source_kinds.add(shard_id.source_kind)
        if shard_id.run_id:
            self.run_ids.add(shard_id.run_id)

    def add_from_path(self, json_path: Path, dataset_hash: str = "") -> None:
        shard_id = parse_shard_id(json_path, dataset_hash=dataset_hash)
        self.add_shard(shard_id)

    def add_from_paths(self, paths: Iterable[Path], dataset_hash: str = "") -> None:
        for entry in paths:
            path = Path(entry)
            if path.is_dir():
                candidates = list(discover_shard_jsons(path))
                for pattern in ("*.dcd", "*.xtc", "*.nc"):
                    candidates.extend(sorted(path.rglob(pattern)))
            else:
                candidates = [path]

            for candidate in candidates:
                try:
                    shard = parse_shard_id(candidate, dataset_hash=dataset_hash)
                    self.add_shard(shard)
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
        missing = sorted(catalog_ids - set(used_canonical_ids))
        extras = sorted(set(used_canonical_ids) - catalog_ids)
        warnings: List[str] = []

        if len(self.source_kinds) > 1:
            warnings.append(
                "Mixed source kinds detected; expected a single DEMUX source."
            )

        if len(self.run_ids) > 1:
            warnings.append(
                "Multiple runs detected: " + ", ".join(sorted(self.run_ids))
            )

        warnings.extend(self._analyze_temperature_distribution())
        warnings.extend(self._check_replica_contiguity())

        return {
            "missing": missing,
            "extra": extras,
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
                    "temperature_K": (
                        f"{float(shard.temperature_K):.3f}"
                        if shard.temperature_K is not None
                        else ""
                    ),
                    "replica_id": ""
                    if shard.replica_index is None
                    else str(shard.replica_index),
                    "segment_id": str(shard.local_index),
                    "run_id": shard.run_id,
                    "source_kind": shard.source_kind,
                    "path": str(shard.json_path or shard.source_path or ""),
                }
            )
        return sorted(rows, key=lambda x: x["canonical_id"])

    def _analyze_temperature_distribution(self) -> List[str]:
        warnings: List[str] = []
        temps = sorted(
            {
                int(round(float(shard.temperature_K)))
                for shard in self.shards.values()
                if shard.source_kind == "demux" and shard.temperature_K is not None
            }
        )
        if not temps:
            return warnings

        # Simple check for missing temperatures assuming equal spacing
        if len(temps) > 1:
            diffs = [temps[i + 1] - temps[i] for i in range(len(temps) - 1)]
            diffs = [d for d in diffs if d > 0]
            base_step = min(diffs) if diffs else None
            if diffs:
                gcd_step = diffs[0]
                for diff in diffs[1:]:
                    gcd_step = math.gcd(gcd_step, diff)
                if gcd_step > 0:
                    base_step = gcd_step
            if base_step is None:
                base_step = 50
            elif base_step > 50 and base_step % 50 == 0:
                base_step = 50

            total_steps = ((temps[-1] - temps[0]) // base_step) + 1
            expected = {temps[0] + i * base_step for i in range(total_steps)}
            missing = expected - set(temps)
            if missing:
                warnings.append(
                    "Missing temperatures detected: "
                    + ", ".join(str(t) for t in sorted(missing))
                )
        return warnings

    def _check_replica_contiguity(self) -> List[str]:
        warnings: List[str] = []
        by_run: Dict[str, List[int]] = {}
        for shard in self.shards.values():
            if shard.source_kind != "replica" or shard.replica_index is None:
                continue
            by_run.setdefault(shard.run_id, []).append(int(shard.replica_index))

        for run_id, replicas in by_run.items():
            sorted_replicas = sorted(set(replicas))
            expected = list(range(sorted_replicas[0], sorted_replicas[-1] + 1))
            if sorted_replicas != expected:
                missing = sorted(set(expected) - set(sorted_replicas))
                warnings.append(
                    f"Replica indices not contiguous for run {run_id}: missing {missing}"
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
