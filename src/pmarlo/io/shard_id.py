from __future__ import annotations

"""Lightweight compatibility shims around the new shards metadata APIs."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import warnings

from pmarlo.shards.id import canonical_shard_id
from pmarlo.shards.meta import load_shard_meta
from pmarlo.shards.schema import ShardMeta

__all__ = [
    "ShardId",
    "parse_shard_id",
]


@dataclass(frozen=True)
class ShardId:
    """Compatibility wrapper exposing legacy fields derived from ``ShardMeta``."""

    meta: ShardMeta
    json_path: Path
    dataset_hash: str = ""

    def canonical(self) -> str:
        return canonical_shard_id(self.meta)

    @property
    def shard_id(self) -> str:
        return self.meta.shard_id

    @property
    def temperature_K(self) -> float:
        return float(self.meta.temperature_K)

    @property
    def replica_index(self) -> int:
        return int(self.meta.replica_id)

    @property
    def segment_id(self) -> int:
        return int(self.meta.segment_id)

    @property
    def exchange_window_id(self) -> int:
        return int(self.meta.exchange_window_id)

    @property
    def run_id(self) -> str:
        provenance = self.meta.provenance or {}
        return str(
            provenance.get("run_id")
            or provenance.get("run_uid")
            or provenance.get("run")
            or ""
        )

    @property
    def source_kind(self) -> str:
        provenance = self.meta.provenance or {}
        return str(provenance.get("kind") or provenance.get("source_kind") or "demux")

    @property
    def local_index(self) -> int:
        """Provide backwards-compatible index (maps to ``segment_id``)."""

        return self.segment_id

    @property
    def source_path(self) -> Optional[Path]:
        provenance = self.meta.provenance or {}
        src = provenance.get("trajectory") or provenance.get("source_path")
        if not src:
            return None
        try:
            return Path(str(src))
        except Exception:
            return None

    @classmethod
    def from_meta(
        cls, meta: ShardMeta, json_path: Path, dataset_hash: str = ""
    ) -> "ShardId":
        return cls(meta=meta, json_path=Path(json_path), dataset_hash=dataset_hash)


def parse_shard_id(
    path: Path | str,
    dataset_hash: str = "",
    *,
    require_exists: bool = True,
) -> ShardId:
    """Deprecated shim that now expects a shard JSON metadata path."""

    warnings.warn(
        "parse_shard_id is deprecated; load metadata via pmarlo.shards.meta and "
        "use canonical_shard_id instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    json_path = Path(path)
    if require_exists and not json_path.exists():
        raise FileNotFoundError(f"Shard metadata not found: {json_path}")
    if json_path.suffix.lower() != ".json":
        raise ValueError("parse_shard_id now expects a shard JSON metadata path.")

    meta = load_shard_meta(json_path)
    return ShardId.from_meta(meta, json_path, dataset_hash)
