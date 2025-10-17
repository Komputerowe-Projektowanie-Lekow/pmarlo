"""Compatibility-friendly shard identifier utilities."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from pmarlo.shards.meta import load_shard_meta
from pmarlo.shards.schema import ShardMeta

__all__ = ["ShardId", "parse_shard_id"]


_DEMUX_PATTERN = re.compile(r"demux_T(?P<temp>\d+)(?:K)?", re.IGNORECASE)
_REPLICA_PATTERN = re.compile(r"replica_(?P<rep>\d+)", re.IGNORECASE)


@dataclass(frozen=True)
class ShardId:
    """Lightweight shard identifier mirroring the previous API surface."""

    run_id: str
    source_kind: str
    temperature_K: Optional[float] = None
    replica_index: Optional[int] = None
    local_index: int = 0
    source_path: Optional[Path] = None
    dataset_hash: str = ""
    meta: Optional[ShardMeta] = None
    json_path: Optional[Path] = None

    def canonical(self) -> str:
        """Return the canonical identifier used throughout workflow tests."""

        payload: str
        if self.source_kind == "demux":
            if self.temperature_K is None:
                raise ValueError("demux shards require temperature_K")
            payload = f"T{int(round(self.temperature_K))}"
        elif self.source_kind == "replica":
            if self.replica_index is None:
                raise ValueError("replica shards require replica_index")
            payload = f"R{int(self.replica_index)}"
        else:
            raise ValueError(f"Unsupported source_kind: {self.source_kind}")

        return f"{self.run_id}:{self.source_kind}:{payload}:{int(self.local_index)}"

    @property
    def shard_id(self) -> str:
        """Expose the shard identifier expected by downstream code."""

        if self.meta is not None:
            return str(self.meta.shard_id)
        return self.canonical()

    @property
    def segment_id(self) -> int:
        """Backwards-compatible alias for the local segment index."""

        return int(self.local_index)

    @classmethod
    def from_meta(
        cls, meta: ShardMeta, json_path: Path, dataset_hash: str = ""
    ) -> "ShardId":
        """Build a :class:`ShardId` instance from strict shard metadata."""

        provenance = meta.provenance or {}
        run_id = str(
            provenance.get("run_id")
            or provenance.get("run_uid")
            or provenance.get("run")
            or ""
        )
        source_kind = str(
            provenance.get("kind") or provenance.get("source_kind") or "demux"
        )

        temperature = float(meta.temperature_K) if source_kind == "demux" else None
        replica = int(meta.replica_id) if source_kind == "replica" else None
        local_index = int(meta.segment_id)

        src_path = provenance.get("trajectory") or provenance.get("source_path")
        source_path = Path(src_path) if src_path else None

        return cls(
            run_id=run_id,
            source_kind=source_kind,
            temperature_K=temperature,
            replica_index=replica,
            local_index=local_index,
            source_path=source_path,
            dataset_hash=dataset_hash,
            meta=meta,
            json_path=Path(json_path),
        )

    @classmethod
    def from_canonical(
        cls, canonical: str, source_path: Optional[Path], dataset_hash: str = ""
    ) -> "ShardId":
        """Reconstruct a :class:`ShardId` from its canonical identifier."""

        parts = canonical.split(":")
        if len(parts) != 4:
            raise ValueError("Invalid canonical format")

        run_id, source_kind, payload, local_str = parts
        try:
            local_index = int(local_str)
        except ValueError as exc:  # pragma: no cover - defensive
            raise ValueError("Invalid local index in canonical identifier") from exc

        temperature: Optional[float] = None
        replica: Optional[int] = None

        if source_kind == "demux":
            match = re.fullmatch(r"T(\d+)", payload)
            if not match:
                raise ValueError("Invalid temp/replica format for demux shard")
            temperature = float(match.group(1))
        elif source_kind == "replica":
            match = re.fullmatch(r"R(\d+)", payload)
            if not match:
                raise ValueError("Invalid temp/replica format for replica shard")
            replica = int(match.group(1))
        else:
            raise ValueError("Invalid source_kind in canonical identifier")

        return cls(
            run_id=run_id,
            source_kind=source_kind,
            temperature_K=temperature,
            replica_index=replica,
            local_index=local_index,
            source_path=source_path,
            dataset_hash=dataset_hash,
        )


def parse_shard_id(
    path: Path | str,
    dataset_hash: str = "",
    *,
    require_exists: bool = True,
) -> ShardId:
    """Parse a shard identifier from either metadata or trajectory paths."""

    file_path = Path(path)
    if require_exists and not file_path.exists():
        raise FileNotFoundError(f"Shard path does not exist: {file_path}")

    suffix = file_path.suffix.lower()
    if suffix == ".json":
        meta = load_shard_meta(file_path)
        return ShardId.from_meta(meta, file_path, dataset_hash)

    if suffix not in {".dcd", ".xtc", ".nc"}:
        raise ValueError(f"Unsupported shard file type: {file_path.suffix}")

    run_dir = _find_run_directory(file_path)
    run_id = run_dir.name

    name = file_path.name
    demux_match = _DEMUX_PATTERN.search(name)
    replica_match = _REPLICA_PATTERN.search(name)

    if demux_match:
        temperature = float(demux_match.group("temp"))
        local_index = _local_index_in_group(file_path, run_dir, _DEMUX_PATTERN)
        return ShardId(
            run_id=run_id,
            source_kind="demux",
            temperature_K=temperature,
            replica_index=None,
            local_index=local_index,
            source_path=file_path,
            dataset_hash=dataset_hash,
        )

    if replica_match:
        replica_index = int(replica_match.group("rep"))
        local_index = _local_index_in_group(file_path, run_dir, _REPLICA_PATTERN)
        return ShardId(
            run_id=run_id,
            source_kind="replica",
            temperature_K=None,
            replica_index=replica_index,
            local_index=local_index,
            source_path=file_path,
            dataset_hash=dataset_hash,
        )

    raise ValueError(f"Unrecognised shard filename pattern: {file_path.name}")


def _find_run_directory(path: Path) -> Path:
    """Walk upwards until a ``run-*`` directory is located."""

    for parent in [path.parent, *path.parents]:
        if parent.name.startswith("run-"):
            return parent
    raise ValueError(f"Run directory not found for shard path: {path}")


def _local_index_in_group(path: Path, run_dir: Path, pattern: re.Pattern[str]) -> int:
    """Compute the lexicographic index of ``path`` among matching siblings."""

    target = path.resolve() if path.exists() else path
    siblings = sorted(
        p.resolve() for p in run_dir.iterdir() if p.is_file() and pattern.search(p.name)
    )
    try:
        return siblings.index(target)
    except ValueError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Could not determine local index for {path}") from exc
