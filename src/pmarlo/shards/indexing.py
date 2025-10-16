from __future__ import annotations

"""Helpers for discovering shard indices on disk."""

from dataclasses import dataclass
from pathlib import Path

__all__ = ["ShardIndexState", "initialise_shard_indices"]


@dataclass(frozen=True)
class ShardIndexState:
    """Container describing the next shard index and seed bookkeeping."""

    next_index: int
    start_index: int
    seed_base: int

    def seed_for(self, shard_index: int) -> int:
        """Return the RNG seed for a specific shard index."""

        delta = int(shard_index) - int(self.start_index)
        if delta < 0:
            raise ValueError(
                f"shard_index {shard_index} precedes starting index {self.start_index}"
            )
        return int(self.seed_base) + delta


def _collect_existing_indices(out_dir: Path) -> list[int]:
    """Scan an output directory and return sorted shard indices."""

    indices: list[int] = []
    for shard_file in sorted(Path(out_dir).glob("shard_*.json")):
        stem = shard_file.stem
        try:
            indices.append(int(stem.split("_")[1]))
        except (IndexError, ValueError):
            continue
    return indices


def initialise_shard_indices(
    out_dir: Path,
    seed_start: int | float = 0,
) -> ShardIndexState:
    """Determine the next shard index and its seed base for an output directory."""

    existing = _collect_existing_indices(out_dir)
    next_index = max(existing) + 1 if existing else 0
    start_index = next_index
    seed_base = int(seed_start) + start_index
    return ShardIndexState(
        next_index=int(next_index),
        start_index=int(start_index),
        seed_base=int(seed_base),
    )
