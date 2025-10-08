from __future__ import annotations

"""Deterministic shard selection and loading helpers."""

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from pmarlo import constants as const

from .format import read_shard_npz_json
from .meta import load_shard_meta
from .schema import Shard

__all__ = ["select_shards", "load_shards", "group_by_temperature"]


def select_shards(root: Path, *, temperature_K: Optional[float] = None) -> List[Path]:
    """Return sorted JSON shard paths filtered by temperature."""

    root = Path(root)
    jsons = sorted(root.rglob("*.json"))
    if temperature_K is None:
        return jsons
    out: List[Path] = []
    for json_path in jsons:
        meta = load_shard_meta(json_path)
        if abs(meta.temperature_K - float(temperature_K)) < const.NUMERIC_ABSOLUTE_TOLERANCE:
            out.append(json_path)
    return out


def load_shards(json_paths: Sequence[Path]) -> List[Shard]:
    """Load shards from JSON paths ensuring DEMUX uniqueness."""

    shards: List[Shard] = []
    seen: Dict[tuple, Path] = {}
    for json_path in json_paths:
        npz_path = Path(json_path).with_suffix(".npz")
        shard = read_shard_npz_json(npz_path, json_path)
        key = (
            int(shard.meta.replica_id),
            int(shard.meta.segment_id),
            round(shard.meta.temperature_K, 3),
        )
        if key in seen:
            raise ValueError(
                "Duplicate DEMUX shard detected for replica/segment/temperature: "
                f"{key} (existing={seen[key]}, new={json_path})"
            )
        seen[key] = Path(json_path)
        shards.append(shard)
    return shards


def group_by_temperature(shards: Iterable[Shard]) -> Dict[float, List[Shard]]:
    grouped: Dict[float, List[Shard]] = {}
    for shard in shards:
        grouped.setdefault(shard.meta.temperature_K, []).append(shard)
    return grouped
