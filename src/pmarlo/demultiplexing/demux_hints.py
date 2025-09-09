from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class DemuxHints:
    """Lightweight guidance for downstream MSM consumers.

    Attributes
    ----------
    contiguous_blocks : list of (start, stop)
        Frame index intervals [start, stop) containing only real (non-filled)
        frames. Downstream can restrict analysis to these ranges.
    fill_policy : str
        Policy used during demux for gap handling.
    repaired_segments : list[int]
        Segments where gaps were filled.
    skipped_segments : list[int]
        Segments omitted from output.
    total_expected_frames : int
        Planned total frames for this demuxed trajectory.
    """

    contiguous_blocks: List[Tuple[int, int]]
    fill_policy: str
    repaired_segments: List[int]
    skipped_segments: List[int]
    total_expected_frames: int


def _to_tuple_blocks(blocks: List[List[int]]) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for b in blocks:
        if not isinstance(b, list) or len(b) != 2:
            continue
        out.append((int(b[0]), int(b[1])))
    return out


def load_demux_hints(meta_path: str | Path | Dict[str, object]) -> DemuxHints:
    """Load demux hints from a metadata JSON path or dict.

    Parameters
    ----------
    meta_path : str | Path | dict
        Path to the metadata JSON or a pre-parsed metadata dictionary.
    """
    if isinstance(meta_path, (str, Path)):
        try:
            d = json.loads(Path(str(meta_path)).read_text())
        except Exception:
            d = {}
    else:
        try:
            d = dict(meta_path)
        except Exception:
            d = {}
    if not isinstance(d, dict):
        d = {}
    blocks = _to_tuple_blocks(d.get("contiguous_blocks", []) or [])
    total = int(d.get("total_expected_frames", 0) or 0)
    return DemuxHints(
        contiguous_blocks=blocks,
        fill_policy=str(d.get("fill_policy", "unknown")),
        repaired_segments=[int(x) for x in (d.get("repaired_segments", []) or [])],
        skipped_segments=[int(x) for x in (d.get("skipped_segments", []) or [])],
        total_expected_frames=total,
    )
