"""Metadata for demultiplexed REMD trajectories.

Backward-compatible schema enriched with additional provenance in v2.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, TypedDict, cast

from typing_extensions import NotRequired

from pmarlo.utils.errors import DemuxIntegrityError as _DemuxIntegrityError

logger = logging.getLogger("pmarlo")

# Backward-compatibility: alias project-specific error in this module
DemuxIntegrityError = _DemuxIntegrityError

if TYPE_CHECKING:  # pragma: no cover - typing-only imports
    from .demux_engine import DemuxResult
    from .demux_plan import DemuxPlan


class DemuxMetadataDict(TypedDict, total=False):
    exchange_frequency_steps: int
    integration_timestep_ps: float
    frames_per_segment: int
    temperature_schedule: Dict[str, Dict[str, float]]
    segment_count: int
    repaired_segments: List[int]
    skipped_segments: List[int]
    fill_policy: str
    time_per_frame_ps: Optional[float]
    source_files_checksum: Dict[str, str]
    plan_checksum: Optional[str]
    schema_version: int
    total_expected_frames: Optional[int]
    contiguous_blocks: List[List[int]]
    warnings: NotRequired[List[str]]
    equilibration_steps_total: int
    overlap_corrections: List[int]


@dataclass
class DemuxMetadata:
    """Container for provenance of a demultiplexed trajectory.

    Attributes:
        exchange_frequency_steps: MD steps between exchange attempts.
        integration_timestep_ps: Integration timestep in picoseconds.
        frames_per_segment: Number of frames originating from each REMD segment.
        temperature_schedule: Mapping of replica id and segment to temperature.
            Keys are stringified replica indices mapping to dictionaries whose keys
            are segment indices and values are temperatures in Kelvin.
    """

    # --- Core fields (v1) ---
    exchange_frequency_steps: int
    integration_timestep_ps: float
    frames_per_segment: int
    temperature_schedule: Dict[str, Dict[str, float]]

    # --- Extended fields (v2) ---
    segment_count: int = 0
    repaired_segments: List[int] = field(default_factory=list)
    skipped_segments: List[int] = field(default_factory=list)
    fill_policy: Optional[str] = None
    time_per_frame_ps: Optional[float] = None
    source_files_checksum: Dict[str, str] = field(default_factory=dict)
    plan_checksum: Optional[str] = None
    schema_version: int = 2
    total_expected_frames: Optional[int] = None
    contiguous_blocks: List[List[int]] = field(default_factory=list)
    equilibration_steps_total: int = 0
    overlap_corrections: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to a JSON-serialisable dictionary (schema v2)."""
        d = asdict(self)
        # Ensure proper scalar types
        d["exchange_frequency_steps"] = int(self.exchange_frequency_steps)
        d["integration_timestep_ps"] = float(self.integration_timestep_ps)
        d["frames_per_segment"] = int(self.frames_per_segment)
        d["schema_version"] = int(getattr(self, "schema_version", 2))
        d["equilibration_steps_total"] = int(
            getattr(self, "equilibration_steps_total", 0)
        )
        # temperature_schedule keys are already strings; make sure mapping types are basic
        d["temperature_schedule"] = {
            str(rep): {str(seg): float(temp) for seg, temp in segs.items()}
            for rep, segs in self.temperature_schedule.items()
        }
        return d

    def to_json(self, path: Path) -> None:
        """Serialize metadata to ``path`` as JSON (with schema_version)."""
        path.write_text(json.dumps(self.to_dict(), indent=2))
        logger.debug(f"Demux metadata written to {path}")

    @classmethod
    def from_dict(cls, data: Dict[str, Any] | Any) -> "DemuxMetadata":
        """Create :class:`DemuxMetadata` from a dictionary (v1/v2 tolerant)."""
        if not isinstance(data, dict):
            data = {}
        schedule = {
            str(replica): {str(seg): float(temp) for seg, temp in segments.items()}
            for replica, segments in data.get("temperature_schedule", {}).items()
        }
        # Core required fields
        obj = cls(
            exchange_frequency_steps=int(data.get("exchange_frequency_steps", 0)),
            integration_timestep_ps=float(data.get("integration_timestep_ps", 0.0)),
            frames_per_segment=int(data.get("frames_per_segment", 0)),
            temperature_schedule=schedule,
        )
        # Optional v2 fields
        obj.segment_count = int(
            data.get("segment_count", getattr(obj, "segment_count", 0))
        )
        obj.repaired_segments = [int(x) for x in data.get("repaired_segments", [])]
        obj.skipped_segments = [int(x) for x in data.get("skipped_segments", [])]
        obj.fill_policy = data.get("fill_policy", None)
        tpf = data.get("time_per_frame_ps", None)
        try:
            obj.time_per_frame_ps = float(tpf) if (tpf is not None) else None
        except Exception:
            obj.time_per_frame_ps = None
        obj.source_files_checksum = {
            str(k): str(v) for k, v in data.get("source_files_checksum", {}).items()
        }
        obj.plan_checksum = data.get("plan_checksum")
        obj.schema_version = int(data.get("schema_version", 2))
        try:
            obj.equilibration_steps_total = int(
                data.get("equilibration_steps_total", 0)
            )
        except Exception:
            obj.equilibration_steps_total = 0
        try:
            obj.overlap_corrections = [
                int(x) for x in data.get("overlap_corrections", [])
            ]
        except Exception:
            obj.overlap_corrections = []
        return obj

    @classmethod
    def from_json(cls, path: Path) -> "DemuxMetadata":
        """Load metadata from a JSON file."""
        try:
            raw = json.loads(path.read_text())
        except Exception:
            raw = {}
        return cls.from_dict(raw)


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with path.open("rb") as f:
            while True:
                b = f.read(chunk_size)
                if not b:
                    break
                h.update(b)
        return h.hexdigest()
    except Exception:
        return None


def _checksum_sources(plan: Any) -> Dict[str, str]:
    checksums: Dict[str, str] = {}
    seen: set[str] = set()
    for seg in getattr(plan, "segments", []) or []:
        p = getattr(seg, "source_path", "")
        if not p or p in seen:
            continue
        seen.add(p)
        digest = _sha256_file(Path(p))
        if digest:
            checksums[p] = digest
    return checksums


def _checksum_plan(plan: Any) -> str:
    # Create a stable JSON representation of the plan content
    payload = {
        "target_temperature": getattr(plan, "target_temperature", None),
        "frames_per_segment": getattr(plan, "frames_per_segment", None),
        "total_expected_frames": getattr(plan, "total_expected_frames", None),
        "segments": [
            {
                "segment_index": getattr(s, "segment_index", None),
                "replica_index": getattr(s, "replica_index", None),
                "source_path": getattr(s, "source_path", None),
                "start_frame": getattr(s, "start_frame", None),
                "stop_frame": getattr(s, "stop_frame", None),
                "expected_frames": getattr(s, "expected_frames", None),
                "needs_fill": getattr(s, "needs_fill", None),
            }
            for s in getattr(plan, "segments", []) or []
        ],
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def _compute_contiguous_blocks(result: Any, plan: Any) -> List[List[int]]:
    """Compute [start, stop) intervals of real (non-filled) frames.

    Parameters
    ----------
    result : DemuxResult-like
        Object with attribute ``segment_real_frames`` giving count of real
        frames per segment in order.
    plan : DemuxPlan-like
        Object with attribute ``segments`` whose items each expose
        ``expected_frames``.

    Returns
    -------
    list of list of int
        Sorted, non-overlapping intervals ``[start, stop)`` along the demuxed
        output timeline where every frame is from original data (no fill).
    """
    segs = getattr(plan, "segments", []) or []
    real_counts = list(getattr(result, "segment_real_frames", []))
    if not segs or not real_counts or len(real_counts) != len(segs):
        return []
    blocks: List[List[int]] = []
    pos = 0
    block_start: Optional[int] = None
    repaired_set = set(getattr(result, "repaired_segments", []) or [])
    for i, seg in enumerate(segs):
        expected = int(getattr(seg, "expected_frames", 0) or 0)
        real = int(real_counts[i])
        is_repaired = (i in repaired_set) or (real < expected)
        if expected <= 0:
            # no output frames; close current block if any
            if block_start is not None:
                blocks.append([int(block_start), int(pos)])
                block_start = None
            continue
        if is_repaired:
            # close current block; exclude repaired segment entirely
            if block_start is not None:
                blocks.append([int(block_start), int(pos)])
                block_start = None
        else:
            if block_start is None:
                block_start = pos
        pos += expected
    if block_start is not None:
        blocks.append([int(block_start), int(pos)])
    return blocks


def serialize_metadata(
    result: "DemuxResult" | Any,  # fall back to Any at API boundary
    plan: "DemuxPlan" | Any,
    runtime_info: Mapping[str, Any],
) -> DemuxMetadataDict:
    """Build a v2 metadata dictionary from demux result and plan.

    Parameters
    ----------
    result : DemuxResult-like
        Object with attributes: ``total_frames_written``, ``repaired_segments``,
        ``skipped_segments``, optional ``warnings``, and
        ``segment_real_frames`` (list of per-segment real counts).
    plan : DemuxPlan-like
        Object exposing ``segments``, ``target_temperature``,
        ``frames_per_segment``, and ``total_expected_frames``.
    runtime_info : mapping
        Auxiliary information; expected keys include
        ``exchange_frequency_steps`` (int), ``integration_timestep_ps`` (float),
        ``fill_policy`` (str), and optionally ``temperature_schedule`` (mapping).

    Returns
    -------
    dict
        JSON-serialisable dictionary with schema_version=2 and extended
        provenance.
    """

    exchange_freq = int(runtime_info.get("exchange_frequency_steps", 0))
    timestep_ps = float(runtime_info.get("integration_timestep_ps", 0.0))
    frames_per_seg = int(
        runtime_info.get(
            "frames_per_segment", getattr(plan, "frames_per_segment", 0) or 0
        )
    )
    time_per_frame_ps = None
    if frames_per_seg > 0 and exchange_freq > 0 and timestep_ps > 0.0:
        time_per_frame_ps = timestep_ps * (exchange_freq / float(frames_per_seg))

    meta = DemuxMetadata(
        exchange_frequency_steps=exchange_freq,
        integration_timestep_ps=timestep_ps,
        frames_per_segment=frames_per_seg,
        temperature_schedule=dict(runtime_info.get("temperature_schedule", {})),
        segment_count=int(len(getattr(plan, "segments", []) or [])),
        repaired_segments=[int(x) for x in getattr(result, "repaired_segments", [])],
        skipped_segments=[int(x) for x in getattr(result, "skipped_segments", [])],
        fill_policy=str(runtime_info.get("fill_policy", "unknown")),
        time_per_frame_ps=time_per_frame_ps,
        source_files_checksum=_checksum_sources(plan),
        plan_checksum=_checksum_plan(plan),
        schema_version=2,
        total_expected_frames=int(getattr(plan, "total_expected_frames", 0) or 0),
        contiguous_blocks=_compute_contiguous_blocks(result, plan),
        equilibration_steps_total=int(
            runtime_info.get("equilibration_steps_total", 0) or 0
        ),
        overlap_corrections=[
            int(x) for x in (runtime_info.get("overlap_corrections", []) or [])
        ],
    )
    metadata_dict = cast(DemuxMetadataDict, meta.to_dict())
    # Provide a readable summary of warnings if present (non-breaking extra key)
    warnings = list(getattr(result, "warnings", []))
    if warnings:
        metadata_dict["warnings"] = [str(w) for w in warnings]
    # If time_per_frame_ps is None, omit the key to keep round-trips tidy
    if metadata_dict.get("time_per_frame_ps") is None:
        metadata_dict.pop("time_per_frame_ps", None)
    return metadata_dict
