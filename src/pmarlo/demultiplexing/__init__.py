"""
Demultiplexing (DEMUX) utilities and streaming engine.

This package contains the standalone demultiplexing planning and execution
logic, decoupled from the REMD implementation for clarity and reuse.
"""

from .demux import demux_trajectories
from .demux_engine import DemuxResult, demux_streaming
from .demux_hints import DemuxHints, load_demux_hints
from .demux_metadata import DemuxMetadata, serialize_metadata
from .demux_plan import DemuxPlan, DemuxSegmentPlan, build_demux_plan

__all__ = [
    "demux_trajectories",
    "DemuxResult",
    "demux_streaming",
    "DemuxHints",
    "load_demux_hints",
    "DemuxMetadata",
    "serialize_metadata",
    "DemuxPlan",
    "DemuxSegmentPlan",
    "build_demux_plan",
]


