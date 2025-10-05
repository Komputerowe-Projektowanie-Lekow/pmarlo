"""
Demultiplexing (DEMUX) utilities and streaming engine.

This package contains the standalone demultiplexing planning and execution
logic, decoupled from the REMD implementation for clarity and reuse.
"""

from importlib import import_module
from typing import Any, Dict, Tuple

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

_EXPORTS: Dict[str, Tuple[str, str]] = {
    "demux_trajectories": ("pmarlo.demultiplexing.demux", "demux_trajectories"),
    "DemuxResult": ("pmarlo.demultiplexing.demux_engine", "DemuxResult"),
    "demux_streaming": ("pmarlo.demultiplexing.demux_engine", "demux_streaming"),
    "DemuxHints": ("pmarlo.demultiplexing.demux_hints", "DemuxHints"),
    "load_demux_hints": ("pmarlo.demultiplexing.demux_hints", "load_demux_hints"),
    "DemuxMetadata": ("pmarlo.demultiplexing.demux_metadata", "DemuxMetadata"),
    "serialize_metadata": (
        "pmarlo.demultiplexing.demux_metadata",
        "serialize_metadata",
    ),
    "DemuxPlan": ("pmarlo.demultiplexing.demux_plan", "DemuxPlan"),
    "DemuxSegmentPlan": ("pmarlo.demultiplexing.demux_plan", "DemuxSegmentPlan"),
    "build_demux_plan": ("pmarlo.demultiplexing.demux_plan", "build_demux_plan"),
}


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(__all__))
