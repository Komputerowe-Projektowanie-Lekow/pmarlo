"""Compatibility exports for the representative picker API.

Historically the :class:`RepresentativePicker` lived in ``pmarlo.representative_picker``.
The implementation has since been moved under ``pmarlo.conformations`` but a number
of downstream scripts – and the contract tests in this repository – still expect the
original import path.  Re-exporting the public types from this shim keeps both
styles working without duplicating any logic.
"""

from __future__ import annotations

from pmarlo.conformations.representative_picker import (
    FrameIndexLookup,
    RepresentativeFrame,
    RepresentativePicker,
    TrajectoryFrameLocator,
    TrajectorySegment,
    build_frame_index_lookup,
)

__all__ = [
    "RepresentativePicker",
    "TrajectorySegment",
    "TrajectoryFrameLocator",
    "FrameIndexLookup",
    "RepresentativeFrame",
    "build_frame_index_lookup",
]
