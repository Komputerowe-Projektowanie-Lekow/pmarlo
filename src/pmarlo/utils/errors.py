"""Project-specific exception hierarchy for demux and streaming I/O."""

from __future__ import annotations


class DemuxError(Exception):
    """Base class for demux-related errors."""


class DemuxIntegrityError(DemuxError):
    """Logical inconsistency in exchange history, segment times, or frames."""


class DemuxIOError(DemuxError):
    """I/O failure when loading or reading trajectories for demux."""


class DemuxPlanError(DemuxError):
    """Invalid or unsatisfiable demux plan inputs or derived plan."""


class DemuxWriterError(DemuxError):
    """Failure to write output trajectory frames or finalize writing."""


class TemperatureConsistencyError(ValueError):
    """Raised when a dataset violates the temperature/demux contract.

    Contract summary:
    - All shards must be demultiplexed (kind == "demux").
    - All shards must share the same temperature_K.
    - Each shard must represent a contiguous time block (non-empty, start<stop).
    """

    pass
