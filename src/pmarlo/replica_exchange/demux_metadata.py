"""Metadata for demultiplexed REMD trajectories."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger("pmarlo")


class DemuxIntegrityError(Exception):
    """Raised when demultiplexing metadata is inconsistent or corrupted."""


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

    exchange_frequency_steps: int
    integration_timestep_ps: float
    frames_per_segment: int
    temperature_schedule: Dict[str, Dict[str, float]]

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to a JSON-serialisable dictionary."""
        return {
            "exchange_frequency_steps": int(self.exchange_frequency_steps),
            "integration_timestep_ps": float(self.integration_timestep_ps),
            "frames_per_segment": int(self.frames_per_segment),
            "temperature_schedule": self.temperature_schedule,
        }

    def to_json(self, path: Path) -> None:
        """Serialize metadata to ``path`` as JSON."""
        path.write_text(json.dumps(self.to_dict(), indent=2))
        logger.debug(f"Demux metadata written to {path}")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DemuxMetadata":
        """Create :class:`DemuxMetadata` from a dictionary."""
        schedule = {
            str(replica): {str(seg): float(temp) for seg, temp in segments.items()}
            for replica, segments in data.get("temperature_schedule", {}).items()
        }
        return cls(
            exchange_frequency_steps=int(data["exchange_frequency_steps"]),
            integration_timestep_ps=float(data["integration_timestep_ps"]),
            frames_per_segment=int(data["frames_per_segment"]),
            temperature_schedule=schedule,
        )

    @classmethod
    def from_json(cls, path: Path) -> "DemuxMetadata":
        """Load metadata from a JSON file."""
        return cls.from_dict(json.loads(path.read_text()))
