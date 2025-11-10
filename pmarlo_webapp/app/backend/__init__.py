import logging
import os
from pathlib import Path, PureWindowsPath
from typing import Optional, List, Dict, Any

import mdtraj as md
from pmarlo.conformations import find_conformations

from pmarlo.data.aggregate import load_shards_as_dataset
from pmarlo.markov_state_model.reduction import reduce_features
from pmarlo.markov_state_model.clustering import cluster_microstates
from pmarlo.conformations.visualizations import plot_tpt_summary
from pmarlo.markov_state_model._msm_utils import build_simple_msm

from .types import *
from .layout import WorkspaceLayout
from .state import PersistentState
from .sampling import SamplingMixin, run_short_sim
from .shards import ShardsMixin
from .training import TrainingMixin
from .analysis import AnalysisMixin
from .conformations import ConformationsMixin
from .validation import ValidationMixin
from .utils import *
from .utils import _timestamp as _timestamp

logger = logging.getLogger(__name__)


class Backend(
    SamplingMixin,
    ShardsMixin,
    TrainingMixin,
    AnalysisMixin,
    ConformationsMixin,
    ValidationMixin
):
    """Main backend interface for pmarlo-webapp."""

    def __init__(self, layout: WorkspaceLayout):
        """Initialize with a WorkspaceLayout object."""
        self.layout = layout
        self.state = PersistentState(
            self.layout.state_path,
            workspace_layout=self.layout,
        )

    def _path_from_value(self, value: Any) -> Optional[Path]:
        """Convert a value to a Path, handling various input types."""
        if isinstance(value, Path):
            return value
        if isinstance(value, str):
            raw = value.strip()
            if not raw:
                return None
            try:
                candidate = Path(raw)
            except Exception:
                candidate = None
            if candidate is not None and candidate.exists():
                return candidate

            if os.name != "nt" and len(raw) >= 3 and raw[1] == ":" and raw[2] in ("\\", "/"):
                win = PureWindowsPath(raw)
                drive = win.drive.rstrip(":").lower()
                parts = list(win.parts[1:])
                converted = Path("/mnt", drive, *parts)
                if converted.exists():
                    return converted
                return converted

            return candidate if candidate is not None else Path(raw)
        return None

    def sidebar_summary(self) -> Dict[str, int]:
        """Return summary counts for the sidebar display."""
        # Count total individual shard files across all batches
        total_shards = 0
        for entry in self.state.shards:
            paths = entry.get("paths", [])
            total_shards += len(paths)

        return {
            "runs": len(self.state.runs),
            "shards": total_shards,  # Total individual shard files, not batches
            "models": len(self.state.models),
            "builds": len(self.state.builds),
            "conformations": len(self.state.conformations),
        }


# Backward compatibility alias
WorkflowBackend = Backend


# Public exports
__all__ = [
    "Backend",
    "WorkflowBackend",
    "WorkspaceLayout",
    "PersistentState",
    "SimulationResult",
    "ShardRequest",
    "ShardResult",
    "TrainingConfig",
    "TrainingResult",
    "BuildConfig",
    "BuildArtifact",
    "ConformationsConfig",
    "ConformationsResult",
    "load_shards_as_dataset",
    "reduce_features",
    "cluster_microstates",
    "md",
    "plot_tpt_summary",
    "build_simple_msm",
    "find_conformations",
    "run_short_sim",
]
