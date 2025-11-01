import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

from .types import *
from .layout import WorkspaceLayout
from .state import PersistentState
from .sampling import SamplingMixin
from .shards import ShardsMixin
from .training import TrainingMixin
from .analysis import AnalysisMixin
from .conformations import ConformationsMixin
from .utils import *

logger = logging.getLogger(__name__)


class Backend(
    SamplingMixin,
    ShardsMixin,
    TrainingMixin,
    AnalysisMixin,
    ConformationsMixin
):
    """Main backend interface for pmarlo-webapp."""

    def __init__(self, layout: WorkspaceLayout):
        """Initialize with a WorkspaceLayout object."""
        self.layout = layout
        self.state = PersistentState(self.layout.state_path)

    def _path_from_value(self, value: Any) -> Optional[Path]:
        """Convert a value to a Path, handling various input types."""
        if isinstance(value, Path):
            return value
        if isinstance(value, str):
            return Path(value)
        return None

    def sidebar_summary(self) -> Dict[str, int]:
        """Return summary counts for the sidebar display."""
        return {
            "runs": len(self.state.runs),
            "shards": len(self.state.shards),
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
]
