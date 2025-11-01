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

    def __init__(self, workspace: Path):
        self.layout = WorkspaceLayout(workspace)
        manifest = self.layout.workspace_dir / "manifest.json"
        self.state = PersistentState(manifest)

    def _path_from_value(self, value: Any) -> Optional[Path]:
        """Convert a value to a Path, handling various input types."""
        if isinstance(value, Path):
            return value
        if isinstance(value, str):
            return Path(value)
        return None


# Public exports
__all__ = [
    "Backend",
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
