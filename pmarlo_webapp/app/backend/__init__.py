from pathlib import Path
from typing import Optional, List, Dict, Any

from .types import *
from .layout import WorkspaceLayout
from .state import PersistentState
from .simulation import SimulationMixin
from .shards import ShardsMixin
from .training import TrainingMixin
from .analysis import AnalysisMixin
from .conformations import ConformationsMixin
from .utils import *
from .state import StateManager

__all__ = [
    "WorkspaceLayout",
    "SimulationConfig",
    "SimulationResult",
    "ShardRequest",
    "ShardResult",
    "TrainingConfig",
    "TrainingResult",
    "BuildConfig",
    "BuildArtifact",
    "ConformationsConfig",
    "ConformationsResult",
    "WorkflowBackend",
    "choose_sim_seed",
    "run_short_sim",
    "calculate_its",
    "plot_its",
]

logger = logging.getLogger(__name__)

class Backend(
    SimulationMixin,
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

    # ... your implementation (used by multiple mixins)

    # UI helper methods
    def latest_model_path(self) -> Optional[Path]:

    # ... your implementation

    def list_models(self) -> List[Dict[str, Any]]:

    # ... your implementation

    def list_builds(self) -> List[Dict[str, Any]]:

    # ... your implementation

    def list_conformations(self) -> List[Dict[str, Any]]:

    # ... your implementation

    def sidebar_summary(self) -> Dict[str, int]:

    # ... your implementation

    # Config builders (used by UI)
    def build_config_from_entry(self, entry: Dict[str, Any]) -> BuildConfig:

    # ... your implementation

    def training_config_from_entry(self, entry: Dict[str, Any]) -> TrainingConfig:

    # ... your implementation

    @staticmethod
    def _coerce_deeptica_params(raw: Any) -> Optional[Dict[str, Any]]:

    # ... your implementation

    @staticmethod
    def _coerce_hidden_layers(raw: Any) -> tuple[int, ...]:

    # ... your implementation

    @staticmethod
    def _coerce_tau_schedule(raw: Any) -> tuple[int, ...]:

    # ... your implementation

    @staticmethod
    def _load_build_result_from_path(path: Path) -> Optional["_BuildResult"]:


# ... your implementation

# Public exports
__all__ = [
    "Backend",
    "WorkspaceLayout",
    "PersistentState",
    "SimulationResult",
    "TrainingConfig",
    "TrainingResult",
    "BuildConfig",
    "BuildArtifact",
    "ConformationsConfig",
    "ConformationsResult",
]
