from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.backend import WorkflowBackend, WorkspaceLayout


@dataclass
class AppContext:
    """Shared context passed to all tabs."""
    backend: 'WorkflowBackend'
    layout: 'WorkspaceLayout'


def build_context() -> AppContext:
    """Initialize workspace and backend."""
    from app.backend import WorkspaceLayout, WorkflowBackend
    from app.core.logging import configure_file_logging

    configure_file_logging()
    layout = WorkspaceLayout.from_app_package()
    backend = WorkflowBackend(layout)
    return AppContext(backend=backend, layout=layout)
