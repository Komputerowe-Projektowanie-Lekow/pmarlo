from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backend import Backend
    from backend import WorkspaceLayout


@dataclass
class AppContext:
    """Shared context passed to all tabs."""
    backend: 'Backend'
    layout: 'WorkspaceLayout'


def build_context() -> AppContext:
    """Initialize workspace and backend."""
    from backend import WorkspaceLayout, Backend
    from core.logging import configure_file_logging

    configure_file_logging()
    layout = WorkspaceLayout.from_app_package()
    backend = Backend(layout)
    return AppContext(backend=backend, layout=layout)
