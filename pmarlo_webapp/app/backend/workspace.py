from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .layout import WorkspaceLayout

logger = logging.getLogger(__name__)


class WorkflowBackend:
    """High-level orchestration helper for the Streamlit UI."""

    def __init__(self, layout: WorkspaceLayout) -> None:
        # Avoid circular imports at module load time by importing lazily.
        from . import Backend as BackendImpl

        self._backend = BackendImpl(layout)
        self.layout = self._backend.layout
        self.state = self._backend.state
        self._migrate_state_paths()

    def run_sampling(self, config: Any) -> Any:
        return self._backend.run_sampling(config)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._backend, name)

    def _migrate_state_paths(self) -> None:
        """Normalize persisted paths so they match the current layout."""
        layout = self.layout
        state = self.state

        if state.workspace_layout is not layout:
            state.workspace_layout = layout

        def transform(value: str) -> str:
            if not value:
                return value
            try:
                candidate = Path(value)
            except Exception:
                return value
            if not candidate.is_absolute():
                return value
            rebased = layout.rebase_legacy_path(candidate)
            return str(rebased)

        try:
            state.normalize_strings(transform)
        except Exception as exc:
            logger.debug("Failed to normalize state paths: %s", exc)
