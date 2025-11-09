

class WorkflowBackend:
    """High-level orchestration for the Streamlit UI."""

    def __init__(self, layout: WorkspaceLayout) -> None:
        self.layout = layout
        self.state = StateManager(
            layout.state_path,
            workspace_layout=self.layout,
        )
        self._migrate_state_paths()
