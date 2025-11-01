from __future__ import annotations
from pathlib import Path
import os
from dataclasses import dataclass
from pmarlo.utils.path_utils import ensure_directory
from typing import List, Optional, Tuple

def _path_parts_casefold(path: Path) -> Tuple[str, ...]:
    return tuple(os.path.normcase(part) for part in path.parts)

def _rebase_path_with_root(
    path: Path, legacy_root: Path, new_root: Path
) -> Optional[Path]:
    if not path.is_absolute():
        return None
    legacy_root = legacy_root.resolve()
    new_root = new_root.resolve()
    legacy_parts = _path_parts_casefold(legacy_root)
    path_parts = _path_parts_casefold(path.resolve())
    if len(path_parts) < len(legacy_parts):
        return None
    if path_parts[: len(legacy_parts)] != legacy_parts:
        return None
    suffix_parts = path.parts[len(legacy_parts):]
    rebased = new_root.joinpath(*suffix_parts)
    return rebased.resolve()

@dataclass(frozen=True)
class WorkspaceLayout:
    """Resolved paths for the app's workspace tree."""

    app_root: Path
    inputs_dir: Path
    workspace_dir: Path
    sims_dir: Path
    shards_dir: Path
    models_dir: Path
    bundles_dir: Path
    logs_dir: Path
    state_path: Path

    @classmethod
    def from_app_package(cls, file_path: Optional[Path] = None) -> "WorkspaceLayout":
        """Create layout based on the app package location.

        The directory structure should be:
        pmarlo_webapp/
        ├── app/                    # app package (where __file__ is located)
        ├── app_input/              # input files
        └── app_output/             # output workspace
            ├── sims/
            ├── shards/
            ├── models/
            ├── bundles/
            └── logs/
        """
        here = Path(file_path or __file__).resolve()
        # here = .../pmarlo_webapp/app/backend/layout.py
        backend_dir = here.parent  # .../pmarlo_webapp/app/backend
        app_dir = backend_dir.parent  # .../pmarlo_webapp/app
        root = app_dir.parent.resolve()  # .../pmarlo_webapp

        # Ensure we got the right directory by checking for the app folder
        if not (root / "app").exists():
            raise RuntimeError(
                f"Expected to find 'app' directory in {root}, but it doesn't exist. "
                f"Check that the workspace structure is correct."
            )

        workspace = root / "app_output"
        layout = cls(
            app_root=root,
            inputs_dir=root / "app_input",
            workspace_dir=workspace,
            sims_dir=workspace / "sims",
            shards_dir=workspace / "shards",
            models_dir=workspace / "models",
            bundles_dir=workspace / "bundles",
            logs_dir=workspace / "logs",
            state_path=workspace / "state.json",
        )
        layout.ensure()
        return layout

    def ensure(self) -> None:
        for path in (
            self.workspace_dir,
            self.sims_dir,
            self.shards_dir,
            self.models_dir,
            self.bundles_dir,
            self.logs_dir,
        ):
            ensure_directory(path)
        ensure_directory(self.analysis_debug_dir)

    def available_inputs(self) -> List[Path]:
        if not self.inputs_dir.exists():
            return []
        return sorted(p.resolve() for p in self.inputs_dir.glob("*.pdb"))

    @property
    def analysis_debug_dir(self) -> Path:
        return self.workspace_dir / "analysis_debug"

    @property
    def legacy_app_root(self) -> Path:
        return (self.app_root.parent / "example_programs" / "app_usecase").resolve()

    def rebase_legacy_path(self, value: Path | str) -> Path:
        candidate: Path
        if isinstance(value, Path):
            candidate = value
        else:
            candidate = Path(str(value))
        candidate = candidate.expanduser()
        if not candidate.is_absolute():
            return candidate
        rebased = _rebase_path_with_root(candidate, self.legacy_app_root, self.app_root)
        if rebased is not None:
            return rebased
        return candidate.resolve()

    def normalize_path_string(self, value: str) -> str:
        if not isinstance(value, str) or not value:
            return value
        try:
            candidate = Path(value).expanduser()
        except Exception:
            return value
        if candidate.is_absolute():
            return str(self.rebase_legacy_path(candidate))
        return value