from __future__ import annotations

"""Simple JSON-backed state manager for the Streamlit demo."""


import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from pmarlo.utils.path_utils import ensure_directory

logger = logging.getLogger(__name__)


def _transform_strings(value: Any, transform: Callable[[str], str]) -> Tuple[Any, bool]:
    if isinstance(value, dict):
        changed = False
        result: Dict[str, Any] = {}
        for key, item in value.items():
            new_item, delta = _transform_strings(item, transform)
            result[key] = new_item
            if delta:
                changed = True
        return result, changed
    if isinstance(value, list):
        changed = False
        result_list: List[Any] = []
        for item in value:
            new_item, delta = _transform_strings(item, transform)
            result_list.append(new_item)
            if delta:
                changed = True
        return result_list, changed
    if isinstance(value, tuple):
        changed = False
        result_items: List[Any] = []
        for item in value:
            new_item, delta = _transform_strings(item, transform)
            result_items.append(new_item)
            if delta:
                changed = True
        if changed:
            return tuple(result_items), True
        return value, False
    if isinstance(value, str):
        new_value = transform(value)
        return new_value, new_value != value
    return value, False


@dataclass
class _StateData:
    runs: List[Dict[str, Any]] = field(default_factory=list)
    shards: List[Dict[str, Any]] = field(default_factory=list)
    models: List[Dict[str, Any]] = field(default_factory=list)
    builds: List[Dict[str, Any]] = field(default_factory=list)
    conformations: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "runs": list(self.runs),
            "shards": list(self.shards),
            "models": list(self.models),
            "builds": list(self.builds),
            "conformations": list(self.conformations),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "_StateData":
        return cls(
            runs=list(payload.get("runs", [])),
            shards=list(payload.get("shards", [])),
            models=list(payload.get("models", [])),
            builds=list(payload.get("builds", [])),
            conformations=list(payload.get("conformations", [])),
        )


class StateManager:
    """Persist JSON state for runs, shard batches, trained models, and builds.

    This class now supports dynamic filesystem discovery to ensure the state
    stays synchronized with actual files on disk.
    """

    def __init__(self, path: str | Path, *, workspace_layout=None) -> None:
        self.path = Path(path)
        self.workspace_layout = workspace_layout
        ensure_directory(self.path.parent)
        self._data = _StateData()
        self._load()

    # ------------------------------------------------------------------
    # Collection accessors with dynamic reconciliation
    # ------------------------------------------------------------------
    @property
    def runs(self) -> List[Dict[str, Any]]:
        """Get runs, automatically removing entries with missing directories."""
        self._reconcile_runs()
        return self._data.runs

    @property
    def shards(self) -> List[Dict[str, Any]]:
        """Get shards, automatically removing entries with missing files."""
        self._reconcile_shards()
        return self._data.shards

    @property
    def models(self) -> List[Dict[str, Any]]:
        """Get models, automatically removing entries with missing files."""
        self._reconcile_models()
        return self._data.models

    @property
    def builds(self) -> List[Dict[str, Any]]:
        """Get builds, automatically removing entries with missing files."""
        self._reconcile_builds()
        return self._data.builds

    @property
    def conformations(self) -> List[Dict[str, Any]]:
        """Get conformations, automatically removing entries with missing files."""
        self._reconcile_conformations()
        return self._data.conformations

    # ------------------------------------------------------------------
    # Mutation helpers
    # ------------------------------------------------------------------
    def append_run(self, entry: Dict[str, Any]) -> None:
        self._data.runs.append(dict(entry))
        self._save()

    def append_shards(self, entry: Dict[str, Any]) -> None:
        self._data.shards.append(dict(entry))
        self._save()

    def append_model(self, entry: Dict[str, Any]) -> None:
        self._data.models.append(dict(entry))
        self._save()

    def append_build(self, entry: Dict[str, Any]) -> None:
        self._data.builds.append(dict(entry))
        self._save()

    def append_conformations(self, entry: Dict[str, Any]) -> None:
        self._data.conformations.append(dict(entry))
        self._save()

    def remove_run(self, index: int) -> Optional[Dict[str, Any]]:
        return self._remove_from(self._data.runs, index)

    def remove_shards(self, index: int) -> Optional[Dict[str, Any]]:
        return self._remove_from(self._data.shards, index)

    def remove_model(self, index: int) -> Optional[Dict[str, Any]]:
        return self._remove_from(self._data.models, index)

    def remove_build(self, index: int) -> Optional[Dict[str, Any]]:
        return self._remove_from(self._data.builds, index)

    def remove_conformations(self, index: int) -> Optional[Dict[str, Any]]:
        return self._remove_from(self._data.conformations, index)

    # ------------------------------------------------------------------
    # Dynamic reconciliation methods
    # ------------------------------------------------------------------
    def _reconcile_runs(self) -> None:
        """Remove run entries where the run directory no longer exists."""
        to_remove = []
        for i, entry in enumerate(self._data.runs):
            run_dir = entry.get("run_dir")
            if run_dir:
                path = Path(run_dir)
                if not path.exists() or not path.is_dir():
                    to_remove.append(i)

        if to_remove:
            for i in reversed(to_remove):
                self._data.runs.pop(i)
            self._save()
            logger.info(f"Reconciled {len(to_remove)} missing runs from state")

    def _reconcile_shards(self) -> None:
        """Remove shard entries where all referenced files are missing."""
        to_remove = []
        for i, entry in enumerate(self._data.shards):
            # Check if directory exists
            directory = entry.get("directory")
            if directory:
                dir_path = Path(directory)
                if not dir_path.exists():
                    to_remove.append(i)
                    continue

            # Check if any shard files exist
            paths = entry.get("paths", [])
            if paths:
                any_exist = any(Path(p).exists() for p in paths)
                if not any_exist:
                    to_remove.append(i)

        if to_remove:
            for i in reversed(to_remove):
                self._data.shards.pop(i)
            self._save()
            logger.info(f"Reconciled {len(to_remove)} missing shard batches from state")

    def _reconcile_models(self) -> None:
        """Remove model entries where the bundle file no longer exists."""
        to_remove = []
        for i, entry in enumerate(self._data.models):
            bundle = entry.get("bundle")
            if bundle:
                path = Path(bundle)
                if not path.exists():
                    to_remove.append(i)

        if to_remove:
            for i in reversed(to_remove):
                self._data.models.pop(i)
            self._save()
            logger.info(f"Reconciled {len(to_remove)} missing models from state")

    def _reconcile_builds(self) -> None:
        """Remove build entries where the bundle file no longer exists."""
        to_remove = []
        for i, entry in enumerate(self._data.builds):
            bundle = entry.get("bundle")
            if bundle:
                path = Path(bundle)
                if not path.exists():
                    to_remove.append(i)

        if to_remove:
            for i in reversed(to_remove):
                self._data.builds.pop(i)
            self._save()
            logger.info(f"Reconciled {len(to_remove)} missing builds from state")

    def _reconcile_conformations(self) -> None:
        """Remove conformation entries where the PDB file no longer exists."""
        to_remove = []
        for i, entry in enumerate(self._data.conformations):
            pdb_path = entry.get("pdb_path")
            if pdb_path:
                path = Path(pdb_path)
                if not path.exists():
                    to_remove.append(i)

        if to_remove:
            for i in reversed(to_remove):
                self._data.conformations.pop(i)
            self._save()
            logger.info(f"Reconciled {len(to_remove)} missing conformations from state")

    def force_reconcile_all(self) -> Dict[str, int]:
        """Force reconciliation of all collections and return counts of removed items."""
        initial_counts = {
            "runs": len(self._data.runs),
            "shards": len(self._data.shards),
            "models": len(self._data.models),
            "builds": len(self._data.builds),
            "conformations": len(self._data.conformations),
        }

        self._reconcile_runs()
        self._reconcile_shards()
        self._reconcile_models()
        self._reconcile_builds()
        self._reconcile_conformations()

        final_counts = {
            "runs": len(self._data.runs),
            "shards": len(self._data.shards),
            "models": len(self._data.models),
            "builds": len(self._data.builds),
            "conformations": len(self._data.conformations),
        }

        removed = {
            key: initial_counts[key] - final_counts[key]
            for key in initial_counts
        }

        return removed

    # ------------------------------------------------------------------
    # Summaries & persistence
    # ------------------------------------------------------------------
    def summary(self) -> Dict[str, int]:
        """Return counts after reconciliation."""
        return {
            "runs": len(self.runs),
            "shards": len(self.shards),
            "models": len(self.models),
            "builds": len(self.builds),
            "conformations": len(self.conformations),
        }

    def normalize_strings(self, transform: Callable[[str], str]) -> bool:
        payload = self._data.to_dict()
        transformed, changed = _transform_strings(payload, transform)
        if changed:
            self._data = _StateData.from_dict(transformed)
            self._save()
        return changed

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _remove_from(
        self, collection: List[Dict[str, Any]], index: int
    ) -> Optional[Dict[str, Any]]:
        if 0 <= index < len(collection):
            entry = collection.pop(index)
            self._save()
            return entry
        return None

    def _load(self) -> None:
        if not self.path.exists():
            logger.info(f"State file does not exist, starting with empty state: {self.path}")
            return
        try:
            text_content = self.path.read_text(encoding="utf-8")
            payload = json.loads(text_content)
            self._data = _StateData.from_dict(payload)
            logger.info(
                f"Loaded state from {self.path}: "
                f"{len(self._data.runs)} runs, "
                f"{len(self._data.shards)} shards, "
                f"{len(self._data.models)} models, "
                f"{len(self._data.builds)} builds, "
                f"{len(self._data.conformations)} conformations"
            )
        except Exception as exc:
            # Corrupt or unreadable file; start fresh to avoid crashing the UI.
            logger.error(
                f"Failed to load state from {self.path}: {exc}. Starting with empty state.",
                exc_info=True
            )
            self._data = _StateData()

    def _save(self) -> None:
        tmp_path = self.path.with_suffix(".tmp")
        tmp_path.write_text(
            json.dumps(self._data.to_dict(), indent=2), encoding="utf-8"
        )
        tmp_path.replace(self.path)


# Alias for backwards compatibility
PersistentState = StateManager


__all__ = ["StateManager", "PersistentState"]
