from __future__ import annotations

"""Simple JSON-backed state manager for the Streamlit demo."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class _StateData:
    runs: List[Dict[str, Any]] = field(default_factory=list)
    shards: List[Dict[str, Any]] = field(default_factory=list)
    models: List[Dict[str, Any]] = field(default_factory=list)
    builds: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "runs": list(self.runs),
            "shards": list(self.shards),
            "models": list(self.models),
            "builds": list(self.builds),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "_StateData":
        return cls(
            runs=list(payload.get("runs", [])),
            shards=list(payload.get("shards", [])),
            models=list(payload.get("models", [])),
            builds=list(payload.get("builds", [])),
        )


class StateManager:
    """Persist JSON state for runs, shard batches, trained models, and builds."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._data = _StateData()
        self._load()

    # ------------------------------------------------------------------
    # Collection accessors
    # ------------------------------------------------------------------
    @property
    def runs(self) -> List[Dict[str, Any]]:
        return self._data.runs

    @property
    def shards(self) -> List[Dict[str, Any]]:
        return self._data.shards

    @property
    def models(self) -> List[Dict[str, Any]]:
        return self._data.models

    @property
    def builds(self) -> List[Dict[str, Any]]:
        return self._data.builds

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

    def remove_run(self, index: int) -> Optional[Dict[str, Any]]:
        return self._remove_from(self._data.runs, index)

    def remove_shards(self, index: int) -> Optional[Dict[str, Any]]:
        return self._remove_from(self._data.shards, index)

    def remove_model(self, index: int) -> Optional[Dict[str, Any]]:
        return self._remove_from(self._data.models, index)

    def remove_build(self, index: int) -> Optional[Dict[str, Any]]:
        return self._remove_from(self._data.builds, index)

    # ------------------------------------------------------------------
    # Summaries & persistence
    # ------------------------------------------------------------------
    def summary(self) -> Dict[str, int]:
        return {
            "runs": len(self._data.runs),
            "shards": len(self._data.shards),
            "models": len(self._data.models),
            "builds": len(self._data.builds),
        }

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
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            self._data = _StateData.from_dict(payload)
        except Exception:
            # Corrupt or unreadable file; start fresh to avoid crashing the UI.
            self._data = _StateData()

    def _save(self) -> None:
        tmp_path = self.path.with_suffix(".tmp")
        tmp_path.write_text(
            json.dumps(self._data.to_dict(), indent=2), encoding="utf-8"
        )
        tmp_path.replace(self.path)


__all__ = ["StateManager"]
