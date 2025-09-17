from __future__ import annotations

"""Lightweight manifest management for the Streamlit workflow app.

The state file lives under app_output/state.json and records a summary of
simulations, emitted shards, trained models, and analysis bundles.  Each helper
here keeps the schema compact while remaining tolerant to partial writes or
manual edits.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List
import json

__all__ = ["StateManager", "empty_manifest"]


def empty_manifest() -> Dict[str, List[Dict[str, Any]]]:
    """Return a new empty manifest structure."""

    return {
        "runs": [],
        "shards": [],
        "models": [],
        "builds": [],
    }


@dataclass
class StateManager:
    """Tiny helper around the JSON manifest to keep the UI stateless.

    The manager eagerly loads the manifest on construction, provides append
    helpers for the four tracked asset types, and writes atomically (best effort)
    on every update. Consumers are expected to treat the instance as
    short-lived; call :meth:`refresh` if another process may have modified the
    manifest on disk.
    """

    path: Path

    def __post_init__(self) -> None:  # pragma: no cover - trivial container
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._data = self._read()

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def _read(self) -> Dict[str, List[Dict[str, Any]]]:
        if not self.path.exists():
            return empty_manifest()
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return empty_manifest()
        state = empty_manifest()
        for key in state:
            value = raw.get(key)
            if isinstance(value, list):
                state[key] = [dict(item) for item in value if isinstance(item, dict)]
        return state

    def refresh(self) -> None:
        """Reload the manifest from disk."""

        self._data = self._read()

    def save(self) -> None:
        """Write the current manifest to disk (best effort)."""

        text = json.dumps(self._data, sort_keys=True, indent=2)
        self.path.write_text(text, encoding="utf-8")

    # ------------------------------------------------------------------
    # Append helpers
    # ------------------------------------------------------------------
    def _append(self, key: str, entry: Dict[str, Any]) -> None:
        bucket = self._data.setdefault(key, [])
        bucket.append(dict(entry))
        self.save()

    def append_run(self, entry: Dict[str, Any]) -> None:
        self._append("runs", entry)

    def append_shards(self, entry: Dict[str, Any]) -> None:
        self._append("shards", entry)

    def append_model(self, entry: Dict[str, Any]) -> None:
        self._append("models", entry)

    def append_build(self, entry: Dict[str, Any]) -> None:
        self._append("builds", entry)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------
    @property
    def data(self) -> Dict[str, List[Dict[str, Any]]]:
        return {
            key: [dict(item) for item in values]
            for key, values in self._data.items()
        }

    @property
    def runs(self) -> List[Dict[str, Any]]:
        return [dict(run) for run in self._data.get("runs", [])]

    @property
    def shards(self) -> List[Dict[str, Any]]:
        return [dict(entry) for entry in self._data.get("shards", [])]

    @property
    def models(self) -> List[Dict[str, Any]]:
        return [dict(entry) for entry in self._data.get("models", [])]

    @property
    def builds(self) -> List[Dict[str, Any]]:
        return [dict(entry) for entry in self._data.get("builds", [])]

    def summary(self) -> Dict[str, int]:
        """Return lightweight counts for display in the sidebar."""

        return {
            "runs": len(self._data.get("runs", [])),
            "shards": sum(len(entry.get("paths", [])) for entry in self._data.get("shards", [])),
            "models": len(self._data.get("models", [])),
            "builds": len(self._data.get("builds", [])),
        }

    # ------------------------------------------------------------------
    # Delete helpers
    # ------------------------------------------------------------------
    def _remove_entry(self, key: str, index: int) -> Dict[str, Any] | None:
        """Remove entry at index from the specified list and return it."""
        bucket = self._data.get(key, [])
        if 0 <= index < len(bucket):
            removed = bucket.pop(index)
            self.save()
            return removed
        return None

    def remove_run(self, index: int) -> Dict[str, Any] | None:
        """Remove a run entry by index."""
        return self._remove_entry("runs", index)

    def remove_shards(self, index: int) -> Dict[str, Any] | None:
        """Remove a shards entry by index."""
        return self._remove_entry("shards", index)

    def remove_model(self, index: int) -> Dict[str, Any] | None:
        """Remove a model entry by index."""
        return self._remove_entry("models", index)

    def remove_build(self, index: int) -> Dict[str, Any] | None:
        """Remove a build entry by index."""
        return self._remove_entry("builds", index)

