from pathlib import Path
from typing import Dict, List, Any

from __future__ import annotations

"""Simple JSON-backed state manager for the Streamlit demo."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from pmarlo.utils.path_utils import ensure_directory


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
    """Persist JSON state for runs, shard batches, trained models, and builds."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        ensure_directory(self.path.parent)
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

    @property
    def conformations(self) -> List[Dict[str, Any]]:
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
    # Summaries & persistence
    # ------------------------------------------------------------------
    def summary(self) -> Dict[str, int]:
        return {
            "runs": len(self._data.runs),
            "shards": len(self._data.shards),
            "models": len(self._data.models),
            "builds": len(self._data.builds),
            "conformations": len(self._data.conformations),
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



def _migrate_state_paths(self) -> None:
    if not hasattr(self.state, "normalize_strings"):
        return
    try:
        changed = self.state.normalize_strings(self.layout.normalize_path_string)
    except Exception:
        return
    if changed:
        logger.debug(
            "Rebased legacy app_usecase paths to new workspace root %s",
            self.layout.app_root,
        )

def build_config_from_entry(self, entry: Dict[str, Any]) -> BuildConfig:
    bins_raw = entry.get("bins")
    bins = (
        dict(bins_raw) if isinstance(bins_raw, dict) else {"Rg": 64, "RMSD_ref": 64}
    )
    deeptica_params = self._coerce_deeptica_params(entry.get("deeptica_params"))
    notes = {}
    entry_notes = entry.get("notes")
    if isinstance(entry_notes, dict):
        notes.update(entry_notes)
    apply_whitening = bool(entry.get("apply_cv_whitening", True))
    cluster_mode = str(entry.get("cluster_mode", "kmeans"))
    n_microstates = int(entry.get("n_microstates", 20))
    kmeans_kwargs_raw = entry.get("kmeans_kwargs")
    kmeans_kwargs = (
        dict(kmeans_kwargs_raw)
        if isinstance(kmeans_kwargs_raw, dict)
        else {"n_init": 50}
    )
    reweight_mode = str(entry.get("reweight_mode", "MBAR"))
    fes_method = str(entry.get("fes_method", "kde"))
    bw_raw = entry.get("fes_bandwidth", "scott")
    try:
        fes_bandwidth = float(bw_raw)
    except (TypeError, ValueError):
        fes_bandwidth = bw_raw if bw_raw is not None else "scott"
    min_count = int(entry.get("fes_min_count_per_bin", 1))
    return BuildConfig(
        lag=int(entry.get("lag", 10)),
        bins=bins,
        seed=int(entry.get("seed", 2025)),
        temperature=float(entry.get("temperature", 300.0)),
        learn_cv=bool(entry.get("learn_cv", False)),
        deeptica_params=deeptica_params,
        notes=notes,
        apply_cv_whitening=apply_whitening,
        cluster_mode=cluster_mode,
        n_microstates=n_microstates,
        reweight_mode=reweight_mode,
        fes_method=fes_method,
        fes_bandwidth=fes_bandwidth,
        fes_min_count_per_bin=min_count,
    )

def sidebar_summary(self) -> Dict[str, int]:
    # Reconcile stale manifest entries first
    self._reconcile_shard_state()
    self._reconcile_conformation_state()

    # Count shard files on disk for accuracy
    try:
        shard_files = len(self.discover_shards())
    except Exception:
        shard_files = len(self.state.shards)

    return {
        "runs": len(self.state.runs),
        "shards": int(shard_files),
        "models": len(self.state.models),
        "builds": len(self.state.builds),
        "conformations": len(self.state.conformations),
    }


class PersistentState:
    """Manages workspace state persistence."""

    def __init__(self, manifest_path: Path):

    # ... all PersistentState methods

    def append_run(self, entry: Dict[str, Any]) -> None:

    # ... your implementation

    def remove_run(self, index: int) -> Optional[Dict[str, Any]]:
# ... your implementation

# ... all other state methods
