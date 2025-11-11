from __future__ import annotations

"""Simple JSON-backed state manager for the Streamlit demo."""


import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

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

    def upsert_run(self, entry: Dict[str, Any]) -> None:
        """Insert or replace a run entry keyed by ``run_id``."""
        run_id = entry.get("run_id")
        if not run_id:
            raise ValueError("run metadata missing run_id")
        run_id_str = str(run_id)
        for idx, existing in enumerate(self._data.runs):
            if str(existing.get("run_id")) == run_id_str:
                self._data.runs[idx] = dict(entry)
                self._save()
                return
        self._data.runs.append(dict(entry))
        self._save()

    def append_shards(self, entry: Dict[str, Any]) -> None:
        self._data.shards.append(dict(entry))
        self._save()

    def append_model(self, entry: Dict[str, Any]) -> None:
        self._data.models.append(dict(entry))
        self._save()

    def update_model(self, index: int, entry: Dict[str, Any]) -> None:
        """Replace an existing model entry."""
        if not (0 <= index < len(self._data.models)):
            raise IndexError(f"Model index {index} out of range")
        self._data.models[index] = dict(entry)
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
        """Keep model entries synchronized with files on disk."""
        removed: List[int] = []
        existing_paths: Dict[str, int] = {}

        for i, entry in enumerate(self._data.models):
            bundle = entry.get("bundle")
            if not bundle:
                removed.append(i)
                continue

            path = Path(bundle)
            if path.exists():
                existing_paths[self._norm_path_key(path)] = i
            else:
                removed.append(i)

        if removed:
            for idx in reversed(removed):
                self._data.models.pop(idx)
            logger.info(f"Reconciled {len(removed)} missing models from state")

        added = 0
        if self.workspace_layout is not None:
            try:
                added = self._discover_models_on_disk(existing_paths)
            except Exception as exc:
                logger.warning("Failed to discover models on disk: %s", exc)

        if removed or added:
            self._save()
            if added:
                logger.info(f"Discovered {added} new model bundle(s)")

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

    # ------------------------------------------------------------------
    # Model discovery helpers
    # ------------------------------------------------------------------
    def _discover_models_on_disk(self, existing_paths: Dict[str, int]) -> int:
        """Scan the workspace for bundle files that are not in state."""
        layout = self.workspace_layout
        if layout is None:
            return 0

        models_dir = layout.models_dir
        if not models_dir.exists():
            return 0

        added = 0
        for bundle_path in sorted(models_dir.glob("deeptica-*.pbz")):
            key = self._norm_path_key(bundle_path)
            if key in existing_paths:
                continue

            entry = self._model_entry_from_bundle(bundle_path)
            self._data.models.append(entry)
            existing_paths[key] = len(self._data.models) - 1
            added += 1

        return added

    @staticmethod
    def _norm_path_key(path: Path) -> str:
        """Normalize a path for robust comparisons."""
        try:
            resolved = path.expanduser().resolve()
        except Exception:
            resolved = path
        return os.path.normcase(str(resolved))

    @staticmethod
    def _maybe_int(value: Any) -> Optional[int]:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _maybe_float(value: Any) -> Optional[float]:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _as_mapping(value: Any) -> Dict[str, Any]:
        if isinstance(value, Mapping):
            return dict(value)
        return {}

    def _model_entry_from_bundle(self, bundle_path: Path) -> Dict[str, Any]:
        """Create a state entry for a bundle discovered on disk."""
        entry: Dict[str, Any] = {
            "bundle": str(bundle_path.expanduser().resolve()),
            "created_at": self._infer_bundle_timestamp(bundle_path),
        }

        payload = self._read_bundle_json(bundle_path)
        if not isinstance(payload, Mapping):
            return entry

        metadata = self._as_mapping(payload.get("metadata"))
        applied_opts = self._as_mapping(metadata.get("applied_opts"))
        artifacts = self._as_mapping(payload.get("artifacts"))

        plan_steps = []
        plan = self._as_mapping(applied_opts.get("actual_plan"))
        steps = plan.get("steps")
        if isinstance(steps, list):
            plan_steps = steps
        elif isinstance(metadata.get("transform_plan"), list):
            plan_steps = metadata["transform_plan"]

        learn_params: Dict[str, Any] = {}
        for step in plan_steps:
            if isinstance(step, Mapping) and step.get("name") == "LEARN_CV":
                params = step.get("params")
                if isinstance(params, Mapping):
                    learn_params = dict(params)
                break

        def assign(key: str, value: Any) -> None:
            if value is not None:
                entry[key] = value

        assign("dataset_hash", metadata.get("dataset_hash"))
        assign("lag", self._maybe_int(
            learn_params.get("lag") or metadata.get("lag") or applied_opts.get("lag")
        ))
        assign("bins", self._as_mapping(applied_opts.get("bins")))
        assign("seed", self._maybe_int(metadata.get("seed")))
        assign("temperature", self._maybe_float(metadata.get("temperature")))
        hidden = learn_params.get("hidden")
        if isinstance(hidden, list):
            normalized_hidden: List[int] = []
            for value in hidden:
                converted = self._maybe_int(value)
                if converted is not None:
                    normalized_hidden.append(converted)
            if normalized_hidden:
                assign("hidden", normalized_hidden)
        assign("max_epochs", self._maybe_int(learn_params.get("max_epochs")))
        assign("early_stopping", self._maybe_int(learn_params.get("early_stopping")))
        tau_schedule = learn_params.get("tau_schedule")
        if isinstance(tau_schedule, list):
            normalized_tau: List[int] = []
            for value in tau_schedule:
                converted = self._maybe_int(value)
                if converted is not None:
                    normalized_tau.append(converted)
            if normalized_tau:
                assign("tau_schedule", normalized_tau)
        assign("val_tau", self._maybe_int(learn_params.get("val_tau")))
        assign("epochs_per_tau", self._maybe_int(learn_params.get("epochs_per_tau")))
        assign("gradient_clip_val", self._maybe_float(learn_params.get("gradient_clip_val")))
        assign("learning_rate", self._maybe_float(learn_params.get("learning_rate")))
        assign("weight_decay", self._maybe_float(learn_params.get("weight_decay")))
        checkpoint_dir = learn_params.get("checkpoint_dir")
        if isinstance(checkpoint_dir, str) and checkpoint_dir:
            assign("checkpoint_dir", checkpoint_dir)
            cv_bundle = self._discover_cv_bundle(Path(checkpoint_dir))
            if cv_bundle:
                assign("cv_model_bundle", cv_bundle)

        metrics = artifacts.get("mlcv_deeptica")
        if isinstance(metrics, Mapping):
            assign("metrics", dict(metrics))

        return entry

    def _read_bundle_json(self, bundle_path: Path) -> Optional[Dict[str, Any]]:
        try:
            text = bundle_path.read_text(encoding="utf-8")
            payload = json.loads(text)
            if isinstance(payload, Mapping):
                return dict(payload)
        except Exception as exc:
            logger.warning("Failed to parse model bundle %s: %s", bundle_path, exc)
        return None

    @staticmethod
    def _infer_bundle_timestamp(bundle_path: Path) -> str:
        stem = bundle_path.stem
        for prefix in ("deeptica-", "model-", "bundle-"):
            if stem.startswith(prefix):
                stem = stem[len(prefix):]
                break
        try:
            dt = datetime.strptime(stem, "%Y%m%d-%H%M%S")
            return dt.strftime("%Y%m%d-%H%M%S")
        except ValueError:
            pass
        try:
            dt = datetime.fromtimestamp(bundle_path.stat().st_mtime)
            return dt.strftime("%Y%m%d-%H%M%S")
        except Exception:
            return stem

    def _discover_cv_bundle(self, checkpoint_dir: Path) -> Optional[Dict[str, Any]]:
        """Find exported CV bundle artifacts next to a checkpoint directory."""
        candidates = {
            "model_path": checkpoint_dir / "deeptica_cv_model.pt",
            "scaler_path": checkpoint_dir / "deeptica_cv_model_scaler.npz",
            "config_path": checkpoint_dir / "deeptica_cv_model_config.json",
            "metadata_path": checkpoint_dir / "deeptica_cv_model_metadata.json",
        }

        if not all(path.exists() for path in candidates.values()):
            return None

        payload = {}
        metadata_path = candidates["metadata_path"]
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            payload = {}

        cv_dim = payload.get("cv_dim")
        feature_spec_sha = payload.get("feature_spec_sha256")

        bundle_info: Dict[str, Any] = {
            key: str(path)
            for key, path in candidates.items()
        }
        if cv_dim is not None:
            bundle_info["cv_dim"] = cv_dim
        if feature_spec_sha:
            bundle_info["feature_spec_sha256"] = feature_spec_sha
        return bundle_info


# Alias for backwards compatibility
PersistentState = StateManager


__all__ = ["StateManager", "PersistentState"]
