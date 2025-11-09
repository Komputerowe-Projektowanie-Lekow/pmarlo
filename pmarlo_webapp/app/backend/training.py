import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from pmarlo.api import normalize_training_metrics, coerce_tau_schedule
from pmarlo.utils.path_utils import ensure_directory
from pmarlo.features.deeptica.ts_feature_extractor import canonicalize_feature_spec

from .types import TrainingConfig, TrainingResult
from .utils import (
    _build_result_cls,
    _sanitize_artifacts,
    _timestamp,
    build_from_shards,
)

logger = logging.getLogger(__name__)

_CV_BUNDLE_REQUIRED = (
    "deeptica_cv_model.pt",
    "deeptica_cv_model_scaler.npz",
)


# Module-level helper functions
# Note: _coerce_tau_schedule has been moved to pmarlo.api as coerce_tau_schedule


class TrainingMixin:
    """Methods for DeepTICA model training.

    This class is mixed into the Backend class to provide ML model training,
    loading, and management operations.
    """

    def train_model(
        self,
        shard_jsons: Sequence[Path],
        config: TrainingConfig,
    ) -> TrainingResult:
        """Train a DeepTICA model from shards.

        Parameters
        ----------
        shard_jsons : Sequence[Path]
            Paths to shard JSON files for training
        config : TrainingConfig
            Training configuration parameters

        Returns
        -------
        TrainingResult
            Result containing trained model bundle and metadata
        """
        shards = [Path(p).resolve() for p in shard_jsons]
        if not shards:
            raise ValueError("No shards selected for training")

        stamp = _timestamp()
        bundle_path = self.layout.models_dir / f"deeptica-{stamp}.pbz"
        checkpoint_dir = self.layout.models_dir / f"training-{stamp}"
        ensure_directory(checkpoint_dir)

        # Setup logging to file
        log_file = checkpoint_dir / "training.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )

        pmarlo_logger = logging.getLogger("pmarlo")
        pmarlo_logger.addHandler(file_handler)
        pmarlo_logger.setLevel(logging.INFO)

        try:
            pmarlo_logger.info(f"Starting training with {len(shards)} shards")
            pmarlo_logger.info(
                f"Configuration: lag={config.lag}, bins={config.bins}, max_epochs={config.max_epochs}"
            )

            deeptica_params = config.deeptica_params()
            deeptica_params["checkpoint_dir"] = str(checkpoint_dir)

            pmarlo_logger.info("Calling build_from_shards...")
            br, ds_hash = build_from_shards(
                shard_jsons=shards,
                out_bundle=bundle_path,
                bins=dict(config.bins),
                lag=int(config.lag),
                seed=int(config.seed),
                temperature=float(config.temperature),
                learn_cv=True,
                deeptica_params=deeptica_params,
                notes={
                    "model_dir": str(self.layout.models_dir),
                    "checkpoint_dir": str(checkpoint_dir),
                },
            )
            pmarlo_logger.info("Training completed successfully")
        except ImportError as exc:
            pmarlo_logger.error(f"Import error: {exc}")
            raise RuntimeError(
                "Deep-TICA optional dependencies missing. Install pmarlo[mlcv] to enable"
            ) from exc
        except Exception as exc:
            pmarlo_logger.error(f"Training failed: {exc}", exc_info=True)
            raise
        finally:
            pmarlo_logger.removeHandler(file_handler)
            file_handler.close()

        # Export CV model for OpenMM integration
        cv_model_bundle_info = self._export_cv_model(bundle_path, checkpoint_dir, br)

        raw_metrics = _sanitize_artifacts(br.artifacts.get("mlcv_deeptica", {}))
        normalized_metrics = normalize_training_metrics(
            raw_metrics,
            tau_schedule=config.tau_schedule,
            epochs_per_tau=config.epochs_per_tau,
        )
        if isinstance(br.artifacts, dict):
            br.artifacts["mlcv_deeptica"] = normalized_metrics

        result = TrainingResult(
            bundle_path=bundle_path.resolve(),
            dataset_hash=ds_hash,
            build_result=br,
            created_at=stamp,
            checkpoint_dir=checkpoint_dir,
            cv_model_bundle=cv_model_bundle_info,
        )

        model_entry: Dict[str, Any] = {
            "bundle": str(bundle_path.resolve()),
            "checkpoint_dir": str(checkpoint_dir.resolve()),
            "dataset_hash": ds_hash,
            "lag": int(config.lag),
            "bins": dict(config.bins),
            "seed": int(config.seed),
            "temperature": float(config.temperature),
            "hidden": [int(h) for h in config.hidden],
            "max_epochs": int(config.max_epochs),
            "early_stopping": int(config.early_stopping),
            "tau_schedule": [int(t) for t in config.tau_schedule],
            "val_tau": int(config.val_tau),
            "epochs_per_tau": int(config.epochs_per_tau),
            "created_at": stamp,
            "metrics": normalized_metrics,
        }
        if cv_model_bundle_info:
            model_entry["cv_model_bundle"] = dict(cv_model_bundle_info)

        self.state.append_model(model_entry)

        logger.info(f"Trained model saved to {bundle_path}")
        return result

    def load_model(self, index: int) -> Optional[TrainingResult]:
        """Load a trained model by index.

        Parameters
        ----------
        index : int
            Index of the model in state

        Returns
        -------
        Optional[TrainingResult]
            Loaded model result or None if not found
        """
        if index < 0 or index >= len(self.state.models):
            return None
        entry = dict(self.state.models[index])
        return self._load_model_from_entry(entry)

    def delete_model(self, index: int) -> bool:
        """Delete a model and its associated files.

        Parameters
        ----------
        index : int
            Index of the model in state

        Returns
        -------
        bool
            True if deletion was successful, False otherwise
        """
        entry = self.state.remove_model(index)
        if entry is None:
            logger.warning(f"No model found at index {index}")
            return False

        try:
            bundle_path = self._path_from_value(entry.get("bundle"))
            if bundle_path is not None and bundle_path.exists():
                base_name = bundle_path.stem
                model_dir = bundle_path.parent

                # Find and delete related files
                for file_path in model_dir.glob(f"{base_name}.*"):
                    if file_path.is_file():
                        file_path.unlink()

            logger.info(f"Deleted model at index {index}")
            return True
        except Exception as e:
            logger.error(f"Error deleting model {index}: {e}")
            return False

    def get_training_progress(self, checkpoint_dir: Path) -> Optional[Dict[str, Any]]:
        """Read real-time training progress from checkpoint directory."""
        import json

        if not checkpoint_dir or not checkpoint_dir.exists():
            return None

        progress_path = checkpoint_dir / "training_progress.json"
        if not progress_path.exists():
            return None

        try:
            with progress_path.open("r") as f:
                return json.load(f)
        except Exception:
            return None

    def list_models(self) -> List[Dict[str, Any]]:
        """Return list of all trained models from state."""
        return [dict(entry) for entry in self.state.models]

    def training_config_from_entry(self, entry: Dict[str, Any]) -> TrainingConfig:
        """Reconstruct a TrainingConfig from a state entry."""
        return TrainingConfig(
            lag=int(entry.get("lag", 10)),
            bins=dict(entry.get("bins", {})),
            seed=int(entry.get("seed", 0)),
            temperature=float(entry.get("temperature", 300.0)),
            hidden=tuple(entry.get("hidden", [128, 128])),
            max_epochs=int(entry.get("max_epochs", 500)),
            early_stopping=int(entry.get("early_stopping", 20)),
            tau_schedule=tuple(entry.get("tau_schedule", [])),
            val_tau=int(entry.get("val_tau", 10)),
            epochs_per_tau=int(entry.get("epochs_per_tau", 100)),
            gradient_clip_val=float(entry.get("gradient_clip_val", 1.0)),
            learning_rate=float(entry.get("learning_rate", 3e-4)),
            weight_decay=float(entry.get("weight_decay", 0.0)),
        )

    def latest_model_path(self) -> Optional[Path]:
        """Return the path to the most recently trained model, or None if no models exist."""
        if not self.state.models:
            return None
        latest_entry = self.state.models[-1]
        bundle_path = self._path_from_value(latest_entry.get("bundle"))
        if bundle_path is not None and bundle_path.exists():
            return bundle_path
        return None

    def _load_model_from_entry(self, entry: Dict[str, Any]) -> Optional[TrainingResult]:
        """Load model from a state entry."""
        bundle_path = self._path_from_value(entry.get("bundle"))
        if bundle_path is None:
            return None

        br = self._load_build_result_from_path(bundle_path)
        if br is None:
            return None

        dataset_hash = str(entry.get("dataset_hash", "")) or (
            str(getattr(br.metadata, "dataset_hash", "")) if br.metadata else ""
        )
        created_at = str(entry.get("created_at", "")) or _timestamp()

        metrics = br.artifacts.get("mlcv_deeptica")
        if isinstance(metrics, Mapping):
            normalized = normalize_training_metrics(
                metrics,
                tau_schedule=entry.get("tau_schedule"),
                epochs_per_tau=entry.get("epochs_per_tau"),
            )
            br.artifacts["mlcv_deeptica"] = normalized

        return TrainingResult(
            bundle_path=bundle_path.resolve(),
            dataset_hash=dataset_hash,
            build_result=br,
            created_at=created_at,
            cv_model_bundle=entry.get("cv_model_bundle"),
        )

    def _load_build_result_from_path(self, path: Path) -> Optional[Any]:
        """Load a BuildResult from a bundle file."""
        try:
            bundle_path = Path(path)
        except TypeError:
            return None
        if not bundle_path.exists():
            return None
        try:
            text = bundle_path.read_text(encoding="utf-8")
        except Exception:
            return None
        try:
            return _build_result_cls().from_json(text)
        except Exception:
            return None

    def _export_cv_model(
        self, bundle_path: Path, checkpoint_dir: Path, br: Any
    ) -> Optional[Dict[str, Any]]:
        """Export CV model for OpenMM integration."""
        try:
            from pmarlo.features.deeptica import export_cv_model
            import torch
            import numpy as np

            # DeepTICA model files are saved in models_dir, not checkpoint_dir
            # The model is saved when training completes, so its timestamp will be
            # later than the checkpoint_dir timestamp (which is created at start)
            models_dir = checkpoint_dir.parent  # Parent of checkpoint_dir is models_dir

            # Extract timestamp from checkpoint_dir name (e.g., training-20251108-193156)
            checkpoint_timestamp = checkpoint_dir.name.replace("training-", "")

            # Find all model files and filter to those created during/after this training run
            # Only match main .pt files, not .scaler.pt files
            all_model_files = [
                f for f in sorted(models_dir.glob("deeptica-*.pt"))
                if not f.name.endswith(".scaler.pt")
            ]

            if not all_model_files:
                raise FileNotFoundError(
                    f"No DeepTICA model files (deeptica-*.pt) found in {models_dir}"
                )

            # Find model files with timestamps >= checkpoint timestamp
            # Model timestamp format: deeptica-YYYYMMDD-HHMMSS.pt
            matching_models = []
            for mf in all_model_files:
                model_timestamp = mf.stem.replace("deeptica-", "")
                if model_timestamp >= checkpoint_timestamp:
                    matching_models.append(mf)

            if matching_models:
                # Use the earliest model that matches (closest to training start)
                model_file = matching_models[0]
                logger.info(
                    f"Found model file {model_file.name} for training {checkpoint_dir.name}"
                )
            else:
                # Fallback: use the most recent model overall
                model_file = all_model_files[-1]
                logger.warning(
                    f"No model found with timestamp >= {checkpoint_timestamp}, "
                    f"using most recent model: {model_file.name}"
                )

            base_path = model_file.with_suffix("")

            logger.info(f"Loading DeepTICA model from {base_path}")
            
            # Load network state dict
            pt_file = base_path.with_suffix(".pt")
            scaler_file = base_path.with_suffix(".scaler.pt")
            config_file = base_path.with_suffix(".json")
            
            if not pt_file.exists():
                raise FileNotFoundError(f"Model file not found: {pt_file}")
            if not scaler_file.exists():
                raise FileNotFoundError(f"Scaler file not found: {scaler_file}")
            if not config_file.exists():
                raise FileNotFoundError(f"Config file not found: {config_file}")
            
            # Load the full DeepTICA model using the standard load method
            from pmarlo.features.deeptica._full import DeepTICAModel
            deeptica_model = DeepTICAModel.load(base_path)
            
            network = deeptica_model.net
            scaler = deeptica_model.scaler
            history = br.artifacts.get("mlcv_deeptica", {})

            if network is not None and scaler is not None:
                try:
                    import yaml
                except Exception as exc:
                    raise RuntimeError(
                        "PyYAML is required to load feature specifications"
                    ) from exc

                spec_path = self.layout.app_root / "app" / "feature_spec.yaml"
                if not spec_path.exists():
                    raise FileNotFoundError(
                        f"Feature specification not found at {spec_path}"
                    )

                with spec_path.open("r", encoding="utf-8") as spec_file:
                    feature_spec = yaml.safe_load(spec_file)

                normalized_spec = canonicalize_feature_spec(feature_spec)
                expected_features = int(normalized_spec.n_features)
                scaler_mean = np.asarray(getattr(scaler, "mean_", []))
                actual_features = int(scaler_mean.shape[0]) if scaler_mean.size else 0
                if expected_features <= 0:
                    raise RuntimeError(
                        "feature_spec.yaml does not define any molecular features. "
                        "Provide at least one feature to export CV bias bundles."
                    )
                if actual_features != expected_features:
                    raise RuntimeError(
                        "Feature count mismatch detected while exporting the CV bundle. "
                        f"feature_spec.yaml defines {expected_features} feature(s) but the "
                        f"trained model scaler expects {actual_features}. "
                        "Train on shards created with a molecular feature profile or "
                        "update feature_spec.yaml to match the training data."
                    )

                logger.info("Exporting CV model with bias potential for OpenMM integration...")
                cv_bundle = export_cv_model(
                    network=network,
                    scaler=scaler,
                    history=history,
                    output_dir=checkpoint_dir,
                    model_name="deeptica_cv_model",
                    bias_strength=10.0,
                    feature_spec=feature_spec,
                )

                logger.info("OK CV bias potential exported successfully")
                return {
                    "model_path": str(cv_bundle.model_path),
                    "scaler_path": str(cv_bundle.scaler_path),
                    "config_path": str(cv_bundle.config_path),
                    "metadata_path": str(cv_bundle.metadata_path),
                    "cv_dim": cv_bundle.cv_dim,
                    "feature_spec_sha256": cv_bundle.feature_spec_hash,
                }
            else:
                logger.error("Network or scaler is None after loading model, cannot export CV bundle")
                return None
        except Exception as e:
            logger.error(f"Could not export CV model: {e}", exc_info=True)
            raise RuntimeError(
                f"Failed to export CV model bundle: {e}. "
                "The model was trained successfully, but CV bias export failed. "
                "This usually indicates missing dependencies or configuration issues."
            ) from e

    # ------------------------------------------------------------------
    # CV bundle discovery / repair helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _probable_training_dir(bundle_path: Path) -> Optional[Path]:
        stem = bundle_path.stem
        if stem.startswith("deeptica-"):
            suffix = stem[len("deeptica-") :]
            return bundle_path.parent / f"training-{suffix}"
        return None

    @staticmethod
    def _cv_bundle_complete(path: Path) -> bool:
        return all((path / name).exists() for name in _CV_BUNDLE_REQUIRED)

    def _register_candidate_dir(
        self, value: Any, candidates: List[Path]
    ) -> None:
        candidate = self._path_from_value(value)
        if candidate is None:
            return
        if candidate.is_file():
            candidate = candidate.parent
        try:
            candidate = candidate.resolve()
        except Exception:
            candidate = candidate.absolute()
        if candidate not in candidates:
            candidates.append(candidate)

    def _resolve_cv_bundle_dir(self, entry: Mapping[str, Any]) -> Optional[Path]:
        candidates: List[Path] = []

        bundle_info = entry.get("cv_model_bundle")
        if isinstance(bundle_info, Mapping):
            for key in ("model_path", "scaler_path", "config_path", "metadata_path"):
                self._register_candidate_dir(bundle_info.get(key), candidates)

        self._register_candidate_dir(entry.get("checkpoint_dir"), candidates)

        bundle = entry.get("bundle")
        if bundle:
            bundle_path = self._path_from_value(bundle)
            if bundle_path is not None:
                inferred = self._probable_training_dir(bundle_path)
                if inferred is not None:
                    self._register_candidate_dir(inferred, candidates)

        for candidate in candidates:
            if candidate.is_dir() and self._cv_bundle_complete(candidate):
                return candidate
        return None

    def resolve_cv_bundle_dir(self, entry: Mapping[str, Any]) -> Optional[Path]:
        """Public helper to locate the CV bundle directory for a model entry."""
        return self._resolve_cv_bundle_dir(entry)

    def ensure_cv_bundle(self, index: int) -> Optional[Path]:
        """Make sure the selected model has an exported CV bundle on disk."""
        if index < 0 or index >= len(self.state.models):
            raise IndexError(f"Model index {index} is out of range")

        entry = dict(self.state.models[index])
        existing = self._resolve_cv_bundle_dir(entry)
        if existing is not None:
            return existing

        bundle_path = self._path_from_value(entry.get("bundle"))
        if bundle_path is None or not bundle_path.exists():
            raise FileNotFoundError("Model bundle is missing on disk; cannot export CV files.")

        checkpoint_dir = self._path_from_value(entry.get("checkpoint_dir"))
        if checkpoint_dir is None:
            inferred = self._probable_training_dir(bundle_path)
            if inferred is None:
                raise RuntimeError(
                    "Cannot determine checkpoint directory for CV export. "
                    "Re-run training or specify checkpoint_dir in state."
                )
            checkpoint_dir = inferred
        ensure_directory(checkpoint_dir)

        br = self._load_build_result_from_path(bundle_path)
        if br is None:
            raise RuntimeError(
                "Could not load build result from model bundle; CV export requires metadata."
            )

        info = self._export_cv_model(bundle_path, checkpoint_dir, br)
        if not info:
            raise RuntimeError(
                "CV model export completed without producing bundle files. "
                "Check the training logs for errors."
            )

        entry["checkpoint_dir"] = str(checkpoint_dir.resolve())
        entry["cv_model_bundle"] = dict(info)
        self.state.update_model(index, entry)

        resolved = self._resolve_cv_bundle_dir(entry)
        if resolved is None:
            raise RuntimeError(
                "CV model export reported success but the bundle files could not be located."
            )
        return resolved
