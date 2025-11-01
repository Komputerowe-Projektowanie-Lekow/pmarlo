import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from pmarlo.utils.path_utils import ensure_directory

from .types import TrainingConfig, TrainingResult
from .utils import (
    _build_result_cls,
    _normalize_training_metrics,
    _sanitize_artifacts,
    _timestamp,
    build_from_shards,
)

logger = logging.getLogger(__name__)


# Module-level helper functions
def _coerce_hidden_layers(raw: Any) -> tuple[int, ...]:
    """Parse hidden layer specification from various formats."""
    layers: List[int] = []
    if isinstance(raw, (list, tuple)):
        for item in raw:
            try:
                layers.append(int(item))
            except (TypeError, ValueError):
                continue
    elif isinstance(raw, str):
        for token in raw.split(","):
            token = token.strip()
            if not token:
                continue
            try:
                layers.append(int(token))
            except ValueError:
                continue
    if layers:
        return tuple(layers)
    return (128, 128)


def _coerce_tau_schedule(raw: Any) -> tuple[int, ...]:
    """Parse tau schedule from various formats."""
    values: List[int] = []
    if isinstance(raw, (list, tuple)):
        for item in raw:
            try:
                v = int(item)
                if v > 0:
                    values.append(v)
            except (TypeError, ValueError):
                continue
    elif isinstance(raw, str):
        tokens = raw.replace(";", ",").split(",")
        for token in tokens:
            token = token.strip()
            if not token:
                continue
            try:
                v = int(token)
                if v > 0:
                    values.append(v)
            except ValueError:
                continue
    if not values:
        return ()
    return tuple(sorted(set(values)))


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
        normalized_metrics = _normalize_training_metrics(
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

        self.state.append_model(
            {
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
        )

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
            normalized = _normalize_training_metrics(
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
            import pickle

            with open(bundle_path, "rb") as f:
                bundle_data = pickle.load(f)

            network = bundle_data.get("network")
            scaler = bundle_data.get("scaler")
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

                logger.info("✓ CV bias potential exported successfully")
                return {
                    "model_path": str(cv_bundle.model_path),
                    "scaler_path": str(cv_bundle.scaler_path),
                    "config_path": str(cv_bundle.config_path),
                    "metadata_path": str(cv_bundle.metadata_path),
                    "cv_dim": cv_bundle.cv_dim,
                    "feature_spec_sha256": cv_bundle.feature_spec_hash,
                }
        except Exception as e:
            logger.warning(f"Could not export CV model: {e}")
            return None

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

