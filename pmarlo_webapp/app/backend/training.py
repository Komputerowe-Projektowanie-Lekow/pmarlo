from pathlib import Path
from typing import Sequence
from .types import TrainingConfig, TrainingResult

@staticmethod
def _load_build_result_from_path(path: Path) -> Optional["_BuildResult"]:
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

@staticmethod
def _coerce_tau_schedule(raw: Any) -> tuple[int, ...]:
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

def training_config_from_entry(self, entry: Dict[str, Any]) -> TrainingConfig:
    bins_raw = entry.get("bins")
    bins = (
        dict(bins_raw) if isinstance(bins_raw, dict) else {"Rg": 64, "RMSD_ref": 64}
    )
    hidden = self._coerce_hidden_layers(entry.get("hidden"))
    tau_raw = entry.get("tau_schedule")
    tau_schedule = self._coerce_tau_schedule(tau_raw)
    if not tau_schedule:
        tau_schedule = (int(entry.get("lag", 5)),)
    val_tau_entry = entry.get("val_tau")
    val_tau = (
        int(val_tau_entry)
        if val_tau_entry is not None
        else (tau_schedule[-1] if tau_schedule else int(entry.get("lag", 5)))
    )
    epochs_per_tau = int(entry.get("epochs_per_tau", 15))
    return TrainingConfig(
        lag=int(entry.get("lag", 5)),
        bins=bins,
        seed=int(entry.get("seed", 1337)),
        temperature=float(entry.get("temperature", 300.0)),
        hidden=hidden,
        max_epochs=int(entry.get("max_epochs", 200)),
        early_stopping=int(entry.get("early_stopping", 25)),
        tau_schedule=tau_schedule,
        val_tau=val_tau,
        epochs_per_tau=epochs_per_tau,
    )

def _coerce_deeptica_params(self, raw: Any) -> Optional[Dict[str, Any]]:
    if raw is None:
        return None
    if isinstance(raw, dict):
        return {str(k): v for k, v in raw.items()}
    return None

@staticmethod
def _coerce_hidden_layers(raw: Any) -> tuple[int, ...]:
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


def delete_model(self, index: int) -> bool:
    """Delete a model and its associated files."""
    entry = self.state.remove_model(index)
    if entry is None:
        return False

    try:
        # Delete model bundle file and associated files
        bundle_path = self._path_from_value(entry.get("bundle"))
        if bundle_path is not None and bundle_path.exists():
            base_name = bundle_path.stem
            model_dir = bundle_path.parent

            # Find and delete related files (history, json, pt files)
            for file_path in model_dir.glob(f"{base_name}.*"):
                if file_path.is_file():
                    file_path.unlink()

        return True
    except Exception:
        return False

def _load_model_from_entry(self, entry: Dict[str, Any]) -> Optional[TrainingResult]:
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


def load_model(self, index: int) -> Optional[TrainingResult]:
    if index < 0 or index >= len(self.state.models):
        return None
    entry = dict(self.state.models[index])
    return self._load_model_from_entry(entry)

def latest_model_path(self) -> Optional[Path]:
    if not self.state.models:
        return None
    last = self.state.models[-1]
    p = Path(last.get("bundle", ""))
    return p if p.exists() else None


def list_models(self) -> List[Dict[str, Any]]:
    enriched: List[Dict[str, Any]] = []
    for entry in self.state.models:
        data = dict(entry)
        metrics = data.get("metrics")
        tau_schedule = data.get("tau_schedule")
        epochs_per_tau = data.get("epochs_per_tau")
        data["metrics"] = _normalize_training_metrics(
            metrics,
            tau_schedule=tau_schedule if isinstance(tau_schedule, Sequence) else None,
            epochs_per_tau=epochs_per_tau if isinstance(epochs_per_tau, (int, float)) else None,
        )
        enriched.append(data)
    return enriched

def train_model(
        self,
        shard_jsons: Sequence[Path],
        config: TrainingConfig,
) -> TrainingResult:
    import logging

    shards = [Path(p).resolve() for p in shard_jsons]
    if not shards:
        raise ValueError("No shards selected for training")
    stamp = _timestamp()
    bundle_path = self.layout.models_dir / f"deeptica-{stamp}.pbz"

    # Create checkpoint directory for training progress
    checkpoint_dir = self.layout.models_dir / f"training-{stamp}"
    ensure_directory(checkpoint_dir)

    # Setup logging to file
    log_file = checkpoint_dir / "training.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

    # Get the pmarlo logger and add our file handler
    pmarlo_logger = logging.getLogger("pmarlo")
    pmarlo_logger.addHandler(file_handler)
    pmarlo_logger.setLevel(logging.INFO)

    try:
        pmarlo_logger.info(f"Starting training with {len(shards)} shards")
        pmarlo_logger.info(f"Configuration: lag={config.lag}, bins={config.bins}, max_epochs={config.max_epochs}")

        # Add checkpoint_dir to deeptica_params
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
            notes={"model_dir": str(self.layout.models_dir), "checkpoint_dir": str(checkpoint_dir)},
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
        # Remove the handler so it doesn't persist
        pmarlo_logger.removeHandler(file_handler)
        file_handler.close()
    # Export CV model for OpenMM integration
    cv_model_bundle_info = None
    try:
        from pmarlo.features.deeptica import export_cv_model

        # Load the trained model from bundle
        import pickle
        with open(bundle_path, "rb") as f:
            bundle_data = pickle.load(f)

        network = bundle_data.get("network")
        scaler = bundle_data.get("scaler")
        history = br.get("history", {})

        if network is not None and scaler is not None:
            try:
                import yaml
            except Exception as exc:
                raise RuntimeError("PyYAML is required to load feature specifications") from exc
            spec_path = self.layout.app_root / "app" / "feature_spec.yaml"
            if not spec_path.exists():
                raise FileNotFoundError(f"Feature specification not found at {spec_path}")
            with spec_path.open("r", encoding="utf-8") as spec_file:
                feature_spec = yaml.safe_load(spec_file)
            pmarlo_logger.info("Exporting CV model with bias potential for OpenMM integration...")
            pmarlo_logger.info("Creating CVBiasPotential wrapper (harmonic expansion bias)...")
            cv_bundle = export_cv_model(
                network=network,
                scaler=scaler,
                history=history,
                output_dir=checkpoint_dir,
                model_name="deeptica_cv_model",
                bias_strength=10.0,  # Can be made configurable
                feature_spec=feature_spec,
            )
            cv_model_bundle_info = {
                "model_path": str(cv_bundle.model_path),
                "scaler_path": str(cv_bundle.scaler_path),
                "config_path": str(cv_bundle.config_path),
                "metadata_path": str(cv_bundle.metadata_path),
                "cv_dim": cv_bundle.cv_dim,
                "feature_spec_sha256": cv_bundle.feature_spec_hash,
            }
            pmarlo_logger.info("✓ CV bias potential exported successfully")
            pmarlo_logger.info(f"  Model outputs: Energy (kJ/mol) for OpenMM force calculation")
            pmarlo_logger.info(f"  Bias formula: E = 10.0 * sum(cv_i^2)")
            pmarlo_logger.info(f"  Purpose: Encourages conformational exploration")
    except Exception as exc:
        pmarlo_logger.warning(f"Could not export CV model: {exc}")

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
            "checkpoint_dir": str(checkpoint_dir.resolve()),  # ADD THIS for CV model loading
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
    return result

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


class TrainingMixin:
    """Methods for DeepTICA model training."""

    def train_model(
            self,
            shard_jsons: Sequence[Path],
            config: TrainingConfig,
    ) -> TrainingResult:

    # ... your train_model implementation

    def load_model(self, index: int) -> Optional[TrainingResult]:

    # ... your load_model implementation

    def delete_model(self, index: int) -> bool:

    # ... your delete_model implementation

    def _load_model_from_entry(self, entry: Dict[str, Any]) -> Optional[TrainingResult]:
# ... your implementation
