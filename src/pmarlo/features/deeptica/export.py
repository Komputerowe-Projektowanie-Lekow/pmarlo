"""Export trained DeepTICA models for OpenMM integration."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch
from torch import nn

from pmarlo.features.deeptica.ts_feature_extractor import (
    build_feature_extractor_module,
    canonicalize_feature_spec,
)
from pmarlo.utils.path_utils import ensure_directory

logger = logging.getLogger(__name__)

__all__ = [
    "CVModelBundle",
    "export_cv_model",
    "export_cv_bias_potential",
    "load_cv_model_info",
]


@dataclass(slots=True)
class CVModelBundle:
    """Container for exported CV model files and metadata."""

    model_path: Path
    scaler_path: Path
    config_path: Path
    metadata_path: Path
    feature_names: list[str]
    cv_dim: int
    feature_spec_hash: str


def _json_compatible(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        return {str(k): _json_compatible(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_compatible(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _json_compatible(obj.tolist())
    if isinstance(obj, (np.generic,)):
        return obj.item()
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    return str(obj)


def _hash_feature_spec(payload: Any) -> str:
    serialised = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(serialised).hexdigest()


def export_cv_bias_potential(
    network: nn.Module,
    scaler: Any,
    history: dict[str, Any],
    output_dir: str | Path,
    *,
    feature_spec: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    model_name: str = "deeptica_cv_bias",
    bias_strength: float = 10.0,
    feature_names: Optional[list[str]] = None,
) -> CVModelBundle:
    """Export a Deep-TICA network wrapped with TorchScript feature extraction and bias."""

    from pmarlo.features.deeptica.cv_bias_potential import create_cv_bias_potential

    output_path = Path(output_dir)
    ensure_directory(output_path)

    if network is None:
        raise ValueError("network cannot be None")
    if scaler is None:
        raise ValueError("scaler cannot be None")

    scaler_mean = np.asarray(getattr(scaler, "mean_", []), dtype=np.float32)
    scaler_scale = np.asarray(getattr(scaler, "scale_", []), dtype=np.float32)
    if scaler_mean.size == 0 or scaler_scale.size == 0:
        raise ValueError("scaler must expose non-empty mean_ and scale_ arrays")

    normalized_spec = canonicalize_feature_spec(feature_spec)
    if normalized_spec.n_features != int(scaler_mean.shape[0]):
        raise ValueError(
            "feature specification defines "
            f"{normalized_spec.n_features} features but scaler expects "
            f"{int(scaler_mean.shape[0])}"
        )

    spec_payload = _json_compatible(feature_spec)
    feature_spec_hash = _hash_feature_spec(spec_payload)

    feature_extractor = build_feature_extractor_module(normalized_spec)
    feature_extractor.eval()

    resolved_feature_names = (
        feature_names
        if feature_names is not None
        else list(normalized_spec.feature_names)
    )
    if len(resolved_feature_names) != normalized_spec.n_features:
        raise ValueError(
            "feature_names length does not match feature specification "
            f"({len(resolved_feature_names)} vs {normalized_spec.n_features})"
        )

    logger.info(
        "Creating CV bias potential wrapper (strength=%.1f kJ/mol, features=%d)",
        bias_strength,
        normalized_spec.n_features,
    )
    bias_module = create_cv_bias_potential(
        cv_model=network,
        scaler_mean=scaler_mean,
        scaler_scale=scaler_scale,
        feature_extractor=feature_extractor,
        feature_spec_hash=feature_spec_hash,
        bias_strength=bias_strength,
        feature_names=resolved_feature_names,
    )
    bias_module.eval()

    dummy_pos = torch.zeros(normalized_spec.atom_count, 3, dtype=torch.float32)
    dummy_box = torch.eye(3, dtype=torch.float32)
    with torch.no_grad():
        features = feature_extractor(dummy_pos, dummy_box).unsqueeze(0)
        scaled = (features - torch.as_tensor(scaler_mean)) / torch.as_tensor(
            scaler_scale
        )
        cv_outputs = network(scaled.to(torch.float32))
        if not isinstance(cv_outputs, torch.Tensor):
            raise RuntimeError("DeepTICA model returned a non-tensor output")
        cv_dim = int(cv_outputs.shape[-1])
        _ = bias_module(dummy_pos, dummy_box)

    model_path = output_path / f"{model_name}.pt"
    with torch.inference_mode():
        scripted = torch.jit.script(bias_module)
    optimised = torch.jit.optimize_for_inference(scripted)
    import torch._C as torch_c

    optimised._c._register_attribute(
        "feature_spec_sha256",
        torch_c.StringType.get(),
        feature_spec_hash,
    )
    optimised._c._register_attribute(
        "uses_periodic_boundary_conditions",
        torch_c.BoolType.get(),
        bias_module.uses_periodic_boundary_conditions,
    )
    optimised._c._register_attribute(
        "atom_count",
        torch_c.IntType.get(),
        int(bias_module.atom_count),
    )
    optimised.save(str(model_path))
    logger.info("Exported TorchScript bias module to %s", model_path)

    scaler_path = output_path / f"{model_name}_scaler.npz"
    np.savez(
        scaler_path,
        mean=scaler_mean,
        scale=scaler_scale,
        feature_names=np.array(resolved_feature_names, dtype=object),
    )

    config_path = output_path / f"{model_name}_config.json"
    config_payload = {
        "bias_strength": float(bias_strength),
        "feature_names": resolved_feature_names,
        "cv_dim": cv_dim,
        "input_dim": normalized_spec.n_features,
        "model_class": network.__class__.__name__,
        "feature_spec_sha256": feature_spec_hash,
        "uses_pbc": normalized_spec.use_pbc,
        "atom_count": normalized_spec.atom_count,
    }
    config_path.write_text(
        json.dumps(config_payload, indent=2, sort_keys=True), encoding="utf-8"
    )

    metadata_path = output_path / f"{model_name}_metadata.json"
    metadata_payload = {
        "model_name": model_name,
        "export_dir": str(output_path),
        "feature_names": resolved_feature_names,
        "bias_strength": float(bias_strength),
        "cv_dim": cv_dim,
        "history": history,
        "feature_spec_sha256": feature_spec_hash,
        "feature_spec": spec_payload,
    }
    metadata_path.write_text(
        json.dumps(metadata_payload, indent=2, sort_keys=True), encoding="utf-8"
    )

    return CVModelBundle(
        model_path=model_path,
        scaler_path=scaler_path,
        config_path=config_path,
        metadata_path=metadata_path,
        feature_names=resolved_feature_names,
        cv_dim=cv_dim,
        feature_spec_hash=feature_spec_hash,
    )


def export_cv_model(
    network: nn.Module,
    scaler: Any,
    history: dict[str, Any],
    output_dir: str | Path,
    *,
    feature_spec: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    model_name: str = "deeptica_cv_model",
    bias_strength: float = 10.0,
    feature_names: Optional[list[str]] = None,
) -> CVModelBundle:
    """Convenience wrapper that delegates to :func:`export_cv_bias_potential`."""

    logger.info(
        "Exporting CV model with harmonic expansion bias (strength=%.1f kJ/mol)...",
        bias_strength,
    )
    return export_cv_bias_potential(
        network=network,
        scaler=scaler,
        history=history,
        output_dir=output_dir,
        feature_spec=feature_spec,
        model_name=model_name,
        bias_strength=bias_strength,
        feature_names=feature_names,
    )


def load_cv_model_info(
    bundle_dir: str | Path, model_name: str = "deeptica_cv_model"
) -> dict[str, Any]:
    """Load configuration and metadata from an exported CV model bundle."""

    bundle_path = Path(bundle_dir)

    model_path = bundle_path / f"{model_name}.pt"
    scaler_path = bundle_path / f"{model_name}_scaler.npz"
    config_path = bundle_path / f"{model_name}_config.json"
    metadata_path = bundle_path / f"{model_name}_metadata.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler file not found: {scaler_path}")

    config: dict[str, Any] = {}
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))

    metadata: dict[str, Any] = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    scaler_data = np.load(scaler_path, allow_pickle=True)
    scaler_params = {
        "mean": scaler_data["mean"],
        "scale": scaler_data["scale"],
        "feature_names": scaler_data.get("feature_names", []).tolist(),
    }

    return {
        "model_path": str(model_path),
        "scaler_path": str(scaler_path),
        "config": config,
        "metadata": metadata,
        "scaler_params": scaler_params,
        "feature_spec_sha256": metadata.get("feature_spec_sha256")
        or config.get("feature_spec_sha256"),
    }
